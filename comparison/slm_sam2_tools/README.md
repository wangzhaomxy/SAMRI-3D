# SLM-SAM2 comparison + MedSAM2 rerun (reviewer item P4)

SLM-SAM2 (Chen et al., IEEE TMI 2025; `comparison/SLM-SAM2-main`, unmodified) releases **no
trained weights**: its short-term memory branch only exists after fine-tuning. We therefore
train it with its **official recipe** on **the same training data as SAMRI-3D** and evaluate
it with **the same test protocol** as every other method in `comparison_test.py`.

| | SAMRI-3D | SLM-SAM2 (as run here) |
|---|---|---|
| Init | SAM2.1-Small | SAM2.1-Tiny (only config the authors provide) |
| What is trained | mask decoder + memory modules + GVT (encoders frozen) | everything incl. image encoder (official) |
| Training data | `training_list_3d.pkl`, dice ≥ 0.2 | identical volumes + labels (exported) |
| Slicing | shortest axis, min-max uint8 | identical |
| Test prompt | GT box on the cut slice | GT box on the cut slice (`--slm-prompt box`); GT mask as secondary (`--slm-prompt mask`) |
| Propagation | both directions from the cut slice | official `inference.py` pattern (forward, then reverse, same state), `recent_n = 1` |

## Files

| file | what it does |
|---|---|
| `export_slm_dataset.py` | SAMRI-3D training split → `JPEGImages/<vid>/*.jpg`, `Annotations/<vid>/*.png` (palettised, value = label id), `train.txt`, manifest + per-dataset summary. Resumable. |
| `check_slm_dataset.py` | loads the export through SLM-SAM2's own `PNGRawDataset → RandomUniformSampler → VOSDataset` + official transforms. |
| `train_slm_sam2.py` | launches SLM-SAM2's official `training/train.py` with the unmodified official config, only injecting paths as Hydra overrides; one MIOpen cache dir per DDP rank. |
| `bunya_common.sh` | sourced by the jobs: card → conda env, ROCm env vars, GPU preflight, `cd` to the code root. |
| `slurm_slm_export.sh`, `slurm_slm_train.sh`, `slurm_comparison_eval.sh` | Bunya jobs; **smoke mode by default**. |

`comparison_test.py` changes: new `slm_sam2` runner (`--slm-sam2-ckpt`, `--slm-prompt`,
`--slm-recent-n`); the **MedSAM2 box fix** (box now passed in original pixel coordinates;
it was scaled to 512 twice before, which explains MedSAM2's near-zero Dice); and
`--resume` (results saved after every dataset to `<save-path>.partial.pkl`; finished
datasets are skipped on resubmission; the log is appended).

## 0. Put the code on Bunya (once)

Code root on Bunya: `/scratch/user/s4670484/SAMRI2`. From the Mac:
```bash
cd ~/Library/CloudStorage/OneDrive-TheUniversityofQueensland/PhD/CodePractice/SAMRI-3D
ssh s4670484@bunya.rcc.uq.edu.au "cp /scratch/user/s4670484/SAMRI2/comparison_test.py /scratch/user/s4670484/SAMRI2/comparison_test.py.bak_2026-10"
rsync -av comparison_test.py s4670484@bunya.rcc.uq.edu.au:/scratch/user/s4670484/SAMRI2/
rsync -av --exclude demo --exclude assets --exclude __pycache__ --exclude .ipynb_checkpoints \
  comparison/SLM-SAM2-main comparison/slm_sam2_tools \
  s4670484@bunya.rcc.uq.edu.au:/scratch/user/s4670484/SAMRI2/comparison/
```
Needed on Bunya: `/scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_tiny.pt`,
`/scratch/user/s4670484/training_lists/training_list_3d.pkl`, and the log directory
`/scratch/user/s4670484/SAMRI2_log` (must exist before `sbatch`).

## Cards (from the Bunya settings reference)

| job | card / QoS | env (auto from card) | walltime | why |
|---|---|---|---|---|
| export | CPU, `general` | samri-mi300 | 8 h | no GPU needed |
| train | **MI355X ×1**, `gpu` | `/scratch/user/s4670484/conda/samri-mi355x` | 3 days (cap) | ~2.8× MI300X, never cut; resumable anyway |
| eval | **MI300X ×1**, `gpu` | samri-mi300 | 12 h | env of the earlier SAM2 evaluations; resumable across the ~12 h cut |

Change the card with `--gres=gpu:<card>:1` on the `sbatch` line; the env follows
automatically (override with `--export=ALL,...,ENV=<env>`). 4–5 GPUs do not exist on any
QoS; 6–8 need `--qos=sdf`.

## Run order (all from any directory; scripts `cd` to the code root)

```bash
T=/scratch/user/s4670484/SAMRI2/comparison/slm_sam2_tools
```

**1. MedSAM2 rerun** (independent of everything else — can start now)
```bash
sbatch --time=0-01:00:00 --export=ALL,TARGET=medsam2 $T/slurm_comparison_eval.sh   # smoke
sbatch --export=ALL,TARGET=medsam2,MODE=full $T/slurm_comparison_eval.sh           # full
```

**2. Export** (CPU)
```bash
sbatch $T/slurm_slm_export.sh                       # smoke: 2 volumes per dataset
sbatch --export=ALL,MODE=full $T/slurm_slm_export.sh
```

**3. Train** (after the matching export)
```bash
sbatch --time=0-01:00:00 $T/slurm_slm_train.sh      # smoke: 1 epoch on the smoke export
sbatch --export=ALL,MODE=full $T/slurm_slm_train.sh
```

**4. Evaluate SLM-SAM2** (after training)
```bash
for X in slm_box slm_mask; do sbatch --time=0-01:00:00 --export=ALL,TARGET=$X $T/slurm_comparison_eval.sh; done
for X in slm_box slm_mask; do sbatch --export=ALL,TARGET=$X,MODE=full $T/slurm_comparison_eval.sh; done
```

## What to check in the first minutes (`/scratch/user/s4670484/SAMRI2_log/slurm-*-<jobid>.output`)

Every GPU job prints, before any work:
```
==== ... card=mi355x NGPU=1 CPUS=16
==== env=/scratch/user/s4670484/conda/samri-mi355x | python=... | code=/scratch/user/s4670484/SAMRI2
[PREFLIGHT] .../python | torch 2.13.0+rocm7.2 | hip 7.2... | cuda None
[PREFLIGHT] 1 x AMD Instinct MI355X
```
A `[PREFLIGHT] ...` error stops the job in seconds (missing package → it prints the
`pip install` line; wrong env → "no GPU visible").

- **export**: per-dataset table (26 seen datasets), `errors` 0, last line `CHECK PASSED`.
- **train**: `[PREFLIGHT] training videos: N`; `recent_memory_attention: Load memory
  attention weights successfully!`; then `Train Epoch: [0][k/N] | Batch Time: t |
  Losses/train_all_loss ...` with a finite loss (also in `<EXP>/logs/log.txt`).
  **Judge speed from epoch 1+, not epoch 0** (MIOpen tunes kernels during the first epoch).
  Full-run time ≈ `100 × N × t`. Resume: resubmit the same command; it restarts from
  `<EXP>/checkpoints/checkpoint.pt` (rewritten every epoch).
- **eval**: `[SLM-SAM2] loaded ... recent_n=1 | image_size=1024` (SLM runs);
  `[Done] <dataset>: kept=… errors=0` lines; non-zero Dice in the final table; at the end
  `[ERROR] lines in log: 0`. If cut, resubmit the same command: `[RESUME] partial results
  found` then `[Skip] <dataset>` lines.

Outputs: `/scratch/user/s4670484/Eval_results/comparison/<run>/<run>.pkl` (same format as
the other comparison models) and `<run>/log.txt`.

## Notes
- SLM-SAM2's inference YAML sets `recent_n: 7`, but training uses `recent_n: 1` and the
  model reads its own `recent_n`; the runner forces 1 (`--slm-recent-n`).
- SLM-SAM2's own weight loader is `strict=False` and only logs; the runner instead raises
  if any key is missing, so a plain SAM2 checkpoint cannot be evaluated by mistake.
- The hole-filling post-processing step needs SAM2's compiled `_C` extension; where it is
  missing it is skipped with a warning, as for the other SAM2-based models.
- More than one GPU for training raises the effective batch (fewer optimiser steps) — a
  deviation from the official recipe that would have to be stated in the paper.
