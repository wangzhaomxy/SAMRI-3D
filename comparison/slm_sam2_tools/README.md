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

Files

| file | what it does |
|---|---|
| `export_slm_dataset.py` | SAMRI-3D training split → `JPEGImages/<vid>/*.jpg`, `Annotations/<vid>/*.png` (palettised, value = label id), `train.txt`, manifest + per-dataset summary. Resumable. |
| `check_slm_dataset.py` | loads the export through SLM-SAM2's own `PNGRawDataset → RandomUniformSampler → VOSDataset` + official transforms. |
| `train_slm_sam2.py` | launches SLM-SAM2's official `training/train.py` with the unmodified official config, only injecting paths as Hydra overrides. |
| `slurm_slm_export.sh`, `slurm_slm_train.sh`, `slurm_comparison_eval.sh` | Bunya jobs (smoke mode by default). |

`comparison_test.py` changes: new `slm_sam2` runner (`--slm-sam2-ckpt`, `--slm-prompt`,
`--slm-recent-n`), and the **MedSAM2 box fix** (box now passed in original pixel
coordinates; it was scaled to 512 twice before, which explains MedSAM2's near-zero Dice).

## Run order on Bunya (all `sbatch` from the SAMRI-3D repo root)

Prerequisite: `/scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_tiny.pt`
(the official SAM2.1 Tiny weights; the train job's preflight stops if it is missing).

**1. Export — smoke, then full**
```bash
sbatch comparison/slm_sam2_tools/slurm_slm_export.sh                     # 2 vols/dataset
sbatch --export=ALL,MODE=full comparison/slm_sam2_tools/slurm_slm_export.sh
```
Check in the `.output` log: the per-dataset table lists the 26 seen datasets, `errors` = 0,
and the last line of the loader check is `CHECK PASSED`.

**2. Train — smoke (1 epoch on the smoke export), then full**
```bash
sbatch --time=0-01:00:00 comparison/slm_sam2_tools/slurm_slm_train.sh
sbatch --export=ALL,MODE=full comparison/slm_sam2_tools/slurm_slm_train.sh
```
First-minutes checks (`.output` / `<EXP>/logs/log.txt`):
- `[PREFLIGHT] ... GPUs visible: 1 | AMD Instinct MI300X` and `training videos: N`
- `recent_memory_attention: Load memory attention weights successfully!` (the short-term
  branch is initialised from SAM2's memory attention, as in the paper)
- `Train Epoch: [0][k/N] | Batch Time: t` lines with a finite, falling `train_all_loss`.

Full-run time ≈ `100 × N × t / num_gpus`. The official recipe is 1 GPU, batch 1. The
trainer resumes from `<EXP>/checkpoints/checkpoint.pt`, so if the 4-day limit is hit,
resubmit the same command. Using more GPUs raises the effective batch (fewer optimiser
steps) — a deviation from the official recipe that would need to be stated.

**3. Evaluate — smoke (1 case/dataset), then full; one job per target**
```bash
for T in medsam2 slm_box slm_mask; do
  sbatch --time=0-01:00:00 --export=ALL,TARGET=$T comparison/slm_sam2_tools/slurm_comparison_eval.sh
done
# after the smoke logs look right:
for T in medsam2 slm_box slm_mask; do
  sbatch --export=ALL,TARGET=$T,MODE=full comparison/slm_sam2_tools/slurm_comparison_eval.sh
done
```
MedSAM2 does not depend on the SLM-SAM2 training, so it can run immediately. If the
earlier MedSAM2 run used a different conda env, add `COMP_ENV=<env>` to `--export`.
First-minutes checks: `[SLM-SAM2] loaded ... recent_n=1 | image_size=1024`;
`[Done] <dataset>: kept=… errors=0` lines; non-zero Dice in the summary table.
Outputs: `/scratch/user/s4670484/Eval_results/comparison/<run>/<run>.pkl` (same format
as the other comparison models).

## Notes
- SLM-SAM2's inference YAML sets `recent_n: 7`, but training uses `recent_n: 1` and the
  model reads its own `recent_n`; the runner forces 1 (`--slm-recent-n`).
- SLM-SAM2's own weight loader is `strict=False` and only logs; the runner instead raises
  if any key is missing, so a plain SAM2 checkpoint cannot be evaluated by mistake.
- The hole-filling post-processing step needs SAM2's compiled `_C` extension; where it is
  missing it is skipped with a warning, as for the other SAM2-based models.
