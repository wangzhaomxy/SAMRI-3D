# SAMRI-3D — Segment Anything Model for 3D MRI

Official code for **"SAMRI-3D: Adapting SAM2 for 3D MRI Segmentation with Global Volume
Tokens"** (Wang et al., [arXiv:2607.18014](https://arxiv.org/abs/2607.18014)) — the 3D
successor to [SAMRI](https://github.com/wangzhaomxy/SAMRI), our published 2D MRI model
([arXiv:2510.26635](https://arxiv.org/abs/2510.26635)).

SAMRI-3D adapts Meta's **SAM2** for **3D MRI segmentation**. Following the *slice-as-frame*
paradigm, it treats a 3D volume as a bidirectional "video": the volume is split at the slice
with the largest target foreground (the *cut slice*) into a forward and a backward
sub-sequence, and SAM2's video predictor propagates the segmentation outward from a box/point
prompt placed on the cut slice, before merging the two halves into a 3D mask with the original
NIfTI geometry preserved.

For the MRI domain, both encoders (the Hiera image encoder and the prompt encoder) are
**frozen** and only the lightweight **mask decoder, memory encoder, and memory attention** are
fine-tuned — which alone lifts mean Dice from **0.58** (zero-shot SAM2-Small) to **0.76**.
MRI's low soft-tissue contrast, however, leaves many boundaries effectively invisible on
individual slices, while SAM2's sliding-window memory only carries *local* inter-slice
context. To give every slice **whole-volume** shape context, SAMRI-3D adds **Global Volume
Tokens (GVT)** — compact, persistent memory tokens trained with a **Truncated Signed Distance
Field (TSDF)** shape-prior objective that is *discarded at inference* (zero added cost) —
reaching **0.78** mean Dice with no drop on held-out datasets.

![SAMRI-3D pipeline overview](assets/samri3d_pipeline.png)

*Overall SAMRI-3D pipeline. A 3D MRI volume is split at the cut slice (largest foreground)
into forward and backward sub-sequences, each processed by the SAMRI-3D video predictor with
GVT injection, then merged into the full 3D segmentation. Global Volume Tokens summarize the
entire volume and are injected into SAM2's memory bank; a training-only TSDF shape-prior
head supervises them and is discarded at inference.*

> For deep implementation notes and internal design decisions, see [`CLAUDE.md`](CLAUDE.md).

---

## Highlights

- **SAMRI-3D-Bench** — the largest MRI-only SAM2 benchmark: 10,392 volumes, 34 datasets,
  12 anatomical domains, with seen/unseen splits (0.78 mean Dice, no generalization gap).
- **Parameter-efficient fine-tuning** — both encoders frozen; only the mask decoder + memory
  modules trained, on pre-computed SAM2 embeddings (fast, memory-efficient).
- **Global Volume Tokens (GVT)** with a **TSDF shape-prior** objective — whole-volume context
  injected into SAM2's memory bank; the auxiliary decoder is discarded at inference (zero
  overhead).
- **Unified 2D/3D training** controlled by a single `--mix-ratio-3d` parameter.
- **3D evaluation** (per-label cut slices; Dice / Hausdorff / MSD) plus a comparison harness
  for prior SAM-based medical models (SAM-Med3D, Medical-SAM2, SAMed-2; MedSAM2 & MA-SAM
  runners also included).
- **Publication-quality qualitative figures** (8 visualisation methods incl. TSDF panels,
  one folder each).

---

## SAMRI-3D-Bench

SAMRI-3D is evaluated on **SAMRI-3D-Bench**, the largest MRI-only SAM2 benchmark to date:
**10,392 volumes** from **34 datasets** (27 public, 7 in-house) spanning **12 anatomical
domains** (brain, prostate, knee/MSK, cardiac, whole-body, hippocampus, vestibular, hip,
spine, abdominal, shoulder, gynaecology) and **10+ MRI sequences**. Each dataset is split
**80/10/10** train/val/test; models train only on the **26 seen** datasets, and the **8
unseen** datasets are held out entirely for zero-shot generalization testing.

## Key results

Mean over the 34-dataset benchmark (box prompt, mean ± std of per-dataset medians).
Per-domain, per-sequence, and significance statistics are in the paper.

| Method | Type | Dice ↑ | HD ↓ | MSD ↓ |
|--------|------|:------:|:----:|:-----:|
| SAM-Med3D | 3D medical SAM | 0.37 ± 0.21 | 29.04 ± 16.70 | 6.39 ± 2.78 |
| Medical-SAM2 | SAM2, medical | 0.49 ± 0.22 | 33.51 ± 21.81 | 5.92 ± 4.26 |
| SAM2-Small | foundation (zero-shot) | 0.58 ± 0.24 | 28.24 ± 20.42 | 6.32 ± 4.46 |
| SAMed-2 | SAM2, medical | 0.69 ± 0.15 | 22.28 ± 18.82 | 2.91 ± 1.62 |
| **SAMRI-3D** | ours (best) | **0.78 ± 0.14** | **13.85 ± 9.01** | **2.11 ± 0.96** |

SAMRI-3D attains the best mean **0.78 ± 0.14 Dice, 13.85 ± 9.01 HD, 2.11 ± 0.96 MSD**, with
the lowest variance on every metric. All gains over the comparison methods and zero-shot
SAM2-Small are significant at the per-sample and per-label levels (two-sided paired Wilcoxon).
SAMRI-3D shows **no generalization gap** — 0.79 on the 8 unseen datasets vs. 0.78 on the seen
ones — and leads **10 of 12** anatomical domains (trailing SAMed-2 only on whole-body /
abdominal multi-organ cases). The whole-volume shape prior helps most where per-slice contrast
is weakest, e.g. white-matter hyperintensities improve 0.09 → 0.54 and hippocampus 0.65 → 0.84.

![Qualitative comparison](assets/samri3d_qualitative.png)

*Qualitative comparison (triplanar axial/coronal/sagittal views) of SAMRI-3D against
zero-shot SAM2-Small, SAM-Med3D, Medical-SAM2, and SAMed-2. Yellow = ground truth,
magenta = prediction. Figures are produced by [`qualitative_study.py`](qualitative_study.py).*

---

## Repository layout

```
SAMRI2/
├── sam2/                       Vendored Meta SAM2 package (model + configs)
├── training/                   SAM2Train backbone (Hydra/DDP training system)
├── comparison/                 Vendored comparison-model code (each needs its own env)
├── gvt/                        Global Volume Token module
│   ├── gvt_module.py           GVTModule, GVTReconDecoder, TSDFDecoder
│   ├── gvt_integration.py      Wraps SAM2 memory_attention with GVT tokens
│   ├── gvt_inference.py        GVT token computation at inference
│   └── train_gvt_emb_tsdf.py   Canonical GVT training (v1/v2/v3)
├── gui/                        Interactive Jupyter GUI (ipywidgets + ipycanvas)
│   ├── samri3d_gui.py           GUI app
│   ├── inference_backend.py    Box/point-prompted bidirectional propagation
│   └── SAMRI3D_GUI.ipynb        Notebook launcher
├── utils/
│   ├── dataloader.py           NiiDataset / Filtered{,3D}NiiDataset
│   ├── losses.py               Dice / Hausdorff / MSD
│   ├── tsdf_utils.py           TSDF computation + boundary-weighted loss
│   └── gen_training_list.py    Build dice-scored training lists
├── generate_3d_embeddings.py   Pre-compute SAM2 embeddings for 3D volumes
├── train_unified_emb.py        Train from embeddings (2D loss) — baseline
├── train_unified_emb_3dloss.py Train from embeddings (2D + 3D volumetric loss)
├── train_unified.py            Train from raw images / training list (2D loss)
├── test_vis3d_multi.py         3D evaluation (per-label cut, metrics)
├── comparison_test.py          Evaluate comparison models on the same test data
├── qualitative_study.py        Render qualitative overlay / 3D figures
├── 3d_inference.py             Batch 3D inference without metrics
├── train_multi_gpus.sh         SLURM launcher for all training modes
└── setup.py
```

---

## Installation

```bash
git clone <repo-url> SAMRI2 && cd SAMRI2
conda create -n samri python=3.10 -y && conda activate samri
pip install -e .          # installs the `samri` package + dependencies (see setup.py)
```

- **SAM2** is vendored under `sam2/` — no separate install.
- **PyTorch** with a CUDA build matching your GPU should be installed separately
  (follow the [official instructions](https://pytorch.org/)).
- **SAMRI-3D pretrained weights** — download the trained checkpoint from
  [Google Drive](https://drive.google.com/drive/folders/1LuKt_l-4w8Lbn6ZBFo6uSxkFXAcEU4kg?usp=sharing)
  and pass it via `--ckpt-path ... --use-gvt` for evaluation/inference.
- **Pretrained SAM2.1 checkpoints** (`sam2.1_hiera_{tiny,small,base_plus,large}.pt`), needed
  only to *train* from scratch, come from the
  [official SAM2 repository](https://github.com/facebookresearch/sam2)
  (passed via `--pretrained-ckpt`).
- **Comparison models** (`comparison/`) each require their **own conda environment** and
  checkpoints; see `comparison_test.py` for per-model setup.

The model config is referenced as `configs/sam2.1/sam2.1_hiera_s.yaml` (resolved through
the vendored `sam2` package); available variants: `_t` (tiny), `_s` (small), `_b+`
(base+), `_l` (large). SAMRI-3D fine-tunes the **small** backbone by default.

---

## Data format

- Images and masks live in the **same folder**.
- Naming: `*_img_*` for images, `*_gt_*` (or `*_seg_*`) for ground-truth masks.
- Supported formats: `.nii.gz`, `.nii`, `.mha`, `.mhd`, `.nrrd`.
- Test data is auto-detected in `<dataset>/testing/` or scanned directly.
- Masks are multi-label integer volumes; each non-zero id is segmented independently.

**Image normalization (important):** all images are normalized with plain **min-max**
scaling to `uint8 [0, 255]` before SAM2 (which then applies ImageNet normalization
internally). Do **not** use percentile clipping — it breaks train/inference consistency.

---

## Workflow

### 1. Pre-compute 3D embeddings (for embedding-based training)

```bash
python generate_3d_embeddings.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/sam2.1_hiera_small.pt \
  --data-paths /path/to/Datasets/ \
  --save-dir /path/to/3D_embeddings/
```

### 2. Train SAMRI-3D

SAMRI-3D fine-tunes the SAM2.1 Hiera-Small backbone from pre-computed embeddings (both
encoders frozen) with the GVT + TSDF shape-prior objective:

```bash
python gvt/train_gvt_emb_tsdf.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --pretrained-ckpt /path/to/sam2.1_hiera_small.pt \
  --embedding-dir /path/to/3D_embeddings/ \
  --save-path /path/to/checkpoints/ \
  --batch-size 1 --num-epochs 200 --lr 5e-6 --weight-decay 0.01 \
  --lambda-2d 1.0 --lambda-3d 5.0 --mix-ratio-3d 0.8 --max-slices-per-video 32 \
  --use-gvt --gvt-num-tokens 16 --gvt-num-layers 2 --gvt-num-heads 4 \
  --gvt-recon --gvt-recon-res 32 --lambda-aux 1.0 \
  --use-tsdf --lambda-tsdf 1.0 --tsdf-tau 32.0 --tsdf-alpha 4.0 \
  --use-amp
```

Or launch on SLURM via `train_multi_gpus.sh` (set `TRAINING_MODE="gvt_v3"`), which wraps the
command above; the script also exposes the reduced configurations used for the paper's
ablation study.

```bash
sbatch train_multi_gpus.sh
```

Key knobs: `--mix-ratio-3d` (0 = pure 2D slices … 1 = pure 3D volumes) and `--lambda-tsdf`
(weight of the TSDF shape-prior loss).

### 3. Evaluate (Dice / HD / MSD)

```bash
# SAMRI-3D (add --use-gvt for GVT checkpoints; GVT config auto-detected from ckpt)
python test_vis3d_multi.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pth \
  --dataset-path /path/to/test_data/ /path/to/zeroshot_data/ \
  --save-path /path/to/results.pkl \
  --num-workers 12 --use-gvt
```

### 4. Comparison models

Each baseline runs in its **own conda environment**; checkpoints live under
`<weights-dir>/<model>/`.

```bash
python comparison_test.py \
  --model medsam2 \                # sam_med3d | medsam2 | medical_sam2 | samed2 | ma_sam
  --dataset-path /path/to/test_data/ /path/to/zeroshot_data/ \
  --weights-dir /path/to/comparison_ckpt/ \
  --save-path /path/to/results_medsam2.pkl
```

### 5. Qualitative figures

Runs inference once, saves per-case `.nii.gz` under `pred_volumes/`, then renders 7
visualisation methods (each into its own `visN_*` folder): cut slice, triplanar, montage,
MIP, 3D surface, surface-distance heatmap, and FP/FN error volume.

```bash
# Full run (inference + all figures)
python qualitative_study.py \
  --samples-json /path/to/qualitative_samples.json \
  --output-dir /path/to/qualitative/ \
  --models sam2_small sam_med3d medical_sam2 samed2 baseline gvt_v1 gvt_v2 gvt_v3

# Re-render from saved volumes only (no inference); subset models/methods
python qualitative_study.py \
  --from-volumes /path/to/qualitative/pred_volumes \
  --output-dir /path/to/qualitative_v2/ \
  --models sam2_small baseline gvt_v3 --vis-methods vis2 vis5
```

---

## Interactive GUI

> **🚧 Under construction.** The GUI is experimental and not yet feature-complete — expect
> rough edges and API changes.

For point-and-click segmentation (the 3D analogue of [SAMRI v1's](https://github.com/wangzhaomxy/SAMRI)
ipywidgets demo), a Jupyter GUI is provided under `gui/`:

```bash
conda activate samri          # on a machine with a CUDA GPU
jupyter notebook gui/SAMRI3D_GUI.ipynb     # then "Run all cells"
```

or from any notebook cell:

```python
from gui import launch
app = launch(device="cuda")
```

Workflow: **load a volume → scrub slices → draw a bounding box and/or click +/- points on
one slice → load a SAMRI-3D checkpoint (tick "GVT" for GVT models) → Segment** (SAM2
propagates the prompt through the whole volume) **→ toggle the overlay and save as NIfTI.**
Built with `ipywidgets` + `ipycanvas` (already in `setup.py`); use **Jupyter Notebook**
rather than JupyterLab unless the matching labextensions are installed. The GUI reuses the
exact propagation pipeline from `test_vis3d_multi.py` via `gui/inference_backend.py`.

---

## Citation

If you use SAMRI-3D or SAMRI-3D-Bench, please cite our paper
([arXiv:2607.18014](https://arxiv.org/abs/2607.18014)):

```bibtex
@misc{wang2026samri3dadaptingsam23d,
  title         = {SAMRI-3D: Adapting SAM2 for 3D MRI Segmentation with Global Volume Tokens},
  author        = {Zhao Wang and Wei Dai and Hongfu Sun and Craig Engstrom and Shekhar S. Chandra},
  year          = {2026},
  eprint        = {2607.18014},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.18014}
}
```

Please also cite the 2D predecessor **SAMRI** and Meta's **SAM2**:

```bibtex
@misc{wang2025samri,
  title         = {SAMRI: Segment Any MRI},
  author        = {Zhao Wang and Wei Dai and Thuy Thanh Dao and Steffen Bollmann and Hongfu Sun and Craig Engstrom and Shekhar S. Chandra},
  year          = {2025},
  eprint        = {2510.26635},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  url           = {https://arxiv.org/abs/2510.26635}
}

@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and others},
  journal = {arXiv preprint arXiv:2408.00714},
  year    = {2024}
}
```

## Contact

- **Zhao Wang** — zhao.wang1@uq.edu.au
- **Shekhar S. Chandra** — shekhar.chandra@uq.edu.au

School of Electrical Engineering and Computer Science, The University of Queensland.

## Acknowledgements & license

SAMRI-3D builds on Meta AI's [SAM2](https://github.com/facebookresearch/sam2) and the
[MedSAM](https://github.com/bowang-lab/MedSAM) line of open-source medical SAM adaptations,
and was developed with the support of the University of Queensland Research Computing
Centre's **Bunya** supercomputer. We thank the MRI research community for the public
datasets that make up SAMRI-3D-Bench.

The `sam2/` and `training/` directories are vendored from the SAM2 release; `comparison/`
contains third-party model code under their respective licenses. SAMRI-3D code is released
under the **Apache-2.0 License**.
