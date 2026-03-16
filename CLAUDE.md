# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAMRI2 (Segment Anything Model for MRI - Version 2) adapts Meta's SAM2 for medical imaging, specifically 3D MRI segmentation using decoder-only fine-tuning on pre-computed embeddings.

## Common Commands

### Installation
```bash
pip install -e .                    # Install SAMRI package
python setup_sam2.py install        # Install SAM2 with CUDA extensions
```

### Running Evaluation
```bash
# 3D evaluation (per-label cut, semantic volume output)
python test_vis3d_multi.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --dataset-path /path/to/test_data/ /path/to/zeroshot_results/ \
  --save-path /path/to/results.pkl \
  --prompt-mode box \
  --num-workers 8

# 3D evaluation with Dense CRF post-processing
python test_vis3d_crf.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --dataset-path /path/to/test_data/ /path/to/zeroshot_results/ \
  --save-path /path/to/results_crf.pkl \
  --num-workers 8 \
  --use-crf

# 3D evaluation with two-pass auto-context refinement (topk/allconf/revstyle)
python test_vis3d_twopass.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --dataset-path /path/to/test_data/ /path/to/zeroshot_results/ \
  --save-path /path/to/results_topk5.pkl \
  --num-workers 8 \
  --two-pass topk --two-pass-k 5

# 2D slice evaluation
python test_vis2d.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --test-file-paths /path/to/datasets/ \
  --save-path /path/to/results.pkl \
  --device cuda

# 3D inference without metrics
python 3d_inference.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --dataset-path /path/to/dataset/ \
  --save-root /path/to/output/ \
  --prompt-mode box
```

### Generating Training Lists (for filtered training)
```bash
# Generate 2D training list with dice scores
python gen_training_list.py \
  --mode 2d \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --data-paths /path/to/datasets/ \
  --save-path /path/to/training_list_2d.pkl \
  --num-workers 12

# Generate 3D training list with dice scores
python gen_training_list.py \
  --mode 3d \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --data-paths /path/to/datasets/ \
  --save-path /path/to/training_list_3d.pkl \
  --num-workers 16
```

### Training

Three training modes are available via `train_multi_gpus.sh`:

```bash
# 2D Loss from Embeddings (2d_emb) - Default, fastest
# Uses: train_unified_emb.py
python train_unified_emb.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --pretrained-ckpt /path/to/pretrained.pth \
  --embedding-dir /path/to/3D_embeddings/ \
  --save-path /path/to/checkpoints/ \
  --batch-size 2 --num-epochs 50 --lr 5e-6 \
  --mix-ratio-3d 0.8 --max-slices-per-video 32 \
  --use-amp

# 2D + 3D Loss from Embeddings (3d_emb) - Combined volumetric loss
# Uses: train_unified_emb_3dloss.py
python train_unified_emb_3dloss.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --pretrained-ckpt /path/to/pretrained.pth \
  --embedding-dir /path/to/3D_embeddings/ \
  --save-path /path/to/checkpoints/ \
  --batch-size 2 --num-epochs 50 --lr 5e-6 \
  --lambda-2d 1.0 --lambda-3d 0.5 \
  --mix-ratio-3d 0.8 --max-slices-per-video 32 \
  --use-amp

# GVT v3: 2D + 3D + Recon + TSDF Shape Prior (gvt_tsdf) - Full GVT v3 training
# Uses: gvt/train_gvt_emb_tsdf.py
python gvt/train_gvt_emb_tsdf.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --pretrained-ckpt /path/to/pretrained.pth \
  --embedding-dir /path/to/3D_embeddings/ \
  --save-path /path/to/checkpoints/ \
  --batch-size 2 --num-epochs 50 --lr 5e-6 \
  --lambda-2d 1.0 --lambda-3d 0.5 \
  --mix-ratio-3d 0.8 --max-slices-per-video 32 \
  --use-gvt --gvt-num-tokens 32 --gvt-num-layers 2 --gvt-num-heads 4 \
  --gvt-recon --gvt-recon-res 32 --lambda-aux 1.0 --lambda-aux-warmup-epochs 10 \
  --use-tsdf --lambda-tsdf 5.0 --tsdf-tau 32.0 --tsdf-alpha 4.0 \
  --use-amp

# 2D Loss from Raw Images (2d_img) - No pre-computation needed
# Uses: train_unified.py
python train_unified.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --pretrained-ckpt /path/to/pretrained.pth \
  --training-list /path/to/training_list_3d.pkl \
  --save-path /path/to/checkpoints/ \
  --batch-size 2 --num-epochs 50 --lr 5e-6 \
  --mix-ratio-3d 0.8 --max-slices-per-video 32 \
  --use-amp
```

**Loss Configurations:**
- `--lambda-2d`: Weight for 2D per-slice Dice-Focal loss (default: 1.0)
- `--lambda-3d`: Weight for 3D volumetric Dice loss (default: 0.5)
- Set `--lambda-3d 0.0` for 2D-only training (ablation)

**2D/3D Mix Ratio (Key Ablation Parameter):**
- `--mix-ratio-3d 0.0`: Pure 2D slice training (equivalent to legacy 2D-only training)
- `--mix-ratio-3d 0.8`: Default — 80% 3D volumetric, 20% 2D slice batches
- `--mix-ratio-3d 1.0`: Pure 3D volumetric training (no 2D batches)
- Varying this ratio provides a clean ablation of the 2D→3D training contribution

**SLURM submission:**
```bash
# Edit TRAINING_MODE in train_multi_gpus.sh: "2d_emb", "3d_emb", "2d_img", or "gvt_tsdf"
sbatch train_multi_gpus.sh
```

## Architecture

### Core Processing Flow
1. 3D volumes are split into two "videos" around the cut slice (slice with largest foreground)
2. SAM2 video predictor propagates segmentation bidirectionally from the cut slice
3. Predictions are merged back into 3D semantic masks with original NIfTI metadata preserved

### Key Components

**Root scripts:**
- `3d_inference.py` - Batch 3D inference without evaluation
- `test_vis3d_multi.py` - 3D evaluation (per-label cut, semantic volume output)
- `test_vis3d_crf.py` - 3D evaluation with Dense CRF post-processing (experimental)
- `test_vis3d_twopass.py` - 3D evaluation with two-pass auto-context refinement (experimental)
- `test_vis2d.py` - 2D slice evaluation
- `gen_training_list.py` - Generate training lists with dice scores for filtered training
- `generate_sam2_embeddings.py` - Pre-compute SAM2 image embeddings for 2D training
- `generate_3d_embeddings.py` - Pre-compute SAM2 embeddings for 3D volumes
- `train_unified.py` - Unified training from raw images (2D loss)
- `train_unified_emb.py` - Unified training from pre-computed embeddings (2D loss)
- `train_unified_emb_3dloss.py` - Unified training from embeddings (2D + 3D volumetric loss)
- `gvt/train_gvt_emb_tsdf.py` - GVT unified training script: supports v1 (GVT encoder only), v2 (+recon loss), and v3 (+TSDF shape prior). This is the canonical GVT training script — `gvt/train_gvt_emb_3dloss.py` has been merged into this file and deleted.
- `utils/tsdf_utils.py` - TSDF computation, target preparation, and boundary-weighted loss

**sam2/** - Meta's SAM2 package with:
- `modeling/` - Hiera backbone, memory attention, mask decoder
- `sam2_video_predictor.py` - Frame-to-frame propagation interface
- `configs/` - YAML configs for model variants (Tiny/Small/Base+/Large)

**training/** - Training system with:
- `model/sam2.py` - SAM2Train class with point/box prompt sampling
- `trainer.py` - Hydra-based DDP training with AMP support
- `dataset/` - VOSDataset loaders for video object segmentation

**utils/** - SAMRI-specific utilities:
- `dataloader.py` - Dataset classes:
  - `NiiDataset` - Load NIfTI files for 2D slice training
  - `FilteredNiiDataset` - Load filtered 2D samples by dice threshold
  - `Filtered3DNiiDataset` - Load filtered 3D volumes by dice threshold
- `losses.py` - Dice, Hausdorff Distance, Mean Surface Distance metrics

### Dataset Conventions
- Images and masks must be in the same folder
- Naming: `*_img_*` for images, `*_seg_*` (or `*_gt_*`) for ground truth
- Supported formats: .nii.gz, .nii, .mha, .mhd, .nrrd
- Test data auto-detected in `<dataset>/testing/` or scanned directly

### Prompt Modes
- `box` - Bounding box prompts derived from ground truth
- `mask` - Dense point sampling from ground truth masks

### Image Preprocessing (IMPORTANT)
All images must be normalized using **plain min-max scaling** before being fed into SAM2:

```python
def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 range using plain min-max scaling."""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min < 1e-8:
        img_max = img_min + 1e-8
    img = (img - img_min) / (img_max - img_min)
    return (img * 255.0).astype(np.uint8)
```

**Why plain min-max?**
- SAM2's `SAM2Transforms` expects uint8 [0, 255] input, then applies `ToTensor()` (scales to [0,1]) followed by ImageNet normalization
- Using percentile-based normalization (e.g., clipping at 0.5th and 99.5th percentiles) creates inconsistency between training and inference
- All SAMRI2 scripts use plain min-max to ensure consistency across training, validation, and inference

**Do NOT use percentile-based normalization** like:
```python
# WRONG - do not use this
lo = np.percentile(img, 0.5)
hi = np.percentile(img, 99.5)
img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
```

### Per-Label Cut Evaluation State-Reuse Pattern (test_vis3d_multi.py)

`test_vis3d_multi.py` runs one propagation per label using a per-label cut
slice (the slice with maximum foreground for that specific label).  To avoid
re-running the image encoder (O(N) cost) for every label, all N slices are
pre-encoded once and injected into each label's inference state.

**Bug fixes (this session):**
- **Per-label exception isolation**: A single label's inference failure previously propagated to the outer handler and dropped the entire test case (causing model-dependent sample counts). Fixed: each label is wrapped in `except Exception` so failures skip that label only, not the whole case.
- **VRAM memory leak in `precompute_slice_features`**: This function called `init_state_compat` to borrow a video state for N forward_image calls, but never called `reset_state` afterward. The leaked N-frame state (~27 MB/slice × N slices) accumulated across cases. Fixed: `predictor.reset_state(state)` is now called in a `finally` block.

```python
# Pre-encode all N slices once; keyed by slice index
slice_cache = precompute_slice_features(predictor, vol0)

# Per-label: inject relevant features before init_state + propagate
inject_cached_features(state, v_slices, slice_cache)
```

The resulting per-case loop is:
```
precompute_slice_features  (N forward_image calls, once per case)
for each label:
    per-label cut_L = argmax foreground count
    build v1_L, v2_L around cut_L
    init_state(v1_L) -> inject_cached_features -> prompt -> propagate -> reset_state
    init_state(v2_L) -> inject_cached_features -> prompt -> propagate -> reset_state
    layer binary result into semantic volume
```

Image encoding cost: **N** forward_image calls total (not N*K).
`init_state` cost: **2*K** calls (one pair per label), but no disk I/O when
streaming is available — sources are in-memory PIL loaders.

### Dense CRF Post-Processing (test_vis3d_crf.py)

`test_vis3d_crf.py` extends `test_vis3d_multi.py` by applying optional Dense CRF
refinement to SAM2's soft probability outputs **after** propagation, **before**
thresholding into the final binary mask.

**CRF backend priority (auto-detected at runtime):**
1. **SimpleCRF** (`pip install SimpleCRF`) — true 3D CRF, processes the full volume
2. **pydensecrf** (`pip install pydensecrf`) — 2D slice-by-slice fallback
3. **Sigmoid threshold** — if neither library is available

**New CLI flags:**
- `--use-crf` — enable CRF post-processing (default: off)
- `--crf-sigma-spatial` — spatial Gaussian kernel width (default: 5.0)
- `--crf-sigma-intensity` — bilateral intensity kernel width (default: 0.1)
- `--crf-sigma-smooth` — smoothness kernel width (default: 1.0)
- `--crf-w-bilateral` — bilateral energy weight (default: 5.0)
- `--crf-w-smooth` — smoothness energy weight (default: 3.0)
- `--crf-iterations` — CRF mean-field iterations (default: 5)

**Sigmoid overflow (IMPORTANT):**
SAM2's raw logits can have very large magnitude. Using `1.0 / (1.0 + np.exp(-logit))`
on `float32` arrays overflows at ±88 and produces `RuntimeWarning`. The fix used
throughout this script is `scipy.special.expit`, which is numerically stable:

```python
from scipy.special import expit
prob_vol = expit(logit_vol.astype(np.float64)).astype(np.float32)
```

The overflow itself is benign (results in 0.0 or 1.0, which are mathematically correct)
but the warning pollutes logs and indicates float32 precision loss.

### Two-Pass Auto-Context Refinement (test_vis3d_twopass.py)

`test_vis3d_twopass.py` extends `test_vis3d_multi.py` with a second propagation pass
that uses Pass 1's high-confidence predictions as additional prompted frames.

**Two-pass variants (`--two-pass`):**
- `none` — baseline, identical to `test_vis3d_multi.py`
- `topk` — re-prompt Pass 2 with the K frames of highest `object_score_logits`
- `allconf` — re-prompt Pass 2 with all frames where sigmoid(score) ≥ threshold
- `revstyle` — add high-quality anchor frames that bookend low-quality regions

**New CLI flags:**
- `--two-pass {none|topk|allconf|revstyle}` (default: none)
- `--two-pass-k` — K for topk mode (default: 5)
- `--two-pass-conf-threshold` — confidence threshold for allconf/revstyle (default: 0.7)

**Feature-reuse pattern (critical for inference speed):**

Naively, two-pass doubles the image encoder cost (2N forward_image calls vs N for
single-pass). The implementation avoids this by:

1. **Prefill `cached_features` before Pass 1** — `_prefill_features(predictor, state)`
   runs `forward_image` on every frame and writes all results into
   `state["cached_features"]` as a full dict. During propagation, `_get_image_feature`
   finds every frame in the dict (cache hit) and never replaces it.

2. **`reset_state` preserves `cached_features`** — After Pass 1, `predictor.reset_state(state)`
   clears all object/tracking state but does **not** touch `cached_features`. Pass 2
   can re-prompt and re-propagate with zero additional encoder calls.

3. **Sequential per-video processing** — v1 is fully processed (P1 → P2) before v2,
   so only one video's feature cache is in GPU memory at a time:

```
init_state(v1) → prefill(v1) → prompt → Pass1-propagate
   → reset_state(v1) → re-prompt → Pass2-propagate → del(v1)
init_state(v2) → prefill(v2) → prompt → Pass1-propagate
   → reset_state(v2) → re-prompt → Pass2-propagate → del(v2)
```

**Encoding cost: N calls total** (same as single-pass baseline).
**Memory: one video's feature cache at a time** (same peak as baseline).

**`object_score_logits` as quality proxy:**
The per-frame confidence used for frame selection is `object_score_logits`, a `(1,1)`
scalar stored in `state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"|"non_cond_frame_outputs"][frame_idx]`.
Higher values indicate the model is more confident the object is present and
well-segmented in that frame. `select_allconf_frames` and `select_revstyle_frames`
apply sigmoid to convert to a [0,1] confidence using `scipy.special.expit` to avoid
overflow warnings.

## Reference Materials (backup/)

The `backup/` folder contains Meta's original SAM2 files as reference examples. This folder will be removed after project finalization.

### Notebooks (`backup/notebooks/`)
- **video_predictor_example.ipynb** - Most relevant for SAMRI2; shows `SAM2VideoPredictor` workflow: initialize state, add prompts on any frame, propagate through video, multi-object tracking
- **image_predictor_example.ipynb** - Single-image segmentation with point/box prompts, multi-mask output, batched inference
- **automatic_mask_generator_example.ipynb** - Fully automatic segmentation using grid-based point sampling

### Demo Application (`backup/demo/`)
- **Frontend** (`demo/frontend/`): React + TypeScript + Vite app with video player, interaction layers, mask visualization
- **Backend** (`demo/backend/server/`): Flask + GraphQL with `InferenceAPI` class in `inference/predictor.py` showing production inference patterns, session state management, device auto-detection

### SAMRI v1 Training Script (`backup/train_in_batch_multi_gpu.py`)
Reference implementation for decoder-only training on pre-computed embeddings:
- Trains only mask decoder (freezes image encoder and prompt encoder)
- Uses `EmbDataset` / `BalancedEmbDataset` for pre-computed embeddings
- Multi-GPU DDP training via `torch.multiprocessing.spawn`
- Mixed prompts (point + bbox) with `DiceFocalLoss`
- AdamW optimizer: lr=1e-5, weight_decay=0.1

### Other References
- `checkpoints/download_ckpts.sh` - Download script for SAM2.1 model checkpoints (tiny/small/base_plus/large)
- `INSTALL.md` - Installation guide with troubleshooting for CUDA, Flash Attention, Windows issues
- `RELEASE_NOTES.md` - Version history documenting SAM2 -> SAM2.1 improvements

## SAM2 Training Strategy Reference

### Trainable Components (5 total)
| Component | Trainable | Typical Strategy |
|-----------|-----------|------------------|
| Image Encoder | Yes | Freeze (or LoRA for efficiency) |
| Prompt Encoder | Yes | **Always freeze** - spatial encoding is domain-invariant |
| Mask Decoder | Yes | **Primary training target** |
| Memory Encoder | Yes | Train for video/3D tasks |
| Memory Attention | Yes | Train for video/3D tasks |

### Image vs Video Training
- **Image training**: Memory bank empty, model behaves like original SAM
- **Video training**: 8-frame sequences, 1-2 frames randomly prompted, 50% temporal reversal for bidirectional learning
- **Joint training**: Alternating batches (~30-40% images, ~60-70% videos) proportional to dataset size

### SAMRI2 Unified Training Pipeline (3D MRI as Bidirectional Videos)

**Core approach:**
1. Split 3D volume at "central slice" (largest foreground) into two sequences
2. Treat each sequence as a "video" with slices as frames
3. Propagate segmentation bidirectionally from central slice

**Unified training** uses a single pipeline controlled by `--mix-ratio-3d`:
- `--mix-ratio-3d 0.0` → Pure 2D slice training (mask decoder only, memory inactive)
- `--mix-ratio-3d 0.8` → Default joint training (80% 3D volumes, 20% 2D slices)
- `--mix-ratio-3d 1.0` → Pure 3D volumetric training

This single parameter subsumes the old separate training stages. At `mix-ratio-3d=0.0`, the model trains on individual slices with no memory bank involvement, which is functionally equivalent to image-only training. As the ratio increases, more training iterations involve full volumetric propagation through SAM2's memory mechanism.

**Trainable components per mode:**
- 2D slices (`mix-ratio-3d=0.0`): Mask decoder only (memory encoder/attention receive no gradients)
- 3D volumes (`mix-ratio-3d>0`): Mask decoder + memory encoder + memory attention

**Training data options:**
1. **Pre-computed embeddings** (fastest): Use `train_unified_emb.py` with `--embedding-dir`
2. **Filtered training list** (flexible): Use `train_unified.py` with `--training-list` and `--dice-threshold`
3. **Raw images** (slowest): Use `train_unified.py` with `--data-paths` directly

**Key differences from natural video:**
- Less motion between MRI slices (more consistency)
- Occlusion head important (organs appear/disappear across slices)
- Bidirectional propagation crucial for 3D coherence

**AMP and Embedding Training (IMPORTANT):**
- When using `--use-amp`, features are computed in float16 for memory efficiency
- Pre-computed embeddings (`generate_3d_embeddings.py`) are stored as float16
- `train_unified_emb.py` must keep embeddings as float16 (not convert to float32) to match raw image training behavior
- Converting embeddings to float32 doubles memory usage and causes OOM

**Embedding Training Memory Management:**
- Gradient graphs accumulate across forward passes until `backward()` is called
- Each 3D sample with K labels * N slices = K*N forward passes with gradients retained
- Use `--max-labels-per-sample 1` to limit gradient accumulation (randomly samples 1 label per volume per epoch)
- Use `batch_size=1` and `grad_accum_steps=1` for memory-constrained environments
- Empty batches (no valid losses) must be skipped to avoid `scaler.step()` assertion errors
- The `batch_losses` list pattern: collect valid losses, skip if empty, `torch.stack().mean()` for proper gradient attachment

### 3D Volumetric Loss (train_unified_emb_3dloss.py)

The 3D loss training adds a per-label volumetric Dice loss at 256x256 to capture global 3D shape consistency:

**Loss formulation:**
```
total_loss = lambda_2d * per_slice_dice_focal + lambda_3d * volumetric_dice
```

**How it works:**
1. During propagation, per-slice 256x256 predictions (same resolution as 2D loss) are collected
2. After processing all slices for a label, predictions are stacked into a 3D volume `(1, D, 256, 256)`
3. A single-scale 3D Dice loss is computed on the stacked volume for volumetric consistency
4. The 3D Dice is computed per-label, so small objects are not diluted by large ones; Dice ignores true-negative (empty slice) agreement naturally

**Why 3D loss helps (vs. memory bank alone):**
- Memory bank provides **conditioning** (past predictions influence current), but gradients are detached
- 3D loss provides **explicit supervision** on volumetric shape with gradient flow
- Helps prevent accumulating slice-wise errors that compound across the volume

**Recommended hyperparameters:**
- `--lambda-2d 1.0 --lambda-3d 0.5`: Balanced (default, good starting point)
- `--lambda-2d 0.5 --lambda-3d 1.0`: Emphasize volumetric consistency
- `--lambda-2d 1.0 --lambda-3d 0.0`: 2D only (ablation, equivalent to train_unified_emb.py)

### GVT (Global Volume Token) Module (gvt/)

GVT injects volume-level context into SAM2's memory bank without modifying any SAM2 source files.

**Files:**
- `gvt/gvt_module.py` — Three classes:
  - `GVTModule`: Perceiver-style cross-attention, N slice embeddings → K global tokens.
  - `GVTReconDecoder`: training-only auxiliary decoder, expands K tokens → N low-res binary masks (discarded at inference).
  - `TSDFDecoder`: training-only TSDF shape prior decoder, produces full-resolution (256×256) TSDF predictions per slice conditioned on prompted memory features (discarded at inference).
- `gvt/gvt_integration.py` — `GVTMemoryAttentionWrapper`: wraps SAM2's `memory_attention`, appending GVT tokens to the memory bank on every forward call. Helper functions: `wrap_model_with_gvt`, `compute_gvt_tokens`, `clear_gvt_cache`, `has_gvt`, `load_gvt_checkpoint`.
- `gvt/train_gvt_emb_tsdf.py` — GVT v3 training script (2D + 3D + recon + TSDF shape prior).
- `utils/tsdf_utils.py` — `compute_tsdf`, `prepare_tsdf_targets`, `tsdf_loss`.

**GVT versions:**
- **GVT v1**: GVT encoder only. K tokens appended to memory bank; training signal comes solely from segmentation loss. Limited by weak/indirect gradient path to GVT encoder.
- **GVT v2**: Adds `GVTReconDecoder` auxiliary reconstruction loss with **label conditioning**. `GVTReconDecoder` cross-attends GVT tokens (Q) to the cut-slice `maskmem_features` (K/V) from Phase 1, conditioning on the specific label before reconstructing per-slice low-res masks. Each label gets its own GT binary mask (not union). Explicitly forces GVT encoder to learn spatially meaningful representations via a direct, short gradient path.
- **GVT v3** (current): Replaces binary reconstruction with a **TSDF shape prior**. `TSDFDecoder` predicts full-resolution (256×256) Truncated Signed Distance Fields per slice, conditioned on label-prompted memory features via cross-attention. TSDF encodes boundary geometry explicitly (positive inside, negative outside, clamped to ±τ), giving the decoder a richer supervision signal than binary masks. Both `GVTReconDecoder` and `TSDFDecoder` are discarded at inference — zero overhead.

**GVT v3 loss formulation:**
```
total_loss = L_seg + lambda_aux * L_recon + lambda_tsdf * L_tsdf
```
- `L_seg` = mean of per-video (2D Dice-Focal + 3D volumetric Dice) losses
- `L_recon` = BCE + soft Dice on low-resolution (R×R) mask reconstructions from `GVTReconDecoder`
- `L_tsdf` = boundary-weighted SmoothL1 on 256×256 TSDF from `TSDFDecoder`
- All three terms are **additive** (never averaged together) — `lambda_aux` and `lambda_tsdf` have consistent effective weights regardless of how many seg losses exist.

**Three-phase per-volume training loop (`train_gvt_emb_tsdf.py`):**
```
Phase 1 — no_grad: for each label, run prompted forward pass on cut slice to extract
           maskmem_features (label-conditioned memory features). Shared by BOTH
           GVTReconDecoder (v2) and TSDFDecoder (v3). Phase 1 runs when either
           recon_decoder or tsdf_decoder is active (not only when --use-tsdf is set).

Phase 2a — Recon auxiliary loss (v2): GVTReconDecoder(gvt_tokens, num_slices, mem_features)
            per label → per-label binary GT → BCE + soft Dice → mean over labels

Phase 2b — TSDF auxiliary loss (v3): TSDFDecoder(gvt_tokens, maskmem_features, slice_indices)
            per label → predict 256×256 TSDF per slice → tsdf_loss vs prepare_tsdf_targets

Phase 3 — Seg loss: process_video_with_memory_3dloss(v1, v2) for each label
           → collect per-slice predictions → 2D Dice-Focal + 3D volumetric Dice
```

**`GVTReconDecoder` architecture (v2):**
```
Input:  gvt_tokens (K, 1, C_mem) + num_slices + mem_features (1, M, C_mem) [optional]
  1. Transpose tokens to (1, K, C_mem) [batch_first]
  2. Label conditioning (when mem_features provided):
       cross-attention: Q=GVT tokens, K/V=mem_features → label_cross_attn_norm(residual)
  3. Slice queries from sinusoidal PE: (1, N, C_mem)
  4. 2-layer TransformerDecoder: Q=slice_queries, KV=conditioned tokens → (1, N, C_mem)
  5. Linear head: (1, N, C_mem) → (1, N, recon_res²) → reshape (1, N, 1, R, R)
Output: (1, N, 1, recon_res, recon_res) logits
Note:   mem_features=None skips conditioning; backward-compatible with v1-era usage.
```

**`TSDFDecoder` architecture (v3):**
```
Input:  gvt_tokens (B, K, C_mem=64) + mem_features (B, M, 64) + slice_indices (S,)
Per slice:
  1. Add learned slice positional embedding to GVT tokens
  2. Cross-attention: Q=GVT tokens, K/V=prompted memory features
  3. Flatten K tokens → Linear → spatial seed (seed_channels × 4 × 4)
  4. 6× ConvTranspose2d: 4→8→16→32→64→128→256
  5. 1×1 Conv → single-channel TSDF (linear, NO sigmoid)
Output: (B, S, 256, 256) TSDF predictions
```

**TSDF utilities (`utils/tsdf_utils.py`):**
- `compute_tsdf(binary_mask, tau=32.0)`: SDF = dist_inside − dist_outside, clamped to [−τ, +τ]. All-zero mask → all −τ.
- `prepare_tsdf_targets(gt_masks_3d, label_ids, target_size=256, tau=32.0)`: returns `dict[label_id → Tensor(D, 256, 256)]`.
- `tsdf_loss(tsdf_pred, tsdf_gt, alpha=4.0, sigma=10.67, label_area=None)`: boundary-weighted SmoothL1. Weight `w = 1 + alpha * exp(−tsdf_gt² / (2σ²))` — high near boundary (|tsdf_gt|≈0), low far from it. Optional `label_area` divides by `sqrt(area)` for multi-label size balance.

**CLI flags (`gvt/train_gvt_emb_tsdf.py`):**
- `--use-gvt`: Enable GVT encoder (required for v3)
- `--gvt-num-tokens`: K global tokens (default: 16)
- `--gvt-num-layers`: Perceiver cross-attention depth (default: 2)
- `--gvt-num-heads`: Attention heads (default: 4)
- `--gvt-recon`: Enable `GVTReconDecoder` reconstruction loss
- `--gvt-recon-res`: Reconstruction resolution R (default: 16)
- `--lambda-aux`: Recon loss weight (default: 1.0); ramps up over `--lambda-aux-warmup-epochs`
- `--lambda-aux-warmup-epochs`: Warm-up epochs (default: 10)
- `--use-tsdf`: Enable `TSDFDecoder` TSDF shape prior loss
- `--lambda-tsdf`: TSDF loss weight (default: 0.1)
- `--tsdf-tau`: Truncation threshold in pixels (default: 32.0)
- `--tsdf-alpha`: Boundary upweighting factor (default: 4.0)
- `--tsdf-single-label`: Disable per-label `sqrt(area)` normalization (use for single-label datasets)

**GVT token flow per volume:**
```
compute_gvt_tokens(model, image_embeds[all N], ...)   # K tokens from ALL N slices — never capped
   → wrapper.gvt_tokens_cache = (gvt_tokens, gvt_pos)
prepare_tsdf_targets(gt_masks_3d, label_ids)          # precompute TSDF GT for all labels (v3 only)

for each label:
    Phase 1 (no_grad): extract maskmem_features from cut slice
                       [runs when recon_decoder or tsdf_decoder is active]
    Phase 2a (v2): GVTReconDecoder(gvt_tokens, N, mem_features) → L_recon (per-label binary GT)
    Phase 2b (v3): TSDFDecoder(gvt_tokens, maskmem_features) → L_tsdf (capped at max_slices)
    Phase 3: propagate v1 + v2 → L_seg (capped at max_slices per video)

clear_gvt_cache(model)
```

**Checkpoint keys:**
- `"model"`: full SAM2 + GVTModule state dict (including `memory_attention` wrapper)
- `"gvt_recon_decoder"`: `GVTReconDecoder` state dict (key name is `"gvt_recon_decoder"`, not `"recon_decoder"`)
- `"tsdf_decoder"`: `TSDFDecoder` state dict

**DDP for auxiliary decoders (IMPORTANT):**
Both `recon_decoder` and `tsdf_decoder` are wrapped in DDP for gradient all-reduction across GPUs. Save checkpoints using `.module.state_dict()`:
```python
ckpt["gvt_recon_decoder"] = recon_decoder.module.state_dict()
ckpt["tsdf_decoder"] = tsdf_decoder.module.state_dict()
```
When loading in `eval_gvt_recon.py`, old checkpoints missing `label_cross_attn` keys load with `strict=False` and a warning; the new cross-attention weights are randomly initialized (eval Dice is a lower-bound sanity check only for such checkpoints).

**GPU load balance (IMPORTANT — do not regress):**
`compute_gvt_tokens` intentionally processes all N slices for global context. Both `TSDFDecoder` and `GVTReconDecoder` must be capped at `max_slices_per_video` to balance per-sample GPU cost. Without caps, large-N volumes (N=200) cause one GPU to run 2× longer, leaving others idle during NCCL allreduce.

```python
# Correct pattern — TSDF decoder capped at max_slices
tsdf_slice_indices = list(range(N_total))
if N_total > max_slices:
    tsdf_slice_indices = subsample_slices(tsdf_slice_indices, max_slices)
slice_idx_tensor = torch.tensor(tsdf_slice_indices, dtype=torch.long, device=device)
tsdf_pred = tsdf_decoder(gvt_tokens_fp32, maskmem_features, slice_idx_tensor)
```

**MAX_SLICES_PER_VIDEO and training-inference gap:**
- `MAX_SLICES_PER_VIDEO=32` caps video processing during training. At inference the full volume (50–200+ slices) is processed.
- SAM2's memory bank is a **sliding window** (~`num_maskmem=7` frames), so the gap is less severe than it appears — each frame always attends to the same number of recent memory frames.
- GVT tokens (computed from all N slices, uncapped) directly address the gap by providing global volume context independent of sequential propagation depth.
- Memory bank = local (last 7 frames); GVT = global (all N slices). They are complementary, not redundant.

**`_encode_new_memory` signature (IMPORTANT):**
Both `sam2_base.py` and `sam2_video_predictor.py` define `_encode_new_memory` with `object_score_logits` as a required positional argument (4th). This must be captured from `sam_mask_decoder` output and passed explicitly:
```python
low_res_masks, _, _, object_score_logits = predictor.sam_mask_decoder(...)
maskmem_features, _ = predictor._encode_new_memory(
    ..., object_score_logits=object_score_logits, ...
)
```
