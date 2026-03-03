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
# 3D single-object evaluation (iterative, per-label)
python test_vis3d.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --dataset-path /path/to/test_data/ /path/to/zeroshot_results/ \
  --save-path /path/to/results.pkl \
  --prompt-mode box \
  --num-workers 8

# 3D multi-object evaluation (faster, all labels at once)
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
# Edit TRAINING_MODE in train_multi_gpus.sh: "2d_emb", "3d_emb", or "2d_img"
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
- `test_vis3d.py` - 3D evaluation (single object per run, iterative)
- `test_vis3d_multi.py` - 3D evaluation (all objects simultaneously, higher accuracy)
- `test_vis3d_crf.py` - 3D evaluation with Dense CRF post-processing (experimental)
- `test_vis3d_twopass.py` - 3D evaluation with two-pass auto-context refinement (experimental)
- `test_vis2d.py` - 2D slice evaluation
- `gen_training_list.py` - Generate training lists with dice scores for filtered training
- `generate_sam2_embeddings.py` - Pre-compute SAM2 image embeddings for 2D training
- `generate_3d_embeddings.py` - Pre-compute SAM2 embeddings for 3D volumes
- `train_unified.py` - Unified training from raw images (2D loss)
- `train_unified_emb.py` - Unified training from pre-computed embeddings (2D loss)
- `train_unified_emb_3dloss.py` - Unified training from embeddings (2D + 3D volumetric loss)

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

### Single-Object Evaluation State-Reuse Pattern (test_vis3d.py)

`test_vis3d.py` runs one propagation per label (single-object mode).  Naively
this means calling `init_state` 2*K times per volume (K = label count).  Two
SAM2 internals make a much faster pattern possible:

1. **`reset_state()` preserves image data.**  It clears all object/tracking
   state (`obj_ids`, `output_dict_per_obj`, etc.) but does **not** touch
   `cached_features` or `images`.  The same state can be re-prompted after a
   `reset_state` call.

2. **`_get_image_feature` is a 1-entry LRU cache** — each cache miss
   *replaces* the entire `cached_features` dict.  After propagation only the
   last frame's features survive.  To keep all frames cached across labels,
   pre-fill the cache before the label loop:

```python
# Pre-fill: runs forward_image once per frame; persists through reset_state
for i in range(num_frames):
    image = state["images"][i].to(device).float().unsqueeze(0)
    backbone_out = predictor.forward_image(image)
    cache[i] = (image, backbone_out)
state["cached_features"] = cache
```

The resulting per-case loop is:
```
init_state  x2  (video1 + video2)
prefill     x2  (only when K > 1)
for each label:
    add_prompt  ->  propagate  ->  reset_state   (x2 per label)
```

Image encoding cost: **N** forward_image calls total (not N*K).
`init_state` / `load_video_frames` cost: **2** calls total (not 2K).

> **Do not** skip `prefill` when K > 1.  Without it every label re-encodes
> all frames because the 1-entry cache is destroyed during propagation.

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
