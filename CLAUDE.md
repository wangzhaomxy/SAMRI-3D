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
  --dataset-path /path/to/dataset/ \
  --save-path /path/to/results.pkl \
  --prompt-mode box \
  --num-workers 8

# 3D multi-object evaluation (faster, all labels at once)
python test_vis3d_multi.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --dataset-path /path/to/dataset/ \
  --save-path /path/to/results.pkl \
  --prompt-mode box \
  --num-workers 8

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
```bash
# Single-node training
python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
  --use-cluster 0 \
  --num-gpus 8

# Multi-node SLURM training
python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
  --use-cluster 1 \
  --num-gpus 8 \
  --num-nodes 2
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
- `test_vis2d.py` - 2D slice evaluation
- `gen_training_list.py` - Generate training lists with dice scores for filtered training
- `generate_sam2_embeddings.py` - Pre-compute SAM2 image embeddings for faster training
- `train_stage_one_slicewise.py` - Stage 1 training on 2D slices (mask decoder only)

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
this means calling `init_state` 2×K times per volume (K = label count).  Two
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
init_state  ×2  (video1 + video2)
prefill     ×2  (only when K > 1)
for each label:
    add_prompt  →  propagate  →  reset_state   (×2 per label)
```

Image encoding cost: **N** forward_image calls total (not N×K).
`init_state` / `load_video_frames` cost: **2** calls total (not 2K).

> **Do not** skip `prefill` when K > 1.  Without it every label re-encodes
> all frames because the 1-entry cache is destroyed during propagation.

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
- `RELEASE_NOTES.md` - Version history documenting SAM2 → SAM2.1 improvements

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

### SAMRI2 Training Approach (3D MRI as Bidirectional Videos)
1. Split 3D volume at "central slice" (largest foreground) into two sequences
2. Treat each sequence as a "video" with slices as frames
3. Propagate segmentation bidirectionally from central slice

**Recommended training stages:**
- **Stage 1 (Image)**: Train mask decoder on 2D slices to match SAMRI v1 performance
- **Stage 2 (Joint)**: Add memory encoder + memory attention training with mixed image/video batches

**Training data options (train_stage_one_slicewise.py):**
1. **Pre-computed embeddings** (fastest): Use `--embedding-path` with `generate_sam2_embeddings.py` output
2. **Filtered training list** (flexible): Use `--training-list` with `gen_training_list.py` output and `--dice-threshold`
3. **Raw images** (slowest): Use `--data-paths` directly

**Key differences from natural video:**
- Less motion between MRI slices (more consistency)
- Occlusion head important (organs appear/disappear across slices)
- Bidirectional propagation crucial for 3D coherence
