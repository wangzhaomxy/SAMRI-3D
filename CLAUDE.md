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
# 3D single-object evaluation
python test_vis3d.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --dataset-path /path/to/dataset/ \
  --save-path /path/to/results.pkl \
  --prompt-mode box

# 3D multi-object evaluation
python test_vis3d_multi.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --dataset-path /path/to/dataset/ \
  --save-path /path/to/results.pkl \
  --prompt-mode box

# 3D inference without metrics
python 3d_inference.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/checkpoint.pt \
  --dataset-path /path/to/dataset/ \
  --save-root /path/to/output/ \
  --prompt-mode box
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

**sam2/** - Meta's SAM2 package with:
- `modeling/` - Hiera backbone, memory attention, mask decoder
- `sam2_video_predictor.py` - Frame-to-frame propagation interface
- `configs/` - YAML configs for model variants (Tiny/Small/Base+/Large)

**training/** - Training system with:
- `model/sam2.py` - SAM2Train class with point/box prompt sampling
- `trainer.py` - Hydra-based DDP training with AMP support
- `dataset/` - VOSDataset loaders for video object segmentation

**utils/** - SAMRI-specific utilities:
- `dataloader.py` - NiiDataset for NIfTI file loading
- `losses.py` - Dice, Hausdorff Distance, Mean Surface Distance metrics

### Dataset Conventions
- Images and masks must be in the same folder
- Naming: `*_img_*` for images, `*_seg_*` (or `*_gt_*`) for ground truth
- Supported formats: .nii.gz, .nii, .mha, .mhd, .nrrd
- Test data auto-detected in `<dataset>/testing/` or scanned directly

### Prompt Modes
- `box` - Bounding box prompts derived from ground truth
- `mask` - Dense point sampling from ground truth masks

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

**Key differences from natural video:**
- Less motion between MRI slices (more consistency)
- Occlusion head important (organs appear/disappear across slices)
- Bidirectional propagation crucial for 3D coherence
