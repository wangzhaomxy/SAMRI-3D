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
