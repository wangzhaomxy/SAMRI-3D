# -*- coding: utf-8 -*-
"""
Stage 1 Training: Slice-wise mask decoder fine-tuning for SAM2

This script trains only the mask decoder while freezing all other components:
- Image encoder (frozen)
- Prompt encoder (frozen)
- Memory encoder (frozen)
- Memory attention (frozen)

Supports two data modes:
1. Raw images (NiiDataset): Load NIfTI images and compute embeddings on-the-fly
2. Pre-computed embeddings (SAM2EmbDataset): Load pre-computed embeddings for faster training

Based on SAMRI v1 training approach adapted for SAM2 architecture.

=== CLI Examples ===

1. Training from Raw Images (compute embeddings on-the-fly):
   Note: Slower but doesn't require pre-computed embeddings

   python train_stage_one_slicewise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --train-data-paths /scratch/project/samri/Datasets/Dataset1/training/ \\
                          /scratch/project/samri/Datasets/Dataset2/training/ \\
       --save-path /scratch/project/samri/Model_save/samri2_stage1_raw/ \\
       --num-epochs 100 \\
       --batch-size 4 \\
       --lr 1e-5 \\
       --prompt-mode mixed

2. Training from Pre-computed Embeddings (recommended, faster):

   a) Using all pre-filtered samples (train_40_up + train_40_down):
   python train_stage_one_slicewise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --use-embeddings --sam2-embeddings \\
       --embedding-file /scratch/project/samri/SAM2_embeddings/embedding_list.pkl \\
       --subset all \\
       --min-samples-per-epoch 5000 \\
       --save-path /scratch/project/samri/Model_save/samri2_stage1_emb/ \\
       --num-epochs 100 \\
       --batch-size 8

   b) Using only high-quality samples (dice > 0.4):
   python train_stage_one_slicewise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --use-embeddings --sam2-embeddings \\
       --embedding-file /scratch/project/samri/SAM2_embeddings/embedding_list.pkl \\
       --subset 40_up \\
       --min-samples-per-epoch 5000 \\
       --save-path /scratch/project/samri/Model_save/samri2_stage1_60up/

   c) Using custom dice threshold filtering (e.g., 0.2 < dice <= 0.6):
   python train_stage_one_slicewise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --use-embeddings --sam2-embeddings \\
       --embedding-file /scratch/project/samri/SAM2_embeddings/embedding_list.pkl \\
       --subset custom \\
       --dice-threshold-low 0.2 \\
       --dice-threshold-high 0.6 \\
       --min-samples-per-epoch 5000 \\
       --save-path /scratch/project/samri/Model_save/samri2_stage1_custom/

   d) No upper threshold (all samples with dice > 0.2):
   python train_stage_one_slicewise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --use-embeddings --sam2-embeddings \\
       --embedding-file /scratch/project/samri/SAM2_embeddings/embedding_list.pkl \\
       --subset custom \\
       --dice-threshold-low 0.2 \\
       --min-samples-per-epoch 0 \\
       --save-path /scratch/project/samri/Model_save/samri2_stage1_all/

Key Arguments for Embedding Training:
  --subset: "40_up" (dice>0.4), "40_down" (0.2<dice<=0.4), "all", or "custom"
  --dice-threshold-low: Min dice to include (default: 0.2, only for subset=custom)
  --dice-threshold-high: Max dice to include (default: None, only for subset=custom)
  --min-samples-per-epoch: Repeat small datasets to reach this size (default: 5000, set 0 to disable)
"""

import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from utils.dataloader import NiiDataset, BalancedEmbDataset, SAM2EmbDataset
from utils.losses import DiceFocalLoss
from utils.utils import gen_points, gen_bboxes, get_checkpoint


def variable_size_collate_fn(batch):
    """
    Custom collate function for variable-sized images and masks.
    Returns lists instead of stacked tensors.
    """
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    return images, masks


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Slice-wise SAM2 mask decoder training")

    # Model configuration
    parser.add_argument("--sam2-config", type=str, required=True,
                        help="Path to SAM2 config file (e.g., configs/sam2.1/sam2.1_hiera_s.yaml)")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to SAM2 checkpoint")

    # Data configuration
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Use pre-computed embeddings instead of raw images")
    parser.add_argument("--sam2-embeddings", action="store_true",
                        help="Use SAM2-format embeddings with high-res features (requires --use-embeddings)")
    parser.add_argument("--embedding-file", type=str, default=None,
                        help="Path to pickle file containing embedding file list (required if --use-embeddings)")
    parser.add_argument("--subset", type=str, default="all",
                        choices=["40_up", "40_down", "all", "custom"],
                        help="Subset to use from embedding file. Use 'custom' with dice thresholds (default: all)")
    parser.add_argument("--train-data-paths", type=str, nargs="+", default=None,
                        help="List of training data paths (for raw image mode)")
    parser.add_argument("--val-data-paths", type=str, nargs="+", default=None,
                        help="List of validation data paths (optional)")

    # Embedding filtering options (only for --use-embeddings --sam2-embeddings)
    parser.add_argument("--dice-threshold-low", type=float, default=0.2,
                        help="Min dice threshold for custom filtering (samples with dice <= this are excluded). Default: 0.2")
    parser.add_argument("--dice-threshold-high", type=float, default=None,
                        help="Max dice threshold for custom filtering (samples with dice > this are excluded). "
                             "Common values: 0.4 (hard samples), 0.6 (medium), None (no upper limit). Default: None")
    parser.add_argument("--min-samples-per-epoch", type=int, default=5000,
                        help="Minimum samples per epoch. Small datasets are repeated to reach this. "
                             "Set to 0 to disable. Default: 5000")

    # Training configuration
    parser.add_argument("--save-path", type=str, required=True,
                        help="Path to save model checkpoints")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")

    # Prompt configuration
    parser.add_argument("--prompt-mode", type=str, default="mixed",
                        choices=["point", "box", "mixed"],
                        help="Prompt mode: point, box, or mixed")
    parser.add_argument("--jitter", type=int, default=10,
                        help="Bounding box jitter for data augmentation")

    # Loss configuration
    parser.add_argument("--lambda-dice", type=float, default=1.0,
                        help="Weight for dice loss")
    parser.add_argument("--lambda-focal", type=float, default=10.0,
                        help="Weight for focal loss")

    # Other
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--master-port", type=int, default=None,
                        help="Master port for DDP (default: random port between 29500-29999)")

    args = parser.parse_args()

    # Validate arguments
    if args.use_embeddings:
        if args.embedding_file is None:
            parser.error("--embedding-file is required when using --use-embeddings")
    else:
        if args.train_data_paths is None:
            parser.error("--train-data-paths is required when not using --use-embeddings")

    return args


def ddp_setup(rank: int, world_size: int):
    """
    Setup distributed data parallel training.

    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # MASTER_PORT should already be set before spawning processes
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def freeze_model_components(model):
    """
    Freeze all model components except the mask decoder.

    Args:
        model: SAM2 model
    """
    # Freeze image encoder
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # Freeze prompt encoder
    for param in model.sam_prompt_encoder.parameters():
        param.requires_grad = False

    # Freeze memory encoder
    for param in model.memory_encoder.parameters():
        param.requires_grad = False

    # Freeze memory attention
    for param in model.memory_attention.parameters():
        param.requires_grad = False

    # Freeze other parameters (no_mem_embed, maskmem_tpos_enc, etc.)
    if hasattr(model, 'no_mem_embed'):
        model.no_mem_embed.requires_grad = False
    if hasattr(model, 'no_mem_pos_enc'):
        model.no_mem_pos_enc.requires_grad = False
    if hasattr(model, 'maskmem_tpos_enc'):
        model.maskmem_tpos_enc.requires_grad = False

    # Only mask decoder remains trainable
    for param in model.sam_mask_decoder.parameters():
        param.requires_grad = True


def prepare_prompts(masks, prompt_mode, jitter, device, transforms, orig_hw):
    """
    Prepare prompts (points or boxes) from ground truth masks.

    Args:
        masks: Ground truth masks (B, 1, H, W)
        prompt_mode: 'point', 'box', or 'mixed'
        jitter: Bounding box jitter
        device: Target device
        transforms: SAM2Transforms for coordinate transformation
        orig_hw: Original image height and width

    Returns:
        point_coords: Point coordinates tensor or None
        point_labels: Point labels tensor or None
        boxes: Box coordinates tensor or None
    """
    batch_size = masks.shape[0]
    point_coords_list = []
    point_labels_list = []
    boxes_list = []

    for i in range(batch_size):
        mask = masks[i, 0].cpu().numpy()  # (H, W)

        if prompt_mode == "point" or prompt_mode == "mixed":
            # Generate point prompt
            points = gen_points(mask, num_points=1)  # (1, 2) in [W, H] format
            point_coords_list.append(points)
            point_labels_list.append(np.array([1]))  # Positive point

        if prompt_mode == "box" or prompt_mode == "mixed":
            # Generate box prompt
            box = gen_bboxes(mask, num_bboxes=1, jitter=jitter)  # [min_w, min_h, max_w, max_h]
            boxes_list.append(box)

    # Convert to tensors and transform coordinates
    point_coords = None
    point_labels = None
    boxes = None

    if prompt_mode == "point" or prompt_mode == "mixed":
        point_coords = torch.tensor(np.array(point_coords_list), dtype=torch.float, device=device)
        point_labels = torch.tensor(np.array(point_labels_list), dtype=torch.int, device=device)
        # Transform coordinates to model input space
        point_coords = transforms.transform_coords(point_coords, normalize=True, orig_hw=orig_hw)

    if prompt_mode == "box" or prompt_mode == "mixed":
        boxes = torch.tensor(np.array(boxes_list), dtype=torch.float, device=device)
        # Transform box coordinates to model input space
        boxes = transforms.transform_boxes(boxes, normalize=True, orig_hw=orig_hw)

    return point_coords, point_labels, boxes


def prepare_prompts_from_embedding(masks, prompt_mode, jitter, device, transforms, orig_hw_list):
    """
    Prepare prompts for pre-computed embeddings (per-sample original size).

    Args:
        masks: Ground truth masks (B, 1, H, W)
        prompt_mode: 'point', 'box', or 'mixed'
        jitter: Bounding box jitter
        device: Target device
        transforms: SAM2Transforms for coordinate transformation
        orig_hw_list: List of (H, W) tuples for each sample

    Returns:
        point_coords: Point coordinates tensor or None
        point_labels: Point labels tensor or None
        boxes: Box coordinates tensor or None
    """
    batch_size = masks.shape[0]
    point_coords_list = []
    point_labels_list = []
    boxes_list = []

    for i in range(batch_size):
        mask = masks[i, 0].cpu().numpy()  # (H, W)
        orig_hw = orig_hw_list[i]

        if prompt_mode == "point" or prompt_mode == "mixed":
            points = gen_points(mask, num_points=1)
            # Transform per-sample
            pt = torch.tensor(points, dtype=torch.float, device=device).unsqueeze(0)
            pt = transforms.transform_coords(pt, normalize=True, orig_hw=orig_hw)
            point_coords_list.append(pt.squeeze(0))
            point_labels_list.append(torch.tensor([1], dtype=torch.int, device=device))

        if prompt_mode == "box" or prompt_mode == "mixed":
            box = gen_bboxes(mask, num_bboxes=1, jitter=jitter)
            bx = torch.tensor(box, dtype=torch.float, device=device).unsqueeze(0)
            bx = transforms.transform_boxes(bx, normalize=True, orig_hw=orig_hw)
            boxes_list.append(bx.squeeze(0))

    point_coords = None
    point_labels = None
    boxes = None

    if prompt_mode == "point" or prompt_mode == "mixed":
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

    if prompt_mode == "box" or prompt_mode == "mixed":
        boxes = torch.stack(boxes_list, dim=0)

    return point_coords, point_labels, boxes


def forward_sam2_image(model, images, point_coords, point_labels, boxes, transforms, orig_hw):
    """
    Forward pass through SAM2 for image segmentation (raw images).

    Args:
        model: SAM2 model
        images: Input images (B, 3, H, W), already transformed
        point_coords: Point coordinates or None
        point_labels: Point labels or None
        boxes: Box coordinates or None
        transforms: SAM2Transforms
        orig_hw: Original image height and width

    Returns:
        low_res_masks: Low resolution mask logits (B, 1, 256, 256)
    """
    batch_size = images.shape[0]

    # Get image embeddings
    backbone_out = model.forward_image(images)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)

    # Add no_mem_embed for image-only mode
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    # Prepare features
    bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
    feats = [
        feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
    ][::-1]

    image_embed = feats[-1]  # (B, C, 64, 64)
    high_res_feats = feats[:-1]  # List of high-res features

    # Prepare prompts
    concat_points = None
    if point_coords is not None:
        concat_points = (point_coords, point_labels)

    if boxes is not None:
        box_coords = boxes.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
        box_labels = box_labels.repeat(boxes.size(0), 1)

        if concat_points is not None:
            concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
            concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
            concat_points = (concat_coords, concat_labels)
        else:
            concat_points = (box_coords, box_labels)

    # Get prompt embeddings
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=None,
    )

    # Process each image in batch
    all_low_res_masks = []
    for i in range(batch_size):
        # Get high-res features for this image
        high_res_features_i = [feat[i:i+1] for feat in high_res_feats]

        # Predict masks
        low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embed[i:i+1],
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings[i:i+1],
            dense_prompt_embeddings=dense_embeddings[i:i+1],
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features_i,
        )
        all_low_res_masks.append(low_res_masks)

    # Stack all masks
    low_res_masks = torch.cat(all_low_res_masks, dim=0)  # (B, 1, 256, 256)

    return low_res_masks


def forward_sam2_embedding(model, embeddings, point_coords, point_labels, boxes, device, high_res_feats=None):
    """
    Forward pass through SAM2 using pre-computed embeddings (skip image encoder).

    Args:
        model: SAM2 model
        embeddings: Pre-computed image embeddings (B, 256, 64, 64)
        point_coords: Point coordinates or None
        point_labels: Point labels or None
        boxes: Box coordinates or None
        device: Target device
        high_res_feats: Optional tuple of (high_res_feat_0, high_res_feat_1) tensors

    Returns:
        low_res_masks: Low resolution mask logits (B, 1, 256, 256)
    """
    batch_size = embeddings.shape[0]

    # Embeddings are already computed, use them directly
    image_embed = embeddings.to(device)  # (B, 256, 64, 64)

    # Prepare prompts
    concat_points = None
    if point_coords is not None:
        concat_points = (point_coords, point_labels)

    if boxes is not None:
        box_coords = boxes.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
        box_labels = box_labels.repeat(boxes.size(0), 1)

        if concat_points is not None:
            concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
            concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
            concat_points = (concat_coords, concat_labels)
        else:
            concat_points = (box_coords, box_labels)

    # Get prompt embeddings
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=None,
    )

    # Process each image in batch
    all_low_res_masks = []
    for i in range(batch_size):
        # Prepare high-res features for this sample if available
        if high_res_feats is not None:
            high_res_features_i = [
                high_res_feats[0][i:i+1],  # (1, 32, 256, 256)
                high_res_feats[1][i:i+1],  # (1, 64, 128, 128)
            ]
        else:
            high_res_features_i = None

        # Predict masks
        low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embed[i:i+1],
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings[i:i+1],
            dense_prompt_embeddings=dense_embeddings[i:i+1],
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features_i,
        )
        all_low_res_masks.append(low_res_masks)

    # Stack all masks
    low_res_masks = torch.cat(all_low_res_masks, dim=0)  # (B, 1, 256, 256)

    return low_res_masks


def train_epoch(model, dataloader, optimizer, loss_fn, transforms, args, device, rank):
    """
    Train for one epoch (raw image mode).
    Handles variable-sized images by processing each sample individually.

    Args:
        model: SAM2 model (DDP wrapped)
        dataloader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        transforms: SAM2Transforms
        args: Command line arguments
        device: Target device
        rank: Process rank for DDP

    Returns:
        Average epoch loss
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, disable=(rank != 0))
    for images_list, masks_list in pbar:
        # images_list: list of (H, W, 3) numpy arrays
        # masks_list: list of (1, H, W) or (H, W) numpy arrays

        batch_size = len(images_list)

        # Process each image individually due to variable sizes
        batch_loss = 0.0
        valid_samples = 0

        optimizer.zero_grad()

        for i in range(batch_size):
            image = images_list[i]  # (H, W, 3) numpy array
            mask = masks_list[i]  # (1, H, W) or (H, W) numpy array

            # Get original size
            if isinstance(image, np.ndarray):
                orig_hw = (image.shape[0], image.shape[1])
            else:
                orig_hw = (image.shape[0], image.shape[1])
                image = image.numpy()

            # Transform image
            image_tensor = transforms(image).unsqueeze(0).to(device)  # (1, 3, H, W)

            # Prepare mask
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # (1, H, W)
            mask = mask.unsqueeze(0).float().to(device)  # (1, 1, H, W)

            # Resize mask to 256x256
            mask_resized = torch.nn.functional.interpolate(
                mask, size=(256, 256), mode="nearest"
            )

            # Prepare prompts for single sample
            point_coords, point_labels, boxes = prepare_prompts(
                mask, args.prompt_mode, args.jitter, device, transforms, orig_hw
            )

            with torch.cuda.amp.autocast(enabled=False):
                low_res_mask = forward_sam2_image(
                    model.module if hasattr(model, 'module') else model,
                    image_tensor, point_coords, point_labels, boxes,
                    transforms, orig_hw
                )

                if torch.isnan(low_res_mask).any():
                    continue

                # Compute loss for this sample
                sample_loss = loss_fn(low_res_mask, mask_resized)

                if torch.isnan(sample_loss):
                    continue

                # Accumulate gradients
                (sample_loss / batch_size).backward()
                batch_loss += sample_loss.item()
                valid_samples += 1

        if valid_samples > 0:
            optimizer.step()
            epoch_loss += batch_loss / valid_samples
            num_batches += 1

            if rank == 0:
                pbar.set_postfix({"loss": f"{batch_loss / valid_samples:.4f}"})

    return epoch_loss / max(num_batches, 1)


def train_epoch_embedding(model, dataloader, optimizer, loss_fn, transforms, args, device, rank):
    """
    Train for one epoch using pre-computed SAMRI v1 embeddings (deprecated, for backward compatibility).

    Args:
        model: SAM2 model (DDP wrapped)
        dataloader: Training data loader (BalancedEmbDataset)
        optimizer: Optimizer
        loss_fn: Loss function
        transforms: SAM2Transforms
        args: Command line arguments
        device: Target device
        rank: Process rank for DDP

    Returns:
        Average epoch loss
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, disable=(rank != 0))
    for embeddings, masks, ori_sizes in pbar:
        # embeddings: (B, 1, 256, 64, 64) or (B, 256, 64, 64) - pre-computed image embeddings
        # masks: (B, 1, H, W) - ground truth masks
        # ori_sizes: tuple of original sizes

        # Squeeze extra dimension if present (SAMRI v1 embeddings have shape (1, 256, 64, 64))
        if embeddings.dim() == 5:
            embeddings = embeddings.squeeze(1)  # (B, 1, 256, 64, 64) -> (B, 256, 64, 64)

        batch_size = embeddings.shape[0]

        # Convert ori_sizes to list of tuples
        orig_hw_list = [(int(ori_sizes[0][i]), int(ori_sizes[1][i])) for i in range(batch_size)]

        # Move embeddings to device
        embeddings = embeddings.float().to(device)

        # Prepare masks
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # (B, 1, H, W)
        masks = masks.float().to(device)

        # Resize masks to match model output (256x256)
        masks_resized = torch.nn.functional.interpolate(
            masks, size=(256, 256), mode="nearest"
        )

        # Prepare prompts (use original sizes for coordinate transformation)
        point_coords, point_labels, boxes = prepare_prompts_from_embedding(
            masks, args.prompt_mode, args.jitter, device, transforms, orig_hw_list
        )

        # Forward pass
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=False):
            low_res_masks = forward_sam2_embedding(
                model.module if hasattr(model, 'module') else model,
                embeddings, point_coords, point_labels, boxes, device,
                high_res_feats=None  # SAMRI v1 embeddings don't have high-res features
            )

            # Check for NaN
            if torch.isnan(low_res_masks).any():
                print(f"[Rank {rank}] NaN in model output, skipping batch")
                continue

            # Compute loss
            loss = loss_fn(low_res_masks, masks_resized)

            if torch.isnan(loss):
                print(f"[Rank {rank}] NaN loss, skipping batch")
                continue

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        if rank == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return epoch_loss / max(num_batches, 1)


def train_epoch_sam2_embedding(model, dataloader, optimizer, loss_fn, transforms, args, device, rank):
    """
    Train for one epoch using SAM2 pre-computed embeddings with high-res features.

    Args:
        model: SAM2 model (DDP wrapped)
        dataloader: Training data loader (SAM2EmbDataset)
        optimizer: Optimizer
        loss_fn: Loss function
        transforms: SAM2Transforms
        args: Command line arguments
        device: Target device
        rank: Process rank for DDP

    Returns:
        Average epoch loss
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, disable=(rank != 0))
    for image_embed, high_res_feat_0, high_res_feat_1, masks, ori_sizes in pbar:
        # image_embed: (B, 256, 64, 64) - main image embedding
        # high_res_feat_0: (B, 32, 256, 256) - high-res feature level 0
        # high_res_feat_1: (B, 64, 128, 128) - high-res feature level 1
        # masks: (B, 1, H, W) - ground truth masks
        # ori_sizes: tuple of original sizes

        batch_size = image_embed.shape[0]

        # Convert ori_sizes to list of tuples
        orig_hw_list = [(int(ori_sizes[0][i]), int(ori_sizes[1][i])) for i in range(batch_size)]

        # Move to device
        image_embed = image_embed.float().to(device)
        high_res_feat_0 = high_res_feat_0.float().to(device)
        high_res_feat_1 = high_res_feat_1.float().to(device)

        # Prepare masks
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # (B, 1, H, W)
        masks = masks.float().to(device)

        # Resize masks to match model output (256x256)
        masks_resized = torch.nn.functional.interpolate(
            masks, size=(256, 256), mode="nearest"
        )

        # Prepare prompts (use original sizes for coordinate transformation)
        point_coords, point_labels, boxes = prepare_prompts_from_embedding(
            masks, args.prompt_mode, args.jitter, device, transforms, orig_hw_list
        )

        # Forward pass
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=False):
            low_res_masks = forward_sam2_embedding(
                model.module if hasattr(model, 'module') else model,
                image_embed, point_coords, point_labels, boxes, device,
                high_res_feats=(high_res_feat_0, high_res_feat_1)
            )

            # Check for NaN
            if torch.isnan(low_res_masks).any():
                print(f"[Rank {rank}] NaN in model output, skipping batch")
                continue

            # Compute loss
            loss = loss_fn(low_res_masks, masks_resized)

            if torch.isnan(loss):
                print(f"[Rank {rank}] NaN loss, skipping batch")
                continue

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        if rank == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return epoch_loss / max(num_batches, 1)


def main(rank, world_size, args):
    """
    Main training function for each process.

    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Command line arguments
    """
    # Setup DDP
    ddp_setup(rank, world_size)

    # Set random seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Create save directory
    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    # Build SAM2 model
    model = build_sam2(
        config_file=args.sam2_config,
        ckpt_path=args.ckpt_path,
        device=f"cuda:{rank}",
        mode="train",
        apply_postprocessing=False,
    )

    # Freeze all components except mask decoder
    freeze_model_components(model)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Number of GPUs: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Model will be saved to: {args.save_path}")
        print(f"Using pre-computed embeddings: {args.use_embeddings}")

    # Wrap model with DDP
    model = DDP(
        model,
        device_ids=[rank],
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
    )

    # Setup optimizer (only for mask decoder parameters)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Setup loss function
    loss_fn = DiceFocalLoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
        lambda_dice=args.lambda_dice,
        lambda_focal=args.lambda_focal,
    )

    # Setup transforms
    transforms = SAM2Transforms(
        resolution=model.module.image_size if hasattr(model, 'module') else model.image_size,
        mask_threshold=0.0,
    )

    # Setup dataset and dataloader
    if args.use_embeddings:
        if args.sam2_embeddings:
            # Use SAM2-format pre-computed embeddings with high-res features
            train_dataset = SAM2EmbDataset(
                data_root=args.embedding_file,
                sub_set=args.subset,
                resize_mask=True,
                mask_size=256,
                dice_threshold_low=args.dice_threshold_low if args.subset == "custom" else None,
                dice_threshold_high=args.dice_threshold_high if args.subset == "custom" else None,
                min_samples_per_epoch=args.min_samples_per_epoch,
            )
            train_fn = train_epoch_sam2_embedding
            if rank == 0:
                print("Using SAM2 embeddings with high-res features")
                if args.subset == "custom":
                    print(f"  Custom filtering: dice > {args.dice_threshold_low}"
                          + (f" and dice <= {args.dice_threshold_high}" if args.dice_threshold_high else ""))
                print(f"  Original samples: {train_dataset.original_length}")
                if args.min_samples_per_epoch > 0:
                    print(f"  Min samples per epoch: {args.min_samples_per_epoch}")
                    print(f"  Effective samples per epoch: {len(train_dataset)}")
        else:
            # Use SAMRI v1 pre-computed embeddings (deprecated, may not work with SAM2)
            train_dataset = BalancedEmbDataset(
                data_root=args.embedding_file,
                sub_set=args.subset,
                resize_mask=True,
                mask_size=256,
            )
            train_fn = train_epoch_embedding
            if rank == 0:
                print("WARNING: Using SAMRI v1 embeddings without high-res features. "
                      "This may not work with SAM2. Consider using --sam2-embeddings.")
    else:
        # Use raw images
        train_dataset = NiiDataset(
            data_root=args.train_data_paths,
            shuffle=True,
            multi_mask=False,
            with_name=False,
        )
        train_fn = train_epoch

    train_sampler = DistributedSampler(train_dataset)

    # Use custom collate function for raw images (variable sizes)
    collate_fn = variable_size_collate_fn if not args.use_embeddings else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if rank == 0:
        print(f"Training samples: {len(train_dataset)}")

    # Check for existing checkpoint to resume
    ckpt_name, start_epoch = get_checkpoint(args.save_path, args.ckpt_path)
    if start_epoch > 0 and rank == 0:
        print(f"Resuming from epoch {start_epoch}")
        # Load checkpoint
        state_dict = torch.load(ckpt_name, map_location=f"cuda:{rank}")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.module.load_state_dict(state_dict, strict=False)

    # Training loop
    losses = []
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Start training epoch {epoch + 1}/{args.num_epochs}")
        train_sampler.set_epoch(epoch)

        epoch_loss = train_fn(
            model, train_loader, optimizer, loss_fn,
            transforms, args, f"cuda:{rank}", rank
        )

        losses.append(epoch_loss)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 and rank == 0:
            print(f"Saving epoch {epoch + 1}/{args.num_epochs} model, Loss: {epoch_loss:.4f}")

            # Save model state dict
            save_name = os.path.join(
                args.save_path,
                f"sam2_maskdecoder_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {"model": model.module.state_dict(), "epoch": epoch + 1},
                save_name,
            )
            print(f"Checkpoint saved: {save_name}")

    # Cleanup
    destroy_process_group()


if __name__ == "__main__":
    import random as py_random

    args = parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")

    # Setup distributed environment variables before spawning
    os.environ["MASTER_ADDR"] = "localhost"
    if args.master_port is not None:
        os.environ["MASTER_PORT"] = str(args.master_port)
    elif "MASTER_PORT" not in os.environ:
        # Use random port to avoid conflicts
        os.environ["MASTER_PORT"] = str(py_random.randint(29500, 29999))

    print(f"Starting training with {world_size} GPUs")
    print(f"Using MASTER_PORT: {os.environ['MASTER_PORT']}")
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
