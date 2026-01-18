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
2. Pre-computed embeddings (BalancedEmbDataset): Load pre-computed embeddings for faster training

Based on SAMRI v1 training approach adapted for SAM2 architecture.
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
from utils.dataloader import NiiDataset, BalancedEmbDataset
from utils.losses import DiceFocalLoss
from utils.utils import gen_points, gen_bboxes, get_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Slice-wise SAM2 mask decoder training")

    # Model configuration
    parser.add_argument("--sam2-config", type=str, required=True,
                        help="Path to SAM2 config file (e.g., configs/sam2.1/sam2.1_hiera_s.yaml)")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to SAM2 checkpoint")

    # Data configuration
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Use pre-computed embeddings (BalancedEmbDataset) instead of raw images")
    parser.add_argument("--embedding-file", type=str, default=None,
                        help="Path to pickle file containing embedding file list (required if --use-embeddings)")
    parser.add_argument("--subset", type=str, default="all",
                        choices=["60_up", "60_down", "all"],
                        help="Subset to use from embedding file (default: all)")
    parser.add_argument("--train-data-paths", type=str, nargs="+", default=None,
                        help="List of training data paths (for raw image mode)")
    parser.add_argument("--val-data-paths", type=str, nargs="+", default=None,
                        help="List of validation data paths (optional)")

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
    master_port = os.environ.get("MASTER_PORT", "29500")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
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


def forward_sam2_embedding(model, embeddings, point_coords, point_labels, boxes, device):
    """
    Forward pass through SAM2 using pre-computed embeddings (skip image encoder).

    Args:
        model: SAM2 model
        embeddings: Pre-computed image embeddings (B, 256, 64, 64)
        point_coords: Point coordinates or None
        point_labels: Point labels or None
        boxes: Box coordinates or None
        device: Target device

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
    # Note: For pre-computed embeddings, we don't have high-res features
    # Set high_res_features to None or empty list
    all_low_res_masks = []
    for i in range(batch_size):
        # Predict masks without high-res features
        low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embed[i:i+1],
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings[i:i+1],
            dense_prompt_embeddings=dense_embeddings[i:i+1],
            multimask_output=False,
            repeat_image=False,
            high_res_features=None,
        )
        all_low_res_masks.append(low_res_masks)

    # Stack all masks
    low_res_masks = torch.cat(all_low_res_masks, dim=0)  # (B, 1, 256, 256)

    return low_res_masks


def train_epoch(model, dataloader, optimizer, loss_fn, transforms, args, device, rank):
    """
    Train for one epoch (raw image mode).

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
    for images, masks in pbar:
        # images: (B, H, W, 3) uint8
        # masks: (B, 1, H, W) or (B, H, W)

        batch_size = images.shape[0]
        orig_hw = (images.shape[1], images.shape[2])

        # Transform images using SAM2Transforms
        # Convert from HWC uint8 to transformed tensor
        images_list = [img.numpy() for img in images]
        images_tensor = transforms.forward_batch(images_list).to(device)

        # Prepare masks
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # (B, 1, H, W)
        masks = masks.float().to(device)

        # Resize masks to match model output (256x256)
        masks_resized = torch.nn.functional.interpolate(
            masks, size=(256, 256), mode="nearest"
        )

        # Prepare prompts
        point_coords, point_labels, boxes = prepare_prompts(
            masks, args.prompt_mode, args.jitter, device, transforms, orig_hw
        )

        # Forward pass
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=False):
            low_res_masks = forward_sam2_image(
                model.module if hasattr(model, 'module') else model,
                images_tensor, point_coords, point_labels, boxes,
                transforms, orig_hw
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


def train_epoch_embedding(model, dataloader, optimizer, loss_fn, transforms, args, device, rank):
    """
    Train for one epoch using pre-computed embeddings.

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
        # embeddings: (B, 256, 64, 64) - pre-computed image embeddings
        # masks: (B, 1, H, W) - ground truth masks
        # ori_sizes: tuple of original sizes

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
                embeddings, point_coords, point_labels, boxes, device
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
        # Use pre-computed embeddings
        train_dataset = BalancedEmbDataset(
            data_root=args.embedding_file,
            sub_set=args.subset,
            resize_mask=True,
            mask_size=256,
        )
        train_fn = train_epoch_embedding
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
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
        train_sampler.set_epoch(epoch)

        epoch_loss = train_fn(
            model, train_loader, optimizer, loss_fn,
            transforms, args, f"cuda:{rank}", rank
        )

        losses.append(epoch_loss)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 and rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss:.4f}")

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
    args = parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")

    print(f"Starting training with {world_size} GPUs")
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
