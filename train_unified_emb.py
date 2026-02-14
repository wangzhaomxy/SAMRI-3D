#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_embedding_3dwise.py

unified training training from pre-computed 3D embeddings with memory bank.

This script trains SAM2 on a mixture of 2D slices and 3D volumes using
pre-computed embeddings, significantly faster than training from raw images.

=== Training Strategy ===

Trainable Components:
  - Memory Encoder: Encodes predicted masks into memory features
  - Memory Attention: Conditions current frame on past memories
  - Mask Decoder: Generates segmentation masks from prompts

Frozen Components:
  - Image Encoder: Pre-trained Hiera backbone (frozen, embeddings pre-computed)
  - Prompt Encoder: Spatial encoding for points/boxes (domain-invariant)

Sample Mix:
  - ~20% 2D samples: Random slices from 3D volumes processed without memory
  - ~80% 3D samples: Full volumes processed slice-by-slice with memory updates

=== Workflow ===

Step 1: Generate 3D embeddings
  python generate_3d_embeddings.py \\
      --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
      --ckpt-path /path/to/sam2.1_hiera_small.pt \\
      --training-list /path/to/training_list_3d.pkl \\
      --output-dir /path/to/3D_embeddings/ \\
      --dice-threshold 0.2 \\
      --num-workers 8

Step 2: Train from embeddings
  python train_embedding_3dwise.py \\
      --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
      --pretrained-ckpt /path/to/pretrained_checkpoint.pth \\
      --embedding-dir /path/to/3D_embeddings/ \\
      --save-path /path/to/unified_checkpoints/ \\
      --batch-size 2 --num-epochs 50

=== CLI Examples ===

1. Basic training (2x H100 80G GPUs):
python train_embedding_3dwise.py \\
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
--pretrained-ckpt /path/to/pretrained_best.pth \\
--embedding-dir /path/to/3D_embeddings/ \\
--mix-ratio-3d 0.8 \\
--samples-per-dataset 100 \\
--save-path /path/to/unified_checkpoints/ \\
--batch-size 2 --num-epochs 50 --lr 5e-6 \\
--num-workers 16

2. With gradient accumulation for larger effective batch:
   python train_embedding_3dwise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --pretrained-ckpt /path/to/pretrained_best.pth \\
       --embedding-dir /path/to/3D_embeddings/ \\
       --save-path /path/to/unified_checkpoints/ \\
       --batch-size 1 --grad-accum-steps 4 \\
       --num-epochs 50 --lr 5e-6

Key Arguments:
  --pretrained-ckpt           Path to unified (2D mode) checkpoint (mask decoder fine-tuned)
  --embedding-dir         Directory containing 3D embeddings (*_3demb.npz files)
  --mix-ratio-3d          Proportion of 3D samples per epoch (default: 0.8)
  --samples-per-dataset   Volumes per dataset per epoch (default: 100)
  --batch-size            Batch size per GPU (default: 2)
  --grad-accum-steps      Gradient accumulation steps (default: 1)
  --max-slices-per-video  Max slices per video sequence (default: 32)

Logging:
  - Training loss logged to training_log.csv in --save-path every epoch
  - CSV format: epoch, train_loss, loss_2d, loss_3d
  
Example:
python train_embedding_3dwise.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--pretrained-ckpt /path/to/pretrained_best.pth \
--embedding-dir /path/to/3D_embeddings/ \
--mix-ratio-3d 0.8 \
--samples-per-dataset 100 \
--save-path /path/to/unified_checkpoints/ \
--batch-size 2 --num-epochs 50 --lr 5e-6 \
--num-workers 16
"""

import os
import sys
import csv
import argparse
from glob import glob
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

# Prefer local ./sam2
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2
from utils.losses import DiceFocalLoss


# =============================================================================
# Utility Functions
# =============================================================================

def get_latest_checkpoint(save_path: str, initial_ckpt: str) -> Tuple[str, int]:
    """
    Find the latest checkpoint in save_path and return its path and epoch number.
    """
    pattern = os.path.join(save_path, "sam2_emb3d_epoch_*.pth")
    ckpt_files = glob(pattern)

    if not ckpt_files:
        return initial_ckpt, 0

    epochs = []
    for f in ckpt_files:
        try:
            basename = os.path.basename(f)
            epoch_str = basename.replace("sam2_emb3d_epoch_", "").replace(".pth", "")
            epochs.append((int(epoch_str), f))
        except (ValueError, IndexError):
            continue

    if not epochs:
        return initial_ckpt, 0

    epochs.sort(key=lambda x: x[0], reverse=True)
    latest_epoch, latest_ckpt = epochs[0]

    return latest_ckpt, latest_epoch


def mask2d_to_bbox_xyxy(mask2d: np.ndarray) -> Optional[np.ndarray]:
    """Convert binary mask to bounding box in [x0, y0, x1, y1] format."""
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


# =============================================================================
# 3D Embedding Dataset
# =============================================================================

class Embedding3DDataset(Dataset):
    """
    Dataset for loading pre-computed 3D embeddings.

    Each NPZ file contains embeddings for all slices in a 3D volume:
    - image_embeds: (N, 256, 64, 64)
    - high_res_feat_0s: (N, 32, 256, 256)
    - high_res_feat_1s: (N, 64, 128, 128)
    - masks: (N, H, W)
    - cut_idx, v1_slices, v2_slices, labels, etc.
    """

    def __init__(
        self,
        embedding_dir: str,
        samples_per_dataset: int = 100,
        seed: int = 42,
    ):
        super().__init__()
        self.embedding_dir = embedding_dir
        self.samples_per_dataset = samples_per_dataset
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Find all 3D embedding files
        npz_files = sorted(glob(os.path.join(embedding_dir, "**", "*_3demb.npz"), recursive=True))

        if not npz_files:
            raise ValueError(f"No 3D embedding files found in {embedding_dir}")

        # Group by dataset
        self.dataset_files = defaultdict(list)
        for npz_path in npz_files:
            # Extract dataset name from parent directory
            dataset_name = os.path.basename(os.path.dirname(npz_path))
            self.dataset_files[dataset_name].append(npz_path)

        self.dataset_names = sorted(self.dataset_files.keys())

        # Print statistics
        total_files = sum(len(v) for v in self.dataset_files.values())
        print(f"Embedding 3D Dataset:")
        print(f"  Embedding directory: {embedding_dir}")
        print(f"  Total volumes: {total_files}")
        print(f"  Number of datasets: {len(self.dataset_names)}")
        print(f"  Samples per dataset per epoch: {samples_per_dataset}")
        for ds_name in self.dataset_names:
            print(f"    {ds_name}: {len(self.dataset_files[ds_name])} volumes")

        self._build_epoch_indices()

    def _build_epoch_indices(self):
        """Build stratified indices for one epoch."""
        self.epoch_files = []
        for ds_name in self.dataset_names:
            files = self.dataset_files[ds_name]
            n = len(files)
            if n == 0:
                continue
            if n >= self.samples_per_dataset:
                indices = self.rng.choice(n, size=self.samples_per_dataset, replace=False)
            else:
                indices = self.rng.choice(n, size=self.samples_per_dataset, replace=True)

            for idx in indices:
                self.epoch_files.append(files[idx])

        self.rng.shuffle(self.epoch_files)

    def reshuffle(self, epoch: int = None):
        """Reshuffle for a new epoch."""
        if epoch is not None:
            self.rng = np.random.RandomState(self.seed + epoch)
        self._build_epoch_indices()

    def __len__(self):
        return len(self.epoch_files)

    def __getitem__(self, idx):
        npz_path = self.epoch_files[idx]
        data = np.load(npz_path, allow_pickle=True)

        # Load embeddings - keep as float16 to match AMP behavior in raw image training
        # (raw version computes features inside autocast, producing float16 tensors)
        image_embeds = data["image_embeds"]  # float16
        high_res_feat_0s = data["high_res_feat_0s"]  # float16
        high_res_feat_1s = data["high_res_feat_1s"]  # float16
        masks = data["masks"].astype(np.int32)

        # Load metadata
        ori_h = int(data["ori_h"])
        ori_w = int(data["ori_w"])
        n_slices = int(data["n_slices"])
        cut_idx = int(data["cut_idx"])
        labels = data["labels"].tolist()
        v1_slices = data["v1_slices"].tolist()
        v2_slices = data["v2_slices"].tolist()
        dataset = str(data["dataset"])

        return {
            "image_embeds": image_embeds,  # (N, 256, 64, 64) float16
            "high_res_feat_0s": high_res_feat_0s,  # (N, 32, 256, 256) float16
            "high_res_feat_1s": high_res_feat_1s,  # (N, 64, 128, 128) float16
            "masks": masks,  # (N, H, W)
            "ori_h": ori_h,
            "ori_w": ori_w,
            "n_slices": n_slices,
            "cut_idx": cut_idx,
            "labels": labels,
            "v1_slices": v1_slices,
            "v2_slices": v2_slices,
            "dataset": dataset,
            "npz_path": npz_path,
        }


class MixedEmbeddingDataset(Dataset):
    """
    Combined dataset that provides both 2D and 3D samples from 3D embeddings.

    2D samples: Random slices from random volumes (no memory)
    3D samples: Full volumes for memory-based training
    """

    def __init__(
        self,
        embedding_3d_dataset: Embedding3DDataset,
        mix_ratio_3d: float = 0.8,
        seed: int = 42,
    ):
        super().__init__()
        self.emb_dataset = embedding_3d_dataset
        self.mix_ratio_3d = mix_ratio_3d
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Calculate epoch size
        len_3d = len(embedding_3d_dataset)
        self.n_3d = int(len_3d)
        self.n_2d = int(len_3d * (1 - mix_ratio_3d) / mix_ratio_3d) if mix_ratio_3d < 1 else 0

        self.epoch_size = self.n_2d + self.n_3d

        print(f"Mixed Embedding Dataset:")
        print(f"  3D samples per epoch: {self.n_3d}")
        print(f"  2D samples per epoch: {self.n_2d}")
        print(f"  Mix ratio (3D): {mix_ratio_3d:.1%}")

        self._build_epoch_indices()

    def _build_epoch_indices(self):
        """Build mixed indices for one epoch."""
        # 3D samples: use all from epoch
        idx_3d = list(range(len(self.emb_dataset)))

        # 2D samples: random indices into embedding dataset (will extract random slices)
        idx_2d = self.rng.choice(len(self.emb_dataset), size=self.n_2d, replace=True).tolist()

        # Create mixed indices: (is_3d, emb_idx, slice_idx_for_2d)
        self.epoch_indices = []

        # Add 3D samples
        for i in idx_3d:
            self.epoch_indices.append((True, i, None))

        # Add 2D samples (slice_idx will be determined at getitem time)
        for i in idx_2d:
            self.epoch_indices.append((False, i, None))

        self.rng.shuffle(self.epoch_indices)

    def reshuffle(self, epoch: int = None):
        """Reshuffle for a new epoch."""
        if epoch is not None:
            self.rng = np.random.RandomState(self.seed + epoch)
        self.emb_dataset.reshuffle(epoch)
        self._build_epoch_indices()

    def __len__(self):
        return len(self.epoch_indices)

    def __getitem__(self, idx):
        is_3d, emb_idx, _ = self.epoch_indices[idx]
        sample = self.emb_dataset[emb_idx]

        if is_3d:
            sample["is_3d"] = True
            return sample
        else:
            # Extract random 2D slice
            n_slices = sample["n_slices"]
            labels = sample["labels"]

            # Pick a random slice with valid labels
            valid_slices = []
            masks = sample["masks"]
            for z in range(n_slices):
                for label in labels:
                    if (masks[z] == label).sum() > 0:
                        valid_slices.append(z)
                        break

            if not valid_slices:
                valid_slices = list(range(n_slices))

            slice_idx = self.rng.choice(valid_slices)

            # Pick a random label that exists in this slice
            slice_mask = masks[slice_idx]
            valid_labels = [lb for lb in labels if (slice_mask == lb).sum() > 0]
            if not valid_labels:
                valid_labels = labels[:1]
            label = self.rng.choice(valid_labels)

            return {
                "image_embed": sample["image_embeds"][slice_idx],  # (256, 64, 64)
                "high_res_feat_0": sample["high_res_feat_0s"][slice_idx],  # (32, 256, 256)
                "high_res_feat_1": sample["high_res_feat_1s"][slice_idx],  # (64, 128, 128)
                "mask": (slice_mask == label).astype(np.float32),  # (H, W)
                "ori_h": sample["ori_h"],
                "ori_w": sample["ori_w"],
                "label": label,
                "dataset": sample["dataset"],
                "is_3d": False,
            }


def mixed_emb_collate_fn(batch):
    """Custom collate function for mixed embedding dataset."""
    samples_2d = [b for b in batch if not b["is_3d"]]
    samples_3d = [b for b in batch if b["is_3d"]]
    return {"samples_2d": samples_2d, "samples_3d": samples_3d}


# =============================================================================
# Model Component Freezing
# =============================================================================

def freeze_unified_components(model):
    """
    Freeze image encoder and prompt encoder.
    Keep memory encoder, memory attention, and mask decoder trainable.
    """
    # Freeze image encoder
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # Freeze prompt encoder
    for param in model.sam_prompt_encoder.parameters():
        param.requires_grad = False

    # Keep memory attention trainable
    if hasattr(model, 'memory_attention') and model.memory_attention is not None:
        for param in model.memory_attention.parameters():
            param.requires_grad = True

    # Keep memory encoder trainable
    if hasattr(model, 'memory_encoder') and model.memory_encoder is not None:
        for param in model.memory_encoder.parameters():
            param.requires_grad = True

    # Keep mask decoder trainable
    for param in model.sam_mask_decoder.parameters():
        param.requires_grad = True

    # Keep memory-related parameters trainable
    for name, param in model.named_parameters():
        if any(k in name for k in ['maskmem_tpos_enc', 'no_mem_embed', 'no_mem_pos_enc',
                                    'obj_ptr_proj', 'no_obj_ptr', 'mask_downsample']):
            param.requires_grad = True


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# DDP Setup
# =============================================================================

def ddp_setup(rank: int, world_size: int):
    """Setup distributed data parallel training."""
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# =============================================================================
# Training Functions
# =============================================================================

def train_2d_sample_from_embedding(
    model,
    sample: Dict,
    loss_fn,
    args,
    device,
) -> torch.Tensor:
    """
    Train on a single 2D sample using pre-computed embeddings (no memory).
    """
    # Load embeddings
    image_embed = torch.from_numpy(sample["image_embed"]).unsqueeze(0).to(device)  # (1, 256, 64, 64)
    high_res_feat_0 = torch.from_numpy(sample["high_res_feat_0"]).unsqueeze(0).to(device)  # (1, 32, 256, 256)
    high_res_feat_1 = torch.from_numpy(sample["high_res_feat_1"]).unsqueeze(0).to(device)  # (1, 64, 128, 128)

    mask = sample["mask"]  # (H, W) numpy
    ori_h, ori_w = sample["ori_h"], sample["ori_w"]

    # Prepare ground truth mask
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    mask_resized = F.interpolate(mask_tensor, size=(256, 256), mode="nearest")

    # Generate prompt (box)
    box = mask2d_to_bbox_xyxy(mask)
    if box is None:
        return torch.tensor(0.0, device=device)

    # Scale box to image coordinates (assuming model resolution of 1024)
    base_model = model.module if hasattr(model, 'module') else model
    resolution = base_model.image_size
    scale_x = resolution / ori_w
    scale_y = resolution / ori_h
    box_scaled = box.copy()
    box_scaled[0] *= scale_x
    box_scaled[2] *= scale_x
    box_scaled[1] *= scale_y
    box_scaled[3] *= scale_y

    # SAM2 uses point format for box: top-left (label=2) and bottom-right (label=3)
    points = torch.tensor([[
        [box_scaled[0], box_scaled[1]],
        [box_scaled[2], box_scaled[3]],
    ]], dtype=torch.float32, device=device)
    labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)

    point_inputs = {"point_coords": points, "point_labels": labels}

    with torch.amp.autocast("cuda", enabled=args.use_amp):
        # Get prompt embeddings
        sparse_embeddings, dense_embeddings = base_model.sam_prompt_encoder(
            points=(points, labels),
            boxes=None,
            masks=None,
        )

        # Decode mask
        low_res_masks, iou_predictions, _, _ = base_model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=base_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[high_res_feat_0, high_res_feat_1],
        )

        # Compute loss
        loss = loss_fn(low_res_masks, mask_resized)

    return loss


def process_video_with_memory_from_embedding(
    model,
    image_embeds: np.ndarray,
    high_res_feat_0s: np.ndarray,
    high_res_feat_1s: np.ndarray,
    masks: np.ndarray,
    slice_indices: List[int],
    label: int,
    ori_h: int,
    ori_w: int,
    loss_fn,
    device,
    use_amp: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Process a sequence of slices with memory bank using pre-computed embeddings.

    Args:
        model: SAM2 model (unwrapped from DDP)
        image_embeds: (N, 256, 64, 64) embeddings for all slices
        high_res_feat_0s: (N, 32, 256, 256) high-res features
        high_res_feat_1s: (N, 64, 128, 128) high-res features
        masks: (N, H, W) ground truth masks
        slice_indices: List of slice indices to process in order
        label: Target label to segment
        ori_h, ori_w: Original image dimensions
        loss_fn: Loss function
        device: Torch device
        use_amp: Whether to use automatic mixed precision

    Returns:
        total_loss: Accumulated loss over all slices
        num_slices: Number of slices processed
    """
    if len(slice_indices) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    losses = []
    num_slices = 0

    # Memory bank for tracking
    output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }

    num_frames = len(slice_indices)
    resolution = model.image_size

    for frame_idx, slice_z in enumerate(slice_indices):
        is_init_cond_frame = (frame_idx == 0)

        # Get current slice embeddings and mask
        image_embed = torch.from_numpy(image_embeds[slice_z]).unsqueeze(0).to(device)  # (1, 256, 64, 64)
        high_res_0 = torch.from_numpy(high_res_feat_0s[slice_z]).unsqueeze(0).to(device)
        high_res_1 = torch.from_numpy(high_res_feat_1s[slice_z]).unsqueeze(0).to(device)

        mask_2d = (masks[slice_z] == label).astype(np.float32)

        # Skip if no foreground (except for init frame)
        if mask_2d.sum() == 0 and not is_init_cond_frame:
            continue

        # Prepare ground truth
        gt_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0).unsqueeze(0).to(device)
        gt_resized = F.interpolate(gt_tensor, size=(256, 256), mode="nearest")

        # Prepare prompt for initial frame
        point_inputs = None

        if is_init_cond_frame:
            box = mask2d_to_bbox_xyxy(mask_2d.astype(np.uint8))
            if box is None:
                return torch.tensor(0.0, device=device), 0

            scale_x = resolution / ori_w
            scale_y = resolution / ori_h
            box_scaled = box.copy()
            box_scaled[0] *= scale_x
            box_scaled[2] *= scale_x
            box_scaled[1] *= scale_y
            box_scaled[3] *= scale_y

            points = torch.tensor([[
                [box_scaled[0], box_scaled[1]],
                [box_scaled[2], box_scaled[3]],
            ]], dtype=torch.float32, device=device)
            labels_pt = torch.tensor([[2, 3]], dtype=torch.int32, device=device)

            point_inputs = {"point_coords": points, "point_labels": labels_pt}

        with torch.amp.autocast("cuda", enabled=use_amp):
            # Prepare vision features in the format expected by memory conditioning
            # image_embed is (1, 256, 64, 64), need to reshape to (4096, 1, 256) for attention
            B, C, H, W = image_embed.shape
            feat_sizes = [(H, W)]
            vision_feats = [image_embed.flatten(2).permute(2, 0, 1)]  # [(HW, B, C)]
            # Use the FPN neck's PositionEmbeddingSine (matches inference path).
            # Do NOT use sam_prompt_encoder.get_dense_pe() here — that is
            # PositionEmbeddingRandom, intended only for the mask decoder.
            vision_pos_embeds = [model.image_encoder.neck.position_encoding(image_embed).flatten(2).permute(2, 0, 1)]

            # Prepare high-res features
            high_res_features = [high_res_0, high_res_1]

            # Prepare memory-conditioned features
            pix_feat_with_mem = model._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=False,
            )

            # Forward through SAM heads
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=False,
            )

            # Compute loss
            pred_resized = F.interpolate(low_res_masks, size=(256, 256), mode="bilinear", align_corners=False)
            loss = loss_fn(pred_resized, gt_resized)

            if not torch.isnan(loss):
                losses.append(loss)
                num_slices += 1

            # Encode memory for next frames (detached to prevent OOM)
            if model.num_maskmem > 0:
                with torch.no_grad():
                    maskmem_features, maskmem_pos_enc = model._encode_new_memory(
                        current_vision_feats=[v.detach() for v in vision_feats],
                        feat_sizes=feat_sizes,
                        pred_masks_high_res=high_res_masks.detach(),
                        object_score_logits=object_score_logits.detach(),
                        is_mask_from_pts=(point_inputs is not None),
                    )

                current_out = {
                    "maskmem_features": maskmem_features,
                    "maskmem_pos_enc": maskmem_pos_enc,
                    "pred_masks": low_res_masks.detach(),
                    "obj_ptr": obj_ptr.detach(),
                }

                if is_init_cond_frame:
                    output_dict["cond_frame_outputs"][frame_idx] = current_out
                else:
                    output_dict["non_cond_frame_outputs"][frame_idx] = current_out

    if losses:
        total_loss = torch.stack(losses).sum()
        return total_loss, num_slices
    return torch.tensor(0.0, device=device, requires_grad=True), 0


def train_3d_sample_from_embedding(
    model,
    sample: Dict,
    loss_fn,
    args,
    device,
) -> torch.Tensor:
    """
    Train on a 3D volume using memory bank with pre-computed embeddings.

    Note: To prevent OOM from gradient accumulation, we limit the number of
    labels processed per sample. Each label creates ~64 forward passes through
    memory attention, and gradient graphs are kept until backward() is called.
    """
    image_embeds = sample["image_embeds"]  # (N, 256, 64, 64)
    high_res_feat_0s = sample["high_res_feat_0s"]
    high_res_feat_1s = sample["high_res_feat_1s"]
    masks = sample["masks"]  # (N, H, W)
    labels = sample["labels"]
    cut_idx = sample["cut_idx"]
    v1_slices = sample["v1_slices"]
    v2_slices = sample["v2_slices"]
    ori_h, ori_w = sample["ori_h"], sample["ori_w"]

    base_model = model.module if hasattr(model, 'module') else model

    # Subsample video slices if too long
    max_slices = getattr(args, 'max_slices_per_video', 32)

    def subsample_slices(slices: List[int], max_len: int) -> List[int]:
        if len(slices) <= max_len:
            return slices
        step = max(1, (len(slices) - 1) // (max_len - 1))
        subsampled = [slices[0]] + slices[1::step][:max_len - 1]
        return subsampled

    v1_slices_train = subsample_slices(v1_slices, max_slices)
    v2_slices_train = subsample_slices(v2_slices, max_slices)

    all_losses = []
    total_slices = 0

    # Limit labels per sample to prevent OOM from gradient accumulation
    # Each label creates ~64 forward passes with gradient graphs
    max_labels_per_sample = getattr(args, 'max_labels_per_sample', 1)
    if len(labels) > max_labels_per_sample:
        # Randomly select labels for this sample
        rng = np.random.RandomState()
        labels = rng.choice(labels, size=max_labels_per_sample, replace=False).tolist()

    for label in labels:
        # Check if label exists in cut slice
        if (masks[cut_idx] == label).sum() == 0:
            continue

        # Process video 1 (cut -> 0)
        if len(v1_slices_train) > 0:
            loss1, n1 = process_video_with_memory_from_embedding(
                model=base_model,
                image_embeds=image_embeds,
                high_res_feat_0s=high_res_feat_0s,
                high_res_feat_1s=high_res_feat_1s,
                masks=masks,
                slice_indices=v1_slices_train,
                label=label,
                ori_h=ori_h,
                ori_w=ori_w,
                loss_fn=loss_fn,
                device=device,
                use_amp=args.use_amp,
            )
            if n1 > 0 and loss1.requires_grad:
                all_losses.append(loss1)
                total_slices += n1

        # Process video 2 (cut -> end), skip cut slice
        if len(v2_slices_train) > 1:
            v2_remaining = v2_slices_train[1:]
            loss2, n2 = process_video_with_memory_from_embedding(
                model=base_model,
                image_embeds=image_embeds,
                high_res_feat_0s=high_res_feat_0s,
                high_res_feat_1s=high_res_feat_1s,
                masks=masks,
                slice_indices=v2_remaining,
                label=label,
                ori_h=ori_h,
                ori_w=ori_w,
                loss_fn=loss_fn,
                device=device,
                use_amp=args.use_amp,
            )
            if n2 > 0 and loss2.requires_grad:
                all_losses.append(loss2)
                total_slices += n2

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if all_losses and total_slices > 0:
        total_loss = torch.stack(all_losses).sum() / total_slices
        return total_loss
    return torch.tensor(0.0, device=device, requires_grad=True)


def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    args,
    device,
    rank,
    scaler=None,
):
    """Train for one epoch on mixed 2D/3D embedding data."""
    model.train()

    epoch_loss = 0.0
    loss_2d_total = 0.0
    loss_3d_total = 0.0
    num_2d = 0
    num_3d = 0

    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0))

    for batch_idx, batch in enumerate(pbar):
        samples_2d = batch["samples_2d"]
        samples_3d = batch["samples_3d"]

        # Track valid losses to avoid scaler issues with empty batches
        batch_losses = []
        batch_num_2d = 0
        batch_num_3d = 0

        # Process 2D samples
        for sample in samples_2d:
            loss = train_2d_sample_from_embedding(model, sample, loss_fn, args, device)
            if not torch.isnan(loss) and loss.item() > 0:
                batch_losses.append(loss)
                loss_2d_total += loss.item()
                batch_num_2d += 1
                num_2d += 1

        # Process 3D samples
        for sample in samples_3d:
            loss = train_3d_sample_from_embedding(model, sample, loss_fn, args, device)
            if not torch.isnan(loss) and loss.item() > 0:
                batch_losses.append(loss)
                loss_3d_total += loss.item()
                batch_num_3d += 1
                num_3d += 1

        # Skip if no valid losses in batch (prevents scaler assertion error)
        if not batch_losses:
            if rank == 0:
                pbar.set_postfix({
                    "loss": "N/A",
                    "2d": num_2d,
                    "3d": num_3d,
                })
            continue

        # Compute batch loss from valid samples
        batch_loss = torch.stack(batch_losses).mean()

        # Gradient accumulation
        batch_loss = batch_loss / args.grad_accum_steps

        if args.use_amp and scaler is not None:
            scaler.scale(batch_loss).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            batch_loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss += batch_loss.item() * args.grad_accum_steps

        if rank == 0:
            pbar.set_postfix({
                "loss": f"{batch_loss.item() * args.grad_accum_steps:.4f}",
                "2d": num_2d,
                "3d": num_3d,
            })

    # Final gradient step if needed
    if len(dataloader) % args.grad_accum_steps != 0:
        if args.use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = epoch_loss / max(len(dataloader), 1)
    avg_loss_2d = loss_2d_total / max(num_2d, 1)
    avg_loss_3d = loss_3d_total / max(num_3d, 1)

    return avg_loss, avg_loss_2d, avg_loss_3d


# =============================================================================
# Main Training Function
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="unified training: Train SAM2 from 3D embeddings with memory")

    # Model configuration
    parser.add_argument("--sam2-config", type=str, required=True,
                        help="Path to SAM2 config file")
    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="Path to unified (2D mode) checkpoint")

    # Embedding data
    parser.add_argument("--embedding-dir", type=str, required=True,
                        help="Directory containing 3D embeddings (*_3demb.npz files)")

    # Sampling configuration
    parser.add_argument("--mix-ratio-3d", type=float, default=0.8,
                        help="Proportion of 3D samples per epoch (default: 0.8)")
    parser.add_argument("--samples-per-dataset", type=int, default=100,
                        help="Volumes per dataset per epoch (default: 100)")
    parser.add_argument("--max-slices-per-video", type=int, default=32,
                        help="Max slices per video sequence during training (default: 32)")
    parser.add_argument("--max-labels-per-sample", type=int, default=1,
                        help="Max labels per 3D sample to prevent OOM (default: 1)")

    # Training configuration
    parser.add_argument("--save-path", type=str, required=True,
                        help="Path to save checkpoints")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per GPU (default: 2)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (default: 1.0)")

    # Loss configuration
    parser.add_argument("--lambda-dice", type=float, default=1.0,
                        help="Weight for dice loss (default: 1.0)")
    parser.add_argument("--lambda-focal", type=float, default=1.0,
                        help="Weight for focal loss (default: 1.0)")

    # Hardware configuration
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of data loading workers (default: 16)")
    parser.add_argument("--use-amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--master-port", type=int, default=None,
                        help="Master port for DDP")

    # Checkpointing
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs (default: 5)")

    return parser.parse_args()


def main(rank, world_size, args):
    """Main training function for each process."""
    # Setup DDP
    ddp_setup(rank, world_size)
    device = f"cuda:{rank}"

    # Set random seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Create save directory
    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    # Check for existing checkpoint to resume from
    ckpt_path, start_epoch = get_latest_checkpoint(args.save_path, args.pretrained_ckpt)
    resume_training = start_epoch > 0

    if rank == 0:
        if resume_training:
            print(f"Resuming from epoch {start_epoch}, checkpoint: {ckpt_path}")
        else:
            print(f"Starting fresh training from unified (2D mode) checkpoint: {ckpt_path}")

    # Build SAM2 model
    model = build_sam2(
        config_file=args.sam2_config,
        ckpt_path=ckpt_path,
        device=device,
        mode="train",
        apply_postprocessing=False,
    )

    # Freeze appropriate components
    freeze_unified_components(model)

    if rank == 0:
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Number of GPUs: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.grad_accum_steps}")
        print(f"Effective batch size: {args.batch_size * world_size * args.grad_accum_steps}")

    # Wrap model with DDP
    model = DDP(
        model,
        device_ids=[rank],
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
    )

    # Setup optimizer
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

    # Setup AMP scaler
    scaler = torch.amp.GradScaler(device) if args.use_amp else None

    # Setup datasets
    if rank == 0:
        print("\nLoading 3D embedding dataset...")

    emb_3d_dataset = Embedding3DDataset(
        embedding_dir=args.embedding_dir,
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
    )

    if rank == 0:
        print("\nCreating mixed dataset...")

    mixed_dataset = MixedEmbeddingDataset(
        embedding_3d_dataset=emb_3d_dataset,
        mix_ratio_3d=args.mix_ratio_3d,
        seed=args.seed,
    )

    # Create DistributedSampler
    train_sampler = DistributedSampler(
        mixed_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )

    if rank == 0:
        total_samples = len(mixed_dataset)
        samples_per_gpu = len(train_sampler)
        print(f"Total samples: {total_samples}, samples per GPU: {samples_per_gpu}")

    # Dataloader
    train_loader = DataLoader(
        mixed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=mixed_emb_collate_fn,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Setup CSV logging
    csv_path = os.path.join(args.save_path, "training_log.csv")
    csv_file = None
    csv_writer = None
    if rank == 0:
        csv_exists = os.path.exists(csv_path) and resume_training
        csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow(["epoch", "train_loss", "loss_2d", "loss_3d"])
        csv_file.flush()
        print(f"\nLogging to: {csv_path}" + (" (appending)" if csv_exists else ""))

    # Training loop
    if rank == 0:
        if resume_training:
            print(f"\nResuming training from epoch {start_epoch + 1} to {args.num_epochs}...")
        else:
            print(f"\nStarting training for {args.num_epochs} epochs...")

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"{'='*60}")

        mixed_dataset.reshuffle(epoch)

        avg_loss, avg_loss_2d, avg_loss_3d = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            args=args,
            device=device,
            rank=rank,
            scaler=scaler,
        )

        if rank == 0:
            print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, 2D={avg_loss_2d:.4f}, 3D={avg_loss_3d:.4f}")

            # Log to CSV
            if csv_writer is not None:
                csv_writer.writerow([
                    epoch + 1,
                    f"{avg_loss:.6f}",
                    f"{avg_loss_2d:.6f}",
                    f"{avg_loss_3d:.6f}",
                ])
                csv_file.flush()

            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                save_name = os.path.join(
                    args.save_path,
                    f"sam2_emb3d_epoch_{epoch + 1}.pth"
                )
                torch.save({
                    "model": model.module.state_dict(),
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                }, save_name)
                print(f"Checkpoint saved: {save_name}")

    # Save final checkpoint
    if rank == 0:
        save_name = os.path.join(args.save_path, "sam2_emb3d_final.pth")
        torch.save({
            "model": model.module.state_dict(),
            "epoch": args.num_epochs,
            "optimizer": optimizer.state_dict(),
        }, save_name)
        print(f"Final checkpoint saved: {save_name}")

        if csv_file is not None:
            csv_file.close()

    # Cleanup
    destroy_process_group()


if __name__ == "__main__":
    import random as py_random

    args = parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")

    # Setup distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    if args.master_port is not None:
        os.environ["MASTER_PORT"] = str(args.master_port)
    elif "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(py_random.randint(29500, 29999))

    print(f"Starting embedding-based unified training training with {world_size} GPUs")
    print(f"Using MASTER_PORT: {os.environ['MASTER_PORT']}")

    mp.spawn(main, args=(world_size, args), nprocs=world_size)
