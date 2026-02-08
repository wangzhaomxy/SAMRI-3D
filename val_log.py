# -*- coding: utf-8 -*-
"""
Validation Logger: Validate checkpoints on 3D validation datasets using pre-computed embeddings.

This script validates trained SAM2 checkpoints on validation datasets using pre-computed
3D embeddings (from generate_3d_embeddings.py) and logs results to CSV files.

Key Features:
- Uses pre-computed embeddings for fast validation (no image encoder forward pass)
- Multi-object 3D validation with memory bank propagation
- Multiple checkpoint evaluation with resume support
- Parallel processing for efficiency

=== CLI Examples ===

1. Validate all checkpoints on 3D validation embeddings:
python val_log.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-folder /scratch/user/s4670484/Model_save/samri2_stage2_emb \
--embedding-dir /scratch/user/s4670484/SAM2_val_embeddings/ \
--output-path /scratch/user/s4670484/validation_results/samri2_stage2_emb/ \
--prompt-mode box \
--device cuda:0 \
--max-samples-per-dataset 100

2. Validate a single checkpoint:
   python val_log.py \
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
       --ckpt-path /scratch/user/s4670484/Model_save/sam2_emb3d_epoch_50.pth \
       --embedding-dir /scratch/user/s4670484/SAM2_embeddings_val/ \
       --output-path /scratch/user/s4670484/validation_results/ \
       --device cuda:0

3. Multi-GPU validation:
   python val_log.py \
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
       --ckpt-folder /scratch/user/s4670484/Model_save/samri2_stage2/ \
       --embedding-dir /scratch/user/s4670484/SAM2_embeddings_val/ \
       --output-path /scratch/user/s4670484/validation_results/ \
       --devices cuda:0 cuda:1 cuda:2 \
       --num-workers 3

Key Arguments:
  --ckpt-folder           Folder containing multiple checkpoint .pth files
  --ckpt-path             Single checkpoint path (alternative to --ckpt-folder)
  --embedding-dir         Directory containing 3D embeddings (*_3demb.npz files)
  --prompt-mode           Prompt type: "box" or "mask" (default: box)
  --output-path           Output directory for CSV results
  --max-samples-per-dataset  Max volumes per dataset (default: 100)

Output:
  - validation_results_3d.csv: Per-volume dice scores
  - validation_summary.csv: Mean dice per checkpoint and dataset
"""

from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import multiprocessing as mp
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Prefer local ./sam2
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2


# =============================================================================
# Utility Functions
# =============================================================================

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice score between two binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.sum(pred & gt)
    if pred.sum() + gt.sum() == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / (pred.sum() + gt.sum())


def mask2d_to_bbox_xyxy(mask: np.ndarray) -> Optional[np.ndarray]:
    """Convert 2D binary mask to bounding box [x1, y1, x2, y2]."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def get_checkpoint_paths(ckpt_folder: str = None, ckpt_path: str = None) -> List[str]:
    """Get list of checkpoint paths to validate."""
    if ckpt_path:
        return [ckpt_path]

    if ckpt_folder:
        ckpts = sorted(glob(os.path.join(ckpt_folder, "*.pth")))
        ckpts += sorted(glob(os.path.join(ckpt_folder, "*.pt")))
        return ckpts

    return []


# =============================================================================
# 3D Embedding Dataset for Validation
# =============================================================================

class Embedding3DValDataset:
    """
    Dataset for loading pre-computed 3D validation embeddings.

    Each NPZ file contains:
    - image_embeds: (N, 256, 64, 64)
    - high_res_feat_0s: (N, 32, 256, 256)
    - high_res_feat_1s: (N, 64, 128, 128)
    - masks: (N, H, W)
    - cut_idx, v1_slices, v2_slices, labels, dataset, etc.
    """

    def __init__(
        self,
        embedding_dir: str,
        max_samples_per_dataset: Optional[int] = None,
        seed: int = 42,
    ):
        self.embedding_dir = embedding_dir
        self.max_samples_per_dataset = max_samples_per_dataset
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Find all 3D embedding files
        npz_files = sorted(glob(os.path.join(embedding_dir, "**", "*_3demb.npz"), recursive=True))

        if not npz_files:
            raise ValueError(f"No 3D embedding files found in {embedding_dir}")

        # Group by dataset
        self.dataset_files = defaultdict(list)
        for npz_path in npz_files:
            dataset_name = os.path.basename(os.path.dirname(npz_path))
            self.dataset_files[dataset_name].append(npz_path)

        self.dataset_names = sorted(self.dataset_files.keys())

        # Sample files per dataset (use all if max_samples_per_dataset is None or 0)
        self.files = []
        for ds_name in self.dataset_names:
            files = self.dataset_files[ds_name]
            n = len(files)
            if n == 0:
                continue
            if max_samples_per_dataset and n > max_samples_per_dataset:
                indices = self.rng.choice(n, size=max_samples_per_dataset, replace=False)
                files = [files[i] for i in indices]
            self.files.extend([(ds_name, f) for f in files])

        # Print statistics
        print(f"Embedding 3D Validation Dataset:")
        print(f"  Embedding directory: {embedding_dir}")
        print(f"  Total volumes: {len(self.files)}")
        print(f"  Number of datasets: {len(self.dataset_names)}")
        for ds_name in self.dataset_names:
            n_total = len(self.dataset_files[ds_name])
            n_sampled = sum(1 for d, _ in self.files if d == ds_name)
            print(f"    {ds_name}: {n_sampled}/{n_total} volumes")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ds_name, npz_path = self.files[idx]
        data = np.load(npz_path, allow_pickle=True)

        return {
            "dataset": ds_name,
            "npz_path": npz_path,
            "image_embeds": data["image_embeds"],  # (N, 256, 64, 64) float16
            "high_res_feat_0s": data["high_res_feat_0s"],  # (N, 32, 256, 256) float16
            "high_res_feat_1s": data["high_res_feat_1s"],  # (N, 64, 128, 128) float16
            "masks": data["masks"].astype(np.int32),  # (N, H, W)
            "ori_h": int(data["ori_h"]),
            "ori_w": int(data["ori_w"]),
            "n_slices": int(data["n_slices"]),
            "cut_idx": int(data["cut_idx"]),
            "labels": data["labels"].tolist(),
            "v1_slices": data["v1_slices"].tolist(),
            "v2_slices": data["v2_slices"].tolist(),
        }


# =============================================================================
# 3D Validation with Embeddings
# =============================================================================

def sample_positive_points(mask2d: np.ndarray, num_pos: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Sample positive points from a binary mask."""
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int32)
    k = min(num_pos, xs.size)
    idx = np.random.choice(xs.size, size=k, replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    lbls = np.ones((k,), dtype=np.int32)
    return pts, lbls


@torch.no_grad()
def validate_3d_volume_from_embedding(
    model,
    sample: Dict,
    prompt_mode: str,
    device: str,
) -> List[Dict]:
    """
    Validate a 3D volume using pre-computed embeddings with memory bank propagation.

    This mirrors the training path in train_embedding_3dwise.py but collects predictions
    instead of computing loss.

    Returns:
        List of {label, dice, voxel_count} for each label
    """
    image_embeds = sample["image_embeds"]  # (N, 256, 64, 64)
    high_res_feat_0s = sample["high_res_feat_0s"]  # (N, 32, 256, 256)
    high_res_feat_1s = sample["high_res_feat_1s"]  # (N, 64, 128, 128)
    masks = sample["masks"]  # (N, H, W)
    labels = sample["labels"]
    cut_idx = sample["cut_idx"]
    v1_slices = sample["v1_slices"]
    v2_slices = sample["v2_slices"]
    ori_h, ori_w = sample["ori_h"], sample["ori_w"]
    n_slices = sample["n_slices"]

    if not labels:
        return []

    results = []
    resolution = model.image_size

    for label in labels:
        # Check if label exists in cut slice
        if (masks[cut_idx] == label).sum() == 0:
            continue

        # Collect predictions for this label across all slices
        pred_volume = np.zeros((n_slices, ori_h, ori_w), dtype=np.uint8)

        # Process video 1 (cut -> 0)
        if len(v1_slices) > 0:
            preds_v1 = propagate_video_from_embedding(
                model=model,
                image_embeds=image_embeds,
                high_res_feat_0s=high_res_feat_0s,
                high_res_feat_1s=high_res_feat_1s,
                masks=masks,
                slice_indices=v1_slices,
                label=label,
                ori_h=ori_h,
                ori_w=ori_w,
                resolution=resolution,
                prompt_mode=prompt_mode,
                device=device,
            )
            for frame_idx, slice_z in enumerate(v1_slices):
                if frame_idx in preds_v1:
                    pred_volume[slice_z] = preds_v1[frame_idx]

        # Process video 2 (cut -> end), skip cut slice to avoid double-writing
        if len(v2_slices) > 1:
            v2_remaining = v2_slices[1:]  # Skip cut slice
            preds_v2 = propagate_video_from_embedding(
                model=model,
                image_embeds=image_embeds,
                high_res_feat_0s=high_res_feat_0s,
                high_res_feat_1s=high_res_feat_1s,
                masks=masks,
                slice_indices=v2_remaining,
                label=label,
                ori_h=ori_h,
                ori_w=ori_w,
                resolution=resolution,
                prompt_mode=prompt_mode,
                device=device,
            )
            for frame_idx, slice_z in enumerate(v2_remaining):
                if frame_idx in preds_v2:
                    pred_volume[slice_z] = preds_v2[frame_idx]

        # Compute dice for this label
        gt_lb = (masks == label).astype(np.uint8)
        dsc = dice_score(pred_volume, gt_lb)

        results.append({
            "label": int(label),
            "dice": float(dsc),
            "voxel_count": int(gt_lb.sum()),
        })

    return results


def propagate_video_from_embedding(
    model,
    image_embeds: np.ndarray,
    high_res_feat_0s: np.ndarray,
    high_res_feat_1s: np.ndarray,
    masks: np.ndarray,
    slice_indices: List[int],
    label: int,
    ori_h: int,
    ori_w: int,
    resolution: int,
    prompt_mode: str,
    device: str,
) -> Dict[int, np.ndarray]:
    """
    Propagate through a video sequence using pre-computed embeddings.

    Returns:
        Dict mapping frame_idx to predicted binary mask (H, W)
    """
    if len(slice_indices) == 0:
        return {}

    predictions = {}

    # Memory bank for tracking
    output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }

    num_frames = len(slice_indices)

    for frame_idx, slice_z in enumerate(slice_indices):
        is_init_cond_frame = (frame_idx == 0)

        # Get current slice embeddings (convert float16 -> float32 to match model)
        image_embed = torch.from_numpy(image_embeds[slice_z]).unsqueeze(0).float().to(device)  # (1, 256, 64, 64)
        high_res_0 = torch.from_numpy(high_res_feat_0s[slice_z]).unsqueeze(0).float().to(device)
        high_res_1 = torch.from_numpy(high_res_feat_1s[slice_z]).unsqueeze(0).float().to(device)

        mask_2d = (masks[slice_z] == label).astype(np.float32)

        # Prepare prompt for initial frame
        point_inputs = None

        if is_init_cond_frame:
            if prompt_mode == "box":
                box = mask2d_to_bbox_xyxy(mask_2d.astype(np.uint8))
                if box is None:
                    return {}

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

            else:  # mask mode - sample points
                pts, lbls = sample_positive_points(mask_2d.astype(np.uint8), num_pos=256)
                if pts.shape[0] == 0:
                    return {}

                # Scale points to model resolution
                scale_x = resolution / ori_w
                scale_y = resolution / ori_h
                pts_scaled = pts.copy()
                pts_scaled[:, 0] *= scale_x
                pts_scaled[:, 1] *= scale_y

                points = torch.from_numpy(pts_scaled).unsqueeze(0).to(device)
                labels_pt = torch.from_numpy(lbls).unsqueeze(0).to(device)

                point_inputs = {"point_coords": points, "point_labels": labels_pt}

        # Prepare vision features
        B, C, H, W = image_embed.shape
        feat_sizes = [(H, W)]
        vision_feats = [image_embed.flatten(2).permute(2, 0, 1)]  # [(HW, B, C)]
        vision_pos_embeds = [model.image_encoder.neck.position_encoding(image_embed).flatten(2).permute(2, 0, 1)]

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

        # Get prediction at original resolution
        pred_logits = F.interpolate(
            high_res_masks,
            size=(ori_h, ori_w),
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = (pred_logits > 0).squeeze().cpu().numpy().astype(np.uint8)
        predictions[frame_idx] = pred_mask

        # Encode memory for next frames
        if model.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = model._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )

            current_out = {
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": maskmem_pos_enc,
                "pred_masks": low_res_masks,
                "obj_ptr": obj_ptr,
            }

            if is_init_cond_frame:
                output_dict["cond_frame_outputs"][frame_idx] = current_out
            else:
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out

    return predictions


def validate_3d_dataset_from_embeddings(
    sam2_config: str,
    ckpt_path: str,
    embedding_dir: str,
    prompt_mode: str,
    device: str = "cuda:0",
    max_samples_per_dataset: Optional[int] = None,
    sample_seed: int = 42,
) -> List[Dict]:
    """
    Validate a checkpoint on 3D validation datasets using pre-computed embeddings.

    Returns:
        List of results with dataset, npz_path, label, dice, voxel_count
    """
    # Build model
    model = build_sam2(sam2_config, ckpt_path, device=device, mode="eval")
    model.eval()

    # Load validation dataset
    val_dataset = Embedding3DValDataset(
        embedding_dir=embedding_dir,
        max_samples_per_dataset=max_samples_per_dataset,
        seed=sample_seed,
    )

    all_results = []

    for idx in tqdm(range(len(val_dataset)), desc="Validating", leave=False):
        sample = val_dataset[idx]
        ds_name = sample["dataset"]
        npz_path = sample["npz_path"]

        try:
            vol_results = validate_3d_volume_from_embedding(
                model=model,
                sample=sample,
                prompt_mode=prompt_mode,
                device=device,
            )
            for r in vol_results:
                r["dataset"] = ds_name
                r["npz_path"] = npz_path
            all_results.extend(vol_results)

        except Exception as e:
            print(f"  Warning: Failed to process {npz_path}: {e}")

    return all_results


# =============================================================================
# Main Validation Functions
# =============================================================================

def compute_summary(results_3d: List[Dict], ckpt_name: str) -> List[Dict]:
    """Compute summary statistics per dataset."""
    summary = []

    if results_3d:
        by_dataset = defaultdict(list)
        for r in results_3d:
            by_dataset[r["dataset"]].append(r["dice"])

        for ds_name, dices in by_dataset.items():
            summary.append({
                "checkpoint": ckpt_name,
                "dataset": ds_name,
                "mean_dice": float(np.mean(dices)),
                "std_dice": float(np.std(dices)),
                "num_samples": len(dices),
            })

    return summary


def validate_checkpoint_worker(
    sam2_config: str,
    ckpt_path: str,
    embedding_dir: str,
    prompt_mode: str,
    device: str,
    max_samples_per_dataset: Optional[int],
    sample_seed: int,
) -> Tuple[str, List[Dict], List[Dict]]:
    """Worker wrapper for checkpoint validation (for multiprocessing)."""
    ckpt_name = os.path.basename(ckpt_path)

    results_3d = validate_3d_dataset_from_embeddings(
        sam2_config=sam2_config,
        ckpt_path=ckpt_path,
        embedding_dir=embedding_dir,
        prompt_mode=prompt_mode,
        device=device,
        max_samples_per_dataset=max_samples_per_dataset,
        sample_seed=sample_seed,
    )

    for r in results_3d:
        r["checkpoint"] = ckpt_name

    summary = compute_summary(results_3d, ckpt_name)
    return ckpt_path, results_3d, summary


def append_results_to_csv(results: List[Dict], csv_path: str):
    """Append results to CSV file, creating it with a header if missing."""
    if not results:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    keys = list(results[0].keys())
    file_exists = os.path.exists(csv_path)
    write_header = (not file_exists) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            writer.writeheader()
        writer.writerows(results)


def load_existing_checkpoints(output_path: str) -> set:
    """Load checkpoint names already present in result/summary CSVs."""
    existing = set()
    csv_paths = [
        os.path.join(output_path, "validation_summary.csv"),
        os.path.join(output_path, "validation_results_3d.csv"),
    ]
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                if "checkpoint" not in (reader.fieldnames or []):
                    continue
                for row in reader:
                    ckpt = row.get("checkpoint")
                    if ckpt:
                        existing.add(ckpt)
        except Exception as e:
            print(f"  Warning: Failed to read {csv_path}: {e}")
    return existing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate checkpoints on 3D validation datasets using pre-computed embeddings"
    )

    # Model configuration
    parser.add_argument("--sam2-config", type=str, required=True,
                        help="Path to SAM2 config file")

    # Checkpoint configuration
    parser.add_argument("--ckpt-folder", type=str, default=None,
                        help="Folder containing checkpoint .pth files")
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="Single checkpoint path (alternative to --ckpt-folder)")

    # Data configuration
    parser.add_argument("--embedding-dir", type=str, required=True,
                        help="Directory containing 3D embeddings (*_3demb.npz files)")

    # Validation configuration
    parser.add_argument("--prompt-mode", type=str, default="box",
                        choices=["box", "mask"],
                        help="Prompt mode (default: box)")

    # Output configuration
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output directory for CSV results")

    # Other
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--devices", type=str, nargs="+", default=None,
                        help="Device list for parallel validation (e.g., cuda:0 cuda:1)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers for checkpoint-level validation")
    parser.add_argument("--max-samples-per-dataset", type=int, default=None,
                        help="Max volumes per dataset (default: None = use all samples)")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Random seed for dataset sampling (default: 42)")

    args = parser.parse_args()

    # Validate
    if args.ckpt_folder is None and args.ckpt_path is None:
        parser.error("Either --ckpt-folder or --ckpt-path must be provided")

    return args


def main():
    args = parse_args()

    t0 = time.time()

    # Get checkpoints
    ckpt_paths = get_checkpoint_paths(args.ckpt_folder, args.ckpt_path)

    if not ckpt_paths:
        print("No checkpoints found!")
        return

    print(f"Found {len(ckpt_paths)} checkpoints to validate")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Resume support: skip checkpoints already present in CSVs
    existing_ckpts = load_existing_checkpoints(args.output_path)
    if existing_ckpts:
        before = len(ckpt_paths)
        ckpt_paths = [p for p in ckpt_paths if os.path.basename(p) not in existing_ckpts]
        skipped = before - len(ckpt_paths)
        if skipped > 0:
            print(f"Skipping {skipped} checkpoints already in CSVs.")
        if not ckpt_paths:
            print("All checkpoints already validated. Nothing to do.")
            return

    # Results counters
    new_results_3d = 0
    new_summaries = 0

    devices = args.devices if args.devices else [args.device]
    max_workers = min(max(1, args.num_workers), len(ckpt_paths), len(devices))

    if max_workers == 1:
        # Validate each checkpoint sequentially
        for ckpt_path in tqdm(ckpt_paths, desc="Checkpoints"):
            ckpt_name = os.path.basename(ckpt_path)
            print(f"\nValidating: {ckpt_name}")

            try:
                results_3d = validate_3d_dataset_from_embeddings(
                    sam2_config=args.sam2_config,
                    ckpt_path=ckpt_path,
                    embedding_dir=args.embedding_dir,
                    prompt_mode=args.prompt_mode,
                    device=devices[0],
                    max_samples_per_dataset=args.max_samples_per_dataset,
                    sample_seed=args.sample_seed,
                )

                # Add checkpoint name to results
                for r in results_3d:
                    r["checkpoint"] = ckpt_name

                # Compute summary
                summary = compute_summary(results_3d, ckpt_name)

                # Append results
                if results_3d:
                    csv_path = os.path.join(args.output_path, "validation_results_3d.csv")
                    append_results_to_csv(results_3d, csv_path)
                    new_results_3d += len(results_3d)

                if summary:
                    csv_path = os.path.join(args.output_path, "validation_summary.csv")
                    append_results_to_csv(summary, csv_path)
                    new_summaries += len(summary)

                # Print summary for this checkpoint
                if summary:
                    print(f"  Results for {ckpt_name}:")
                    for s in summary:
                        print(f"    {s['dataset']}: "
                              f"mean_dice={s['mean_dice']:.4f} +/- {s['std_dice']:.4f} "
                              f"(n={s['num_samples']})")

            except Exception as e:
                print(f"  Error validating {ckpt_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"\nValidating checkpoints in parallel with {max_workers} workers...")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = []
            for idx, ckpt_path in enumerate(ckpt_paths):
                device = devices[idx % len(devices)]
                ckpt_name = os.path.basename(ckpt_path)
                print(f"Queued: {ckpt_name} on {device}")
                futures.append(executor.submit(
                    validate_checkpoint_worker,
                    args.sam2_config,
                    ckpt_path,
                    args.embedding_dir,
                    args.prompt_mode,
                    device,
                    args.max_samples_per_dataset,
                    args.sample_seed,
                ))

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Checkpoints"):
                try:
                    ckpt_path, results_3d, summary = fut.result()
                    ckpt_name = os.path.basename(ckpt_path)

                    if results_3d:
                        csv_path = os.path.join(args.output_path, "validation_results_3d.csv")
                        append_results_to_csv(results_3d, csv_path)
                        new_results_3d += len(results_3d)

                    if summary:
                        csv_path = os.path.join(args.output_path, "validation_summary.csv")
                        append_results_to_csv(summary, csv_path)
                        new_summaries += len(summary)

                    if summary:
                        print(f"  Results for {ckpt_name}:")
                        for s in summary:
                            print(f"    {s['dataset']}: "
                                  f"mean_dice={s['mean_dice']:.4f} +/- {s['std_dice']:.4f} "
                                  f"(n={s['num_samples']})")
                except Exception as e:
                    print(f"  Error validating checkpoint: {e}")

    # Summary
    if new_results_3d or new_summaries:
        print("\nAppended results:")
        if new_results_3d:
            csv_path = os.path.join(args.output_path, "validation_results_3d.csv")
            print(f"  3D results: {csv_path} (+{new_results_3d} samples)")
        if new_summaries:
            csv_path = os.path.join(args.output_path, "validation_summary.csv")
            print(f"  Summary: {csv_path} (+{new_summaries} entries)")

    print(f"\nTotal elapsed time: {time.time() - t0:.2f}s")
    print("Done!")


if __name__ == "__main__":
    main()
