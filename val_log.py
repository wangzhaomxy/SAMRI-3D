# -*- coding: utf-8 -*-
"""
Validation Logger: Validate multiple checkpoints on 2D and/or 3D validation datasets.

This script validates trained SAM2 checkpoints on validation datasets and logs
results to CSV files. It supports:
- 2D slice-wise validation
- 3D volume validation with multi-object inference
- Parallel processing for efficiency
- Multiple checkpoint evaluation

Validation datasets are expected in "validation" subfolders within each dataset folder.

=== CLI Examples ===

1. Validate all checkpoints on 2D validation data:
python val_log.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-folder /scratch/project/samri/Model_save/samri2_stage1/ \
--data-paths /scratch/project/samri/Datasets/SAMRI_train_test/ \
--mode 2d \
--output-path /scratch/project/samri/validation_results/ \
--prompt-mode box \
--device cuda \
--io-workers 8 \
--max-samples-per-dataset 1000 \
--sample-seed 42

2. Validate all checkpoints on 3D validation data:
   python val_log.py \
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
       --ckpt-folder /scratch/project/samri/Model_save/samri2_stage1/ \
       --data-paths /scratch/project/samri/Datasets/SAMRI_datasets_3D/ \
       --mode 3d \
       --output-path /scratch/project/samri/validation_results/ \
       --prompt-mode box \
       --device cuda:0 \
       --io-workers 8 \
       --max-samples-per-dataset 1000 \
       --sample-seed 42

3. Validate on both 2D and 3D:
   python val_log.py \
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
       --ckpt-folder /scratch/project/samri/Model_save/samri2_stage1/ \
       --data-paths /scratch/project/samri/Datasets/SAMRI_train_test/ \
       --mode both \
       --output-path /scratch/project/samri/validation_results/ \
       --prompt-mode box \
       --device cuda:0 \
       --io-workers 8 \
       --max-samples-per-dataset 1000 \
       --sample-seed 42

4. Validate a single checkpoint:
   python val_log.py \
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
       --ckpt-path /scratch/project/samri/Model_save/samri2_stage1/sam2_maskdecoder_epoch_50.pth \
       --data-paths /scratch/project/samri/Datasets/SAMRI_train_test/ \
       --mode both \
       --output-path /scratch/project/samri/validation_results/ \
       --device cuda:0 \
       --io-workers 8 \
       --max-samples-per-dataset 1000 \
       --sample-seed 42

Key Arguments:
  --ckpt-folder         Folder containing multiple checkpoint .pth files
  --ckpt-path           Single checkpoint path (alternative to --ckpt-folder)
  --data-paths          Dataset folders containing "validation" subfolders
  --mode                Validation mode: "2d", "3d", or "both" (default: both)
  --prompt-mode         Prompt type: "box" or "mask" (default: box)
  --output-path         Output directory for CSV results
  --num-workers         Number of parallel workers (default: 8)

Output:
  - validation_results_2d.csv: Per-slice dice scores for 2D mode
  - validation_results_3d.csv: Per-volume dice scores for 3D mode
  - validation_summary.csv: Mean dice per checkpoint and dataset
"""

from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import contextlib
import tempfile
import shutil
import inspect
import multiprocessing as mp
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# Prefer local ./sam2
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_to_uint8(arr: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Normalize array to uint8 range [0, 255]."""
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo + 1e-6:
        lo, hi = float(arr.min()), float(arr.max() + 1e-6)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


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


def find_validation_folders(data_paths: List[str]) -> Dict[str, str]:
    """
    Find validation folders in dataset paths.
    Returns: {dataset_name: validation_folder_path}
    """
    val_folders = {}
    for base_path in data_paths:
        base_path = Path(base_path)
        if not base_path.exists():
            continue

        # Check if base_path itself has a validation subfolder
        val_path = base_path / "validation"
        if val_path.is_dir():
            ds_name = base_path.name
            val_folders[ds_name] = str(val_path)
        else:
            # Search for validation folders in subdirectories
            for subdir in base_path.iterdir():
                if subdir.is_dir():
                    val_path = subdir / "validation"
                    if val_path.is_dir():
                        ds_name = subdir.name
                        val_folders[ds_name] = str(val_path)

    return val_folders


def find_image_mask_pairs(folder: str) -> List[Tuple[str, str]]:
    """Find image-mask pairs in a folder."""
    supported_exts = [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]

    all_files = []
    for ext in supported_exts:
        all_files.extend(glob(os.path.join(folder, f"*{ext}")))

    img_files = [f for f in all_files if "_img_" in os.path.basename(f) or "_img." in os.path.basename(f)]

    pairs = []
    for img_path in img_files:
        basename = os.path.basename(img_path)
        # Try to find matching mask
        for pattern in ["_seg_", "_gt_", "_mask_"]:
            if "_img_" in basename:
                mask_name = basename.replace("_img_", pattern)
            else:
                mask_name = basename.replace("_img.", pattern.rstrip("_") + ".")

            mask_path = os.path.join(folder, mask_name)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
                break
        else:
            # Try same name with seg/gt prefix
            for ext in supported_exts:
                for pattern in ["seg_", "gt_", "mask_"]:
                    mask_path = os.path.join(folder, pattern + basename)
                    if os.path.exists(mask_path):
                        pairs.append((img_path, mask_path))
                        break

    return pairs


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
# SAM2 Output Suppression
# =============================================================================

@contextlib.contextmanager
def suppress_sam2_output(enabled: bool = True):
    """Suppress stdout/stderr (SAM2 prints + tqdm bars) inside the context."""
    if not enabled:
        yield
        return

    old_env = os.environ.get("TQDM_DISABLE", None)
    os.environ["TQDM_DISABLE"] = "1"

    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()
        if old_env is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = old_env


# =============================================================================
# 2D Validation
# =============================================================================

def validate_2d_sample(
    predictor: SAM2ImagePredictor,
    img_2d: np.ndarray,
    mask_2d: np.ndarray,
    prompt_mode: str,
) -> float:
    """
    Validate a single 2D slice.

    Args:
        predictor: SAM2 image predictor
        img_2d: 2D image (H, W) or (H, W, 3)
        mask_2d: 2D binary mask (H, W)
        prompt_mode: "box" or "mask"

    Returns:
        Dice score
    """
    img_rgb = prepare_image_rgb(img_2d)
    predictor.set_image(img_rgb)
    return predict_dice_for_label_on_set_image(predictor, mask_2d, prompt_mode)


def prepare_image_rgb(img_2d: np.ndarray) -> np.ndarray:
    """Prepare 2D image into uint8 RGB."""
    if img_2d.ndim == 2:
        img_rgb = np.stack([img_2d, img_2d, img_2d], axis=-1)
    else:
        img_rgb = img_2d
    return normalize_to_uint8(img_rgb)


def predict_dice_for_label_on_set_image(
    predictor: SAM2ImagePredictor,
    mask_2d: np.ndarray,
    prompt_mode: str,
) -> float:
    """Predict dice for a single label assuming predictor.set_image was already called."""
    if prompt_mode == "box":
        box = mask2d_to_bbox_xyxy(mask_2d)
        if box is None:
            return 0.0
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )
    else:  # mask mode - sample points
        ys, xs = np.where(mask_2d > 0)
        if xs.size == 0:
            return 0.0
        k = min(16, xs.size)
        idx = np.random.choice(xs.size, size=k, replace=False)
        pts = np.stack([xs[idx], ys[idx]], axis=1)
        labels = np.ones(k, dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=False,
        )

    pred_mask = masks[0] > 0
    return dice_score(pred_mask, mask_2d)


def validate_2d_volume(
    predictor: SAM2ImagePredictor,
    img_path: str,
    mask_path: str,
    prompt_mode: str,
) -> List[Dict]:
    """
    Validate all slices in a 3D volume for 2D evaluation.

    Returns:
        List of {slice_idx, label, dice} for each valid slice
    """
    if not HAS_SITK:
        raise RuntimeError("SimpleITK required for 2D validation")

    img_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)

    img_arr = sitk.GetArrayFromImage(img_sitk)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)

    return validate_2d_volume_from_arrays(predictor, img_arr, mask_arr, prompt_mode)


def validate_2d_volume_from_arrays(
    predictor: SAM2ImagePredictor,
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    prompt_mode: str,
) -> List[Dict]:
    """Validate all slices in a 3D volume for 2D evaluation (arrays)."""

    results = []
    if mask_arr.ndim == 3:
        slice_indices = np.where(mask_arr.sum(axis=(1, 2)) > 0)[0]
    else:
        slice_indices = [0] if mask_arr.sum() > 0 else []

    for slice_idx in slice_indices:
        if img_arr.ndim == 3:
            img_2d = img_arr[slice_idx]
            mask_2d = mask_arr[slice_idx]
        else:
            img_2d = img_arr
            mask_2d = mask_arr

        labels = np.unique(mask_2d)
        labels = labels[labels > 0]
        if labels.size == 0:
            continue

        img_rgb = prepare_image_rgb(img_2d)
        predictor.set_image(img_rgb)

        for label in labels:
            label_mask = (mask_2d == label).astype(np.uint8)
            if label_mask.sum() < 10:  # Skip very small masks
                continue

            dice = predict_dice_for_label_on_set_image(predictor, label_mask, prompt_mode)
            results.append({
                "slice_idx": int(slice_idx),
                "label": int(label),
                "dice": float(dice),
            })

    return results


def load_volume_pair(img_path: str, mask_path: str) -> Tuple[str, str, np.ndarray, np.ndarray]:
    """Load image/mask arrays for a pair path."""
    if not HAS_SITK:
        raise RuntimeError("SimpleITK required for validation")
    img_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    return img_path, mask_path, img_arr, mask_arr


def maybe_sample_pairs(
    pairs: List[Tuple[str, str]],
    max_samples: Optional[int],
    rng: np.random.RandomState,
) -> List[Tuple[str, str]]:
    """Limit number of pairs with deterministic random sampling."""
    if not max_samples or len(pairs) <= max_samples:
        return pairs
    idx = rng.choice(len(pairs), size=max_samples, replace=False)
    return [pairs[i] for i in idx]


def validate_2d_dataset(
    sam2_config: str,
    ckpt_path: str,
    val_folders: Dict[str, str],
    prompt_mode: str,
    device: str = "cuda:0",
    io_workers: int = 0,
    max_samples_per_dataset: Optional[int] = None,
    sample_seed: int = 42,
) -> List[Dict]:
    """
    Validate a checkpoint on 2D validation datasets.

    Returns:
        List of results with dataset, img_path, slice_idx, label, dice
    """
    # Build model
    model = build_sam2(sam2_config, ckpt_path, device=device)
    predictor = SAM2ImagePredictor(model)

    all_results = []

    for ds_name, val_folder in val_folders.items():
        pairs = find_image_mask_pairs(val_folder)
        rng = np.random.RandomState(sample_seed)
        pairs = maybe_sample_pairs(pairs, max_samples_per_dataset, rng)

        if io_workers and io_workers > 0:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=io_workers, mp_context=ctx) as executor:
                futures = [executor.submit(load_volume_pair, p[0], p[1]) for p in pairs]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"2D {ds_name}", leave=False):
                    try:
                        img_path, mask_path, img_arr, mask_arr = fut.result()
                        slice_results = validate_2d_volume_from_arrays(
                            predictor, img_arr, mask_arr, prompt_mode
                        )
                        for r in slice_results:
                            r["dataset"] = ds_name
                            r["img_path"] = img_path
                            r["mask_path"] = mask_path
                        all_results.extend(slice_results)
                    except Exception as e:
                        print(f"  Warning: Failed to process {ds_name}: {e}")
        else:
            for img_path, mask_path in tqdm(pairs, desc=f"2D {ds_name}", leave=False):
                try:
                    slice_results = validate_2d_volume(predictor, img_path, mask_path, prompt_mode)
                    for r in slice_results:
                        r["dataset"] = ds_name
                        r["img_path"] = img_path
                        r["mask_path"] = mask_path
                    all_results.extend(slice_results)
                except Exception as e:
                    print(f"  Warning: Failed to process {img_path}: {e}")

    return all_results


# =============================================================================
# 3D Validation (Multi-object)
# =============================================================================

def get_fast_tmp_root() -> Optional[str]:
    """Get fast tmp directory (/dev/shm if available)."""
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return None


def write_video_frames_to_dir(vol_shw: np.ndarray, slice_indices: List[int], out_dir: str) -> str:
    """Write volume slices as JPEG frames to a directory."""
    os.makedirs(out_dir, exist_ok=True)
    for i, z in enumerate(slice_indices):
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"{i:05d}.jpg"), quality=95)
    return out_dir


def make_pil_loader_from_volume(vol0: np.ndarray, slice_indices: List[int]):
    """Create a callable frame loader from volume slices."""
    def loader(frame_idx: int) -> Image.Image:
        z = slice_indices[frame_idx]
        s2d = normalize_to_uint8(vol0[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        return Image.fromarray(rgb)
    return loader, len(slice_indices)


def init_state_compat(predictor, video_source: Any, num_frames: int, prefer_streaming: bool = True):
    """Version-compatible init_state with fallback."""
    sig = inspect.signature(predictor.init_state)
    params = sig.parameters

    if prefer_streaming and callable(video_source):
        kwargs = {"video_path": video_source}
        if "frame_load_config" in params:
            kwargs["frame_load_config"] = {"max_frames": int(num_frames), "first_frame_num": 0}
        elif "max_frames" in params:
            kwargs["max_frames"] = int(num_frames)
        elif "num_frames" in params:
            kwargs["num_frames"] = int(num_frames)
        try:
            return predictor.init_state(**kwargs)
        except TypeError:
            pass

    if not isinstance(video_source, str):
        raise TypeError("Fallback init_state requires a directory path (str) as video_source.")
    return predictor.init_state(video_path=video_source)


class PreparedVideos:
    """Prepared per-case video sources for video1 and video2."""
    def __init__(
        self,
        v1_source: Any,
        v2_source: Any,
        v1_len: int,
        v2_len: int,
        v1_slices: List[int],
        v2_slices: List[int],
        tmp_base: Optional[str],
        use_streaming: bool,
    ):
        self.v1_source = v1_source
        self.v2_source = v2_source
        self.v1_len = v1_len
        self.v2_len = v2_len
        self.v1_slices = v1_slices
        self.v2_slices = v2_slices
        self.tmp_base = tmp_base
        self.use_streaming = use_streaming

    def cleanup(self):
        if self.tmp_base:
            shutil.rmtree(self.tmp_base, ignore_errors=True)


def prepare_videos_once(
    predictor,
    vol0: np.ndarray,
    v1_slices: List[int],
    v2_slices: List[int],
) -> PreparedVideos:
    """Prepare video sources ONCE per case."""
    loader1, num1 = make_pil_loader_from_volume(vol0, v1_slices)
    loader2, num2 = make_pil_loader_from_volume(vol0, v2_slices)

    try:
        with suppress_sam2_output(True):
            _st = init_state_compat(predictor, loader1, num1, prefer_streaming=True)
        return PreparedVideos(
            v1_source=loader1,
            v2_source=loader2,
            v1_len=num1,
            v2_len=num2,
            v1_slices=v1_slices,
            v2_slices=v2_slices,
            tmp_base=None,
            use_streaming=True,
        )
    except Exception:
        fast_tmp = get_fast_tmp_root()
        tmp_base = tempfile.mkdtemp(prefix="sam2_val_", dir=fast_tmp)
        v1_dir = os.path.join(tmp_base, "video1")
        v2_dir = os.path.join(tmp_base, "video2")
        write_video_frames_to_dir(vol0, v1_slices, v1_dir)
        write_video_frames_to_dir(vol0, v2_slices, v2_dir)
        return PreparedVideos(
            v1_source=v1_dir,
            v2_source=v2_dir,
            v1_len=num1,
            v2_len=num2,
            v1_slices=v1_slices,
            v2_slices=v2_slices,
            tmp_base=tmp_base,
            use_streaming=False,
        )


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


def add_prompts_for_all_labels(
    predictor,
    inference_state,
    prompt_mode: str,
    gt_cut_slice: np.ndarray,
    labels: List[int],
    ann_frame_idx: int = 0,
) -> List[int]:
    """Add prompts for all labels on the cut slice."""
    added: List[int] = []
    for lb in labels:
        obj_id = int(lb)
        mask_lb = (gt_cut_slice == obj_id).astype(np.uint8)
        if mask_lb.sum() == 0:
            continue

        if prompt_mode == "box":
            box = mask2d_to_bbox_xyxy(mask_lb)
            if box is None:
                continue
            try:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    box=box,
                )
            except TypeError:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    points=None,
                    labels=None,
                    box=box,
                )
            added.append(obj_id)

        elif prompt_mode == "mask":
            if hasattr(predictor, "add_new_mask"):
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    mask=mask_lb,
                )
                added.append(obj_id)
            else:
                pts, lbls = sample_positive_points(mask_lb, num_pos=256)
                if pts.shape[0] == 0:
                    continue
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbls,
                )
                added.append(obj_id)

    return added


@torch.no_grad()
def propagate_video_multiobj(
    predictor,
    inference_state,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Multi-object propagation."""
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        fi = int(out_frame_idx)
        out[fi] = {}
        for i, oid in enumerate(out_obj_ids):
            oid_i = int(oid)
            logits = out_mask_logits[i]
            if hasattr(logits, "detach"):
                logits = logits.detach()
            m = (logits > 0.0)
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            out[fi][oid_i] = np.squeeze(m).astype(np.uint8)
    return out


def merge_two_videos_to_semantic_volume(
    N: int,
    H: int,
    W: int,
    v1_slices: List[int],
    v2_slices: List[int],
    v1_pred: Dict[int, Dict[int, np.ndarray]],
    v2_pred: Dict[int, Dict[int, np.ndarray]],
    labels: List[int],
) -> np.ndarray:
    """Merge predictions back to a semantic label volume."""
    pred = np.zeros((N, H, W), dtype=np.int32)
    written = set()

    def write_slice(z: int, obj_to_mask: Dict[int, np.ndarray]):
        for lb in labels:
            oid = int(lb)
            if oid not in obj_to_mask:
                continue
            pred[z][obj_to_mask[oid] > 0] = oid

    for fidx, objm in v1_pred.items():
        z = v1_slices[fidx]
        write_slice(z, objm)
        written.add(z)

    for fidx, objm in v2_pred.items():
        z = v2_slices[fidx]
        if z in written:
            continue
        write_slice(z, objm)
        written.add(z)

    return pred


def find_cut_slice_and_labels(mask_arr: np.ndarray) -> Tuple[int, List[int], int]:
    """
    Find the optimal cut slice (slice with most foreground) and labels.

    Returns:
        (cut_slice_idx, labels, primary_axis)
    """
    if mask_arr.ndim == 2:
        labels = [int(lb) for lb in np.unique(mask_arr) if lb > 0]
        return 0, labels, 0

    # Find primary axis (axis with most slices containing foreground)
    fg_mask = mask_arr > 0
    axis_scores = []
    for ax in range(3):
        other_axes = tuple(i for i in range(3) if i != ax)
        slice_fg = fg_mask.sum(axis=other_axes)
        axis_scores.append(slice_fg.sum())

    primary_axis = int(np.argmax(axis_scores))

    # Move primary axis to front
    mask_0 = np.moveaxis(mask_arr, primary_axis, 0)

    # Find cut slice (slice with most foreground)
    fg_0 = (mask_0 > 0).sum(axis=(1, 2))
    cut_idx = int(np.argmax(fg_0))

    # Get labels present in cut slice
    labels = [int(lb) for lb in np.unique(mask_0[cut_idx]) if lb > 0]

    return cut_idx, labels, primary_axis


def validate_3d_volume(
    predictor: SAM2VideoPredictor,
    img_path: str,
    mask_path: str,
    prompt_mode: str,
) -> List[Dict]:
    """
    Validate a 3D volume using multi-object inference.

    Returns:
        List of {label, dice, voxel_count} for each label
    """
    if not HAS_SITK:
        raise RuntimeError("SimpleITK required for 3D validation")

    img_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)

    img_arr = sitk.GetArrayFromImage(img_sitk)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)

    return validate_3d_volume_from_arrays(predictor, img_arr, mask_arr, prompt_mode)


def validate_3d_volume_from_arrays(
    predictor: SAM2VideoPredictor,
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    prompt_mode: str,
) -> List[Dict]:
    """Validate a 3D volume using multi-object inference (arrays)."""

    # Find cut slice and labels
    cut_idx, labels, primary_axis = find_cut_slice_and_labels(mask_arr)

    if not labels:
        return []

    # Move primary axis to front for both
    vol0 = np.moveaxis(img_arr, primary_axis, 0)
    gt0 = np.moveaxis(mask_arr, primary_axis, 0)

    N, H, W = vol0.shape
    gt_cut = gt0[cut_idx]

    # Build video indices
    v1_slices = list(range(cut_idx, -1, -1))
    v2_slices = [cut_idx] + list(range(cut_idx + 1, N))

    # Prepare videos
    try:
        prepared = prepare_videos_once(predictor, vol0, v1_slices, v2_slices)
    except Exception as e:
        print(f"  Warning: Failed to prepare videos: {e}")
        return []

    results = []

    try:
        with suppress_sam2_output(True):
            # Video 1
            st1 = init_state_compat(predictor, prepared.v1_source, prepared.v1_len,
                                    prefer_streaming=prepared.use_streaming)
            added = add_prompts_for_all_labels(predictor, st1, prompt_mode, gt_cut, labels)
            if not added:
                prepared.cleanup()
                return []
            pred1 = propagate_video_multiobj(predictor, st1)

            # Video 2
            st2 = init_state_compat(predictor, prepared.v2_source, prepared.v2_len,
                                    prefer_streaming=prepared.use_streaming)
            _ = add_prompts_for_all_labels(predictor, st2, prompt_mode, gt_cut, added)
            pred2 = propagate_video_multiobj(predictor, st2)

        # Merge predictions
        pred0 = merge_two_videos_to_semantic_volume(
            N=N, H=H, W=W,
            v1_slices=v1_slices,
            v2_slices=v2_slices,
            v1_pred=pred1,
            v2_pred=pred2,
            labels=added,
        )

        # Restore original axis order
        pred_zyx = np.moveaxis(pred0, 0, primary_axis)

        # Compute dice for each label
        for lb in added:
            gt_lb = (mask_arr == lb).astype(np.uint8)
            pred_lb = (pred_zyx == lb).astype(np.uint8)
            dsc = dice_score(pred_lb, gt_lb)

            results.append({
                "label": int(lb),
                "dice": float(dsc),
                "voxel_count": int(gt_lb.sum()),
            })

    except Exception as e:
        print(f"  Warning: Failed to process: {e}")

    finally:
        prepared.cleanup()

    return results


def validate_3d_dataset(
    sam2_config: str,
    ckpt_path: str,
    val_folders: Dict[str, str],
    prompt_mode: str,
    device: str = "cuda:0",
    io_workers: int = 0,
    max_samples_per_dataset: Optional[int] = None,
    sample_seed: int = 42,
) -> List[Dict]:
    """
    Validate a checkpoint on 3D validation datasets.

    Returns:
        List of results with dataset, img_path, label, dice, voxel_count
    """
    # Build model
    model = build_sam2(sam2_config, ckpt_path, device=device, mode="eval")
    predictor = SAM2VideoPredictor(model)

    all_results = []

    for ds_name, val_folder in val_folders.items():
        pairs = find_image_mask_pairs(val_folder)
        rng = np.random.RandomState(sample_seed)
        pairs = maybe_sample_pairs(pairs, max_samples_per_dataset, rng)

        if io_workers and io_workers > 0:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=io_workers, mp_context=ctx) as executor:
                futures = [executor.submit(load_volume_pair, p[0], p[1]) for p in pairs]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"3D {ds_name}", leave=False):
                    try:
                        img_path, mask_path, img_arr, mask_arr = fut.result()
                        vol_results = validate_3d_volume_from_arrays(
                            predictor, img_arr, mask_arr, prompt_mode
                        )
                        for r in vol_results:
                            r["dataset"] = ds_name
                            r["img_path"] = img_path
                            r["mask_path"] = mask_path
                        all_results.extend(vol_results)
                    except Exception as e:
                        print(f"  Warning: Failed to process {ds_name}: {e}")
        else:
            for img_path, mask_path in tqdm(pairs, desc=f"3D {ds_name}", leave=False):
                try:
                    vol_results = validate_3d_volume(predictor, img_path, mask_path, prompt_mode)
                    for r in vol_results:
                        r["dataset"] = ds_name
                        r["img_path"] = img_path
                        r["mask_path"] = mask_path
                    all_results.extend(vol_results)
                except Exception as e:
                    print(f"  Warning: Failed to process {img_path}: {e}")

    return all_results


# =============================================================================
# Main Validation Functions
# =============================================================================

def validate_checkpoint(
    sam2_config: str,
    ckpt_path: str,
    val_folders: Dict[str, str],
    mode: str,
    prompt_mode: str,
    device: str = "cuda:0",
    io_workers: int = 0,
    max_samples_per_dataset: Optional[int] = None,
    sample_seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Validate a single checkpoint.

    Returns:
        (results_2d, results_3d)
    """
    results_2d = []
    results_3d = []

    if mode in ["2d", "both"]:
        results_2d = validate_2d_dataset(
            sam2_config,
            ckpt_path,
            val_folders,
            prompt_mode,
            device,
            io_workers=io_workers,
            max_samples_per_dataset=max_samples_per_dataset,
            sample_seed=sample_seed,
        )

    if mode in ["3d", "both"]:
        results_3d = validate_3d_dataset(
            sam2_config,
            ckpt_path,
            val_folders,
            prompt_mode,
            device,
            io_workers=io_workers,
            max_samples_per_dataset=max_samples_per_dataset,
            sample_seed=sample_seed,
        )

    return results_2d, results_3d


def compute_summary(results_2d: List[Dict], results_3d: List[Dict], ckpt_name: str) -> List[Dict]:
    """Compute summary statistics per dataset."""
    summary = []

    # 2D summary
    if results_2d:
        by_dataset = defaultdict(list)
        for r in results_2d:
            by_dataset[r["dataset"]].append(r["dice"])

        for ds_name, dices in by_dataset.items():
            summary.append({
                "checkpoint": ckpt_name,
                "dataset": ds_name,
                "mode": "2d",
                "mean_dice": float(np.mean(dices)),
                "std_dice": float(np.std(dices)),
                "num_samples": len(dices),
            })

    # 3D summary
    if results_3d:
        by_dataset = defaultdict(list)
        for r in results_3d:
            by_dataset[r["dataset"]].append(r["dice"])

        for ds_name, dices in by_dataset.items():
            summary.append({
                "checkpoint": ckpt_name,
                "dataset": ds_name,
                "mode": "3d",
                "mean_dice": float(np.mean(dices)),
                "std_dice": float(np.std(dices)),
                "num_samples": len(dices),
            })

    return summary


def validate_checkpoint_worker(
    sam2_config: str,
    ckpt_path: str,
    val_folders: Dict[str, str],
    mode: str,
    prompt_mode: str,
    device: str,
    io_workers: int,
    max_samples_per_dataset: Optional[int],
    sample_seed: int,
) -> Tuple[str, List[Dict], List[Dict], List[Dict]]:
    """Worker wrapper for checkpoint validation (for multiprocessing)."""
    ckpt_name = os.path.basename(ckpt_path)
    results_2d, results_3d = validate_checkpoint(
        sam2_config=sam2_config,
        ckpt_path=ckpt_path,
        val_folders=val_folders,
        mode=mode,
        prompt_mode=prompt_mode,
        device=device,
        io_workers=io_workers,
        max_samples_per_dataset=max_samples_per_dataset,
        sample_seed=sample_seed,
    )

    for r in results_2d:
        r["checkpoint"] = ckpt_name
    for r in results_3d:
        r["checkpoint"] = ckpt_name

    summary = compute_summary(results_2d, results_3d, ckpt_name)
    return ckpt_path, results_2d, results_3d, summary


def save_results_to_csv(results: List[Dict], csv_path: str):
    """Save results to CSV file."""
    if not results:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Get all keys
    keys = list(results[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate multiple checkpoints on 2D and/or 3D validation datasets"
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
    parser.add_argument("--data-paths", type=str, nargs="+", required=True,
                        help="Dataset folders containing 'validation' subfolders")

    # Validation configuration
    parser.add_argument("--mode", type=str, default="both",
                        choices=["2d", "3d", "both"],
                        help="Validation mode (default: both)")
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
                        help="Device list for parallel validation (e.g., cuda:0 cuda:1). "
                             "Overrides --device when provided. Use repeats to oversubscribe.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel workers for checkpoint-level validation")
    parser.add_argument("--io-workers", type=int, default=4,
                        help="Number of CPU workers for IO/preprocessing (default: 4)")
    parser.add_argument("--max-samples-per-dataset", type=int, default=1000,
                        help="Max image-mask pairs per dataset (default: 1000)")
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

    # Find validation folders
    print("Searching for validation folders...")
    val_folders = find_validation_folders(args.data_paths)

    if not val_folders:
        print("No validation folders found!")
        return

    print(f"Found {len(val_folders)} validation datasets:")
    for ds_name, path in val_folders.items():
        pairs = find_image_mask_pairs(path)
        print(f"  {ds_name}: {len(pairs)} image-mask pairs")

    # Get checkpoints
    ckpt_paths = get_checkpoint_paths(args.ckpt_folder, args.ckpt_path)

    if not ckpt_paths:
        print("No checkpoints found!")
        return

    print(f"\nFound {len(ckpt_paths)} checkpoints to validate")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Results accumulators
    all_results_2d = []
    all_results_3d = []
    all_summaries = []

    devices = args.devices if args.devices else [args.device]
    max_workers = min(max(1, args.num_workers), len(ckpt_paths), len(devices))

    if max_workers == 1:
        # Validate each checkpoint sequentially
        for ckpt_path in tqdm(ckpt_paths, desc="Checkpoints"):
            ckpt_name = os.path.basename(ckpt_path)
            print(f"\nValidating: {ckpt_name}")

            try:
                results_2d, results_3d = validate_checkpoint(
                    sam2_config=args.sam2_config,
                    ckpt_path=ckpt_path,
                    val_folders=val_folders,
                    mode=args.mode,
                    prompt_mode=args.prompt_mode,
                    device=devices[0],
                    io_workers=args.io_workers,
                    max_samples_per_dataset=args.max_samples_per_dataset,
                    sample_seed=args.sample_seed,
                )

                # Add checkpoint name to results
                for r in results_2d:
                    r["checkpoint"] = ckpt_name
                for r in results_3d:
                    r["checkpoint"] = ckpt_name

                all_results_2d.extend(results_2d)
                all_results_3d.extend(results_3d)

                # Compute summary
                summary = compute_summary(results_2d, results_3d, ckpt_name)
                all_summaries.extend(summary)

                # Print summary for this checkpoint
                if summary:
                    print(f"  Results for {ckpt_name}:")
                    for s in summary:
                        print(f"    {s['dataset']} ({s['mode']}): "
                              f"mean_dice={s['mean_dice']:.4f} +/- {s['std_dice']:.4f} "
                              f"(n={s['num_samples']})")

            except Exception as e:
                print(f"  Error validating {ckpt_name}: {e}")
    else:
        print(f"\nValidating checkpoints in parallel with {max_workers} workers...")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = []
            for idx, ckpt_path in enumerate(ckpt_paths):
                device = devices[idx % len(devices)]
                ckpt_name = os.path.basename(ckpt_path)
                print(f"\nQueued: {ckpt_name} on {device}")
                futures.append(executor.submit(
                    validate_checkpoint_worker,
                    args.sam2_config,
                    ckpt_path,
                    val_folders,
                    args.mode,
                    args.prompt_mode,
                    device,
                    args.io_workers,
                    args.max_samples_per_dataset,
                    args.sample_seed,
                ))

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Checkpoints"):
                try:
                    ckpt_path, results_2d, results_3d, summary = fut.result()
                    ckpt_name = os.path.basename(ckpt_path)

                    all_results_2d.extend(results_2d)
                    all_results_3d.extend(results_3d)
                    all_summaries.extend(summary)

                    if summary:
                        print(f"  Results for {ckpt_name}:")
                        for s in summary:
                            print(f"    {s['dataset']} ({s['mode']}): "
                                  f"mean_dice={s['mean_dice']:.4f} +/- {s['std_dice']:.4f} "
                                  f"(n={s['num_samples']})")
                except Exception as e:
                    print(f"  Error validating checkpoint: {e}")

    # Save results
    print("\nSaving results...")

    if all_results_2d:
        csv_path = os.path.join(args.output_path, "validation_results_2d.csv")
        save_results_to_csv(all_results_2d, csv_path)
        print(f"  2D results: {csv_path} ({len(all_results_2d)} samples)")

    if all_results_3d:
        csv_path = os.path.join(args.output_path, "validation_results_3d.csv")
        save_results_to_csv(all_results_3d, csv_path)
        print(f"  3D results: {csv_path} ({len(all_results_3d)} samples)")

    if all_summaries:
        csv_path = os.path.join(args.output_path, "validation_summary.csv")
        save_results_to_csv(all_summaries, csv_path)
        print(f"  Summary: {csv_path} ({len(all_summaries)} entries)")

    print(f"\nTotal elapsed time: {time.time() - t0:.2f}s")
    print("Done!")


if __name__ == "__main__":
    main()
