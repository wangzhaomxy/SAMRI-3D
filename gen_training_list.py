# -*- coding: utf-8 -*-
"""
Generate training sample lists with dice scores for filtered training.

This script evaluates SAM2 model performance on training samples and saves the results
for later use in filtered training (train only on samples with dice > threshold).

Two modes:
1. 2D mode: Evaluate each 2D slice with SAM2 image predictor (box prompt)
2. 3D mode: Evaluate 3D volumes with SAM2 video predictor (box prompt)

Output format (pickle):
- training_list_2d.pkl: List of dicts, each containing:
    {img_path, mask_path, label, dataset, dice, pixel_count, area_percentage}
- training_list_3d.pkl: List of dicts, each containing:
    {img_path, mask_path, label, dataset, dice, voxel_count, voxel_percentage}

These files can be used with FilteredNiiDataset or FilteredSAM2EmbDataset (to be implemented)
for training on samples filtered by dice score threshold.

=== Usage ===

1. Generate 2D training list (for slice-wise training):
   python gen_training_list.py \\
       --mode 2d \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --data-paths /scratch/project/samri/Datasets/SAMRI_train_test/ \\
       --save-path /scratch/project/samri/training_lists/training_list_2d.pkl \\
       --num-workers 8

2. Generate 3D training list (for volume training):
   python gen_training_list.py \\
       --mode 3d \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --data-paths /scratch/project/samri/Datasets/SAMRI_datasets_3D/ \\
       --save-path /scratch/project/samri/training_lists/training_list_3d.pkl \\
       --prompt-mode box \\
       --num-workers 4

3. Generate both lists:
   python gen_training_list.py \\
       --mode both \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --data-paths /scratch/project/samri/Datasets/SAMRI_train_test/ \\
       --save-dir /scratch/project/samri/training_lists/

=== Training with Filtered Samples ===

After generating the training list, use with train_stage_one_slicewise.py:
   python train_stage_one_slicewise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --training-list /scratch/project/samri/training_lists/training_list_2d.pkl \\
       --dice-threshold 0.4 \\
       --save-path /scratch/project/samri/Model_save/samri2_filtered/

=== Options ===

  --mode              Evaluation mode: 2d, 3d, or both
  --data-paths        Root path containing dataset folders with /training/ subdirectories
  --save-path         Output pickle file path (for single mode)
  --save-dir          Output directory (for 'both' mode)
  --num-workers       Workers for parallel data loading (default: 8)
  --prompt-mode       Prompt type for 3D: box or mask (default: box)
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import argparse
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Prefer local ./sam2
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import utilities
from utils.dataloader import NiiDataset
from utils.utils import MaskSplit, gen_bboxes
from utils.losses import dice_similarity

# For 3D mode
try:
    import SimpleITK as sitk
    from PIL import Image
    HAS_3D_DEPS = True
except ImportError:
    HAS_3D_DEPS = False


# =============================================================================
# 2D Evaluation Functions (adapted from test_vis2d.py)
# =============================================================================

def _ensure_hwc_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is in HWC uint8 RGB format for SAM2."""
    if image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8:
        return image

    # CHW -> HWC
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    # grayscale -> 3ch
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # to uint8
    if image.dtype != np.uint8:
        vmin, vmax = float(image.min()), float(image.max())
        if 0.0 <= vmin and vmax <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    return image


def get_training_paths_2d(data_paths: str) -> List[str]:
    """Get training subdirectories from dataset root path."""
    print(f"Scanning for training paths in: {data_paths}")
    training_paths = []

    # Check if data_paths is a root with multiple datasets
    ds_folders = [d for d in sorted(glob(data_paths.rstrip("/") + "/*")) if os.path.isdir(d)]

    if ds_folders:
        for ds in ds_folders:
            train_dir = os.path.join(ds, "training")
            if os.path.isdir(train_dir):
                training_paths.append(train_dir + "/")
    else:
        # data_paths itself might be a training folder
        if os.path.isdir(data_paths):
            training_paths.append(data_paths.rstrip("/") + "/")

    print(f"Found {len(training_paths)} training directories")
    return training_paths


@torch.no_grad()
def evaluate_2d_training_samples(
    sam2_model,
    data_paths: str,
    num_workers: int = 8,
) -> List[Dict]:
    """
    Evaluate all 2D training samples and return list of results.

    Returns:
        List of dicts with: img_path, mask_path, label, dataset, dice, pixel_count, area_percentage
    """
    predictor = SAM2ImagePredictor(sam2_model)
    results: List[Dict] = []

    training_paths = get_training_paths_2d(data_paths)

    for file_path in training_paths:
        # Extract dataset name
        ds_name = Path(file_path).parent.name
        print(f"\nProcessing dataset: {ds_name}")

        # Create dataset and loader
        try:
            dataset = NiiDataset([file_path], multi_mask=True, with_name=True)
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
            )
        except Exception as e:
            print(f"  Warning: Failed to load dataset {file_path}: {e}")
            continue

        if len(dataset) == 0:
            print(f"  No samples found in {file_path}")
            continue

        print(f"  Found {len(dataset)} samples")

        for image, mask, img_fullpath, mask_fullpath in tqdm(loader, desc=f"  {ds_name}"):
            # Process batch (batch_size=1)
            image = image.squeeze(0).detach().cpu().numpy()
            mask = mask.squeeze(0).detach().cpu().numpy()

            if mask.ndim == 2:
                mask = mask[None, :, :]  # ensure (1,H,W)

            image = _ensure_hwc_uint8_rgb(image)
            predictor.set_image(image)

            H, W = mask.shape[-2], mask.shape[-1]
            total_pixels = H * W

            # Split mask into per-label masks
            ms = MaskSplit(mask)

            for i in range(len(ms)):
                each_mask, label = ms[i]  # each_mask: (H,W) boolean

                # Generate box prompt
                bbox = gen_bboxes(each_mask, jitter=0)  # [x0, y0, x1, y1]

                # Predict with box prompt
                pre_mask_b, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array(bbox, dtype=np.float32)[None, :],
                    multimask_output=False,
                )

                # Calculate dice
                gt = each_mask.astype(np.uint8)
                bm = pre_mask_b[0].astype(np.uint8)
                b_dice = dice_similarity(bm, gt)

                # Pixel stats
                pc = int(np.sum(each_mask))

                results.append({
                    "img_path": img_fullpath[0] if isinstance(img_fullpath, (list, tuple)) else str(img_fullpath),
                    "mask_path": mask_fullpath[0] if isinstance(mask_fullpath, (list, tuple)) else str(mask_fullpath),
                    "label": int(label),
                    "dataset": ds_name,
                    "dice": float(b_dice),
                    "pixel_count": pc,
                    "area_percentage": pc / float(total_pixels),
                })

    return results


# =============================================================================
# 3D Evaluation Functions (adapted from test_vis3d.py)
# =============================================================================

SUPPORTED_EXTS = {".nii", ".mhd", ".mha", ".nrrd", ".dcm", ".gz"}
PREFERRED_EXTS = [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]


def read_3d_image(path: str) -> Tuple[np.ndarray, "sitk.Image"]:
    """Read 3D volume using SimpleITK."""
    if not HAS_3D_DEPS:
        raise RuntimeError("SimpleITK is required for 3D mode. Install with: pip install SimpleITK")

    p = Path(path)
    if p.is_dir():
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(p))
        if not series_ids:
            raise RuntimeError(f"No DICOM series found in directory: {path}")
        files = reader.GetGDCMSeriesFileNames(str(p), series_ids[0])
        reader.SetFileNames(files)
        img = reader.Execute()
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim != 3:
            raise RuntimeError(f"Expected 3D DICOM series, got shape {arr.shape}")
        return arr, img

    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D image, got shape {arr.shape}")
    return arr, img


def _has_med_ext_or_dir(p: Path) -> bool:
    if p.is_dir():
        return True
    if p.name.lower().endswith(".nii.gz"):
        return True
    return p.suffix.lower() in SUPPORTED_EXTS


def is_img_candidate(p: Path) -> bool:
    return ("_img_" in p.name) and _has_med_ext_or_dir(p) and (p.is_file() or p.is_dir())


def try_find_gt_same_folder(img_path: Path) -> Optional[Path]:
    gt_name = img_path.name.replace("_img_", "_gt_")
    gt_path = img_path.with_name(gt_name)
    if gt_path.exists():
        return gt_path

    # Also try _seg_ pattern
    seg_name = img_path.name.replace("_img_", "_seg_")
    seg_path = img_path.with_name(seg_name)
    if seg_path.exists():
        return seg_path

    # Match by base
    if gt_name.lower().endswith(".nii.gz"):
        base = gt_name[:-7]
    else:
        base = Path(gt_name).stem

    for ext in PREFERRED_EXTS:
        cand = img_path.with_name(base + ext)
        if cand.exists():
            return cand

    return None


def scan_pairs_same_folder(folder: str) -> List[Tuple[str, str]]:
    rootp = Path(folder)
    pairs: List[Tuple[str, str]] = []
    for p in rootp.rglob("*"):
        if not is_img_candidate(p):
            continue
        gt = try_find_gt_same_folder(p)
        if gt is None or not gt.exists():
            continue
        pairs.append((str(p), str(gt)))
    pairs.sort()
    return pairs


def get_training_folders_3d(data_paths: str) -> List[str]:
    """Get training folders for 3D data."""
    data_paths = data_paths.rstrip("/")
    ds_folders = [d for d in sorted(glob(data_paths + "/*")) if os.path.isdir(d)]

    if not ds_folders:
        return [data_paths]

    training_folders: List[str] = []
    for ds in ds_folders:
        cand = os.path.join(ds, "training")
        training_folders.append(cand if os.path.isdir(cand) else ds)
    return training_folders


def normalize_to_uint8(slice2d: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    sl = slice2d.astype(np.float32)
    lo = np.percentile(sl, p_low)
    hi = np.percentile(sl, p_high)
    if hi <= lo + 1e-6:
        lo, hi = float(sl.min()), float(sl.max() + 1e-6)
    sl = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
    return (sl * 255.0).astype(np.uint8)


def make_pil_loader_from_volume(vol_shw: np.ndarray, slice_indices: List[int]):
    """Create a frame loader function for SAM2 video predictor."""
    def loader_func(frame_idx: int) -> "Image.Image":
        z = slice_indices[frame_idx]
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        return Image.fromarray(rgb)
    return loader_func, len(slice_indices)


def mask2d_to_bbox_xyxy(mask2d: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def dice_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = (pred & gt).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + gt.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def largest_mask_slice_index(mask: np.ndarray, axis: int, label: Optional[int] = None) -> int:
    moved = np.moveaxis(mask, axis, 0)
    fg = (moved == label) if (label is not None) else (moved > 0)
    sums = fg.sum(axis=tuple(range(1, moved.ndim)))
    return int(np.argmax(sums))


def build_two_videos_indices(N: int, n: int) -> Tuple[List[int], List[int]]:
    v1 = list(range(n, -1, -1))
    v2 = [n] + list(range(n + 1, N))
    return v1, v2


def load_3d_case(args_tuple):
    """
    Load and preprocess a 3D case for parallel loading.

    Args:
        args_tuple: Tuple of (img_path, gt_path, ds_name)

    Returns:
        dict with loaded data or None if failed
    """
    img_path, gt_path, ds_name = args_tuple
    try:
        img_zyx, img_sitk = read_3d_image(img_path)
        gt_zyx, _ = read_3d_image(gt_path)

        if img_zyx.shape != gt_zyx.shape:
            return None

        # Get labels
        labels = [int(x) for x in np.unique(gt_zyx) if x != 0]
        if not labels:
            return None

        # Choose cut axis (shortest)
        primary_axis = int(np.argmin(img_zyx.shape))
        cut_idx = largest_mask_slice_index(gt_zyx, primary_axis, label=None)

        # Move axis to front
        vol0 = np.moveaxis(img_zyx, primary_axis, 0)
        gt0 = np.moveaxis(gt_zyx, primary_axis, 0)
        N, H, W = vol0.shape
        n = int(cut_idx)
        v1_slices, v2_slices = build_two_videos_indices(N, n)
        gt_cut = gt0[n]

        return {
            'img_path': img_path,
            'gt_path': gt_path,
            'ds_name': ds_name,
            'img_zyx': img_zyx,
            'gt_zyx': gt_zyx,
            'vol0': vol0,
            'gt_cut': gt_cut,
            'v1_slices': v1_slices,
            'v2_slices': v2_slices,
            'labels': labels,
            'primary_axis': primary_axis,
        }
    except Exception as e:
        print(f"  Warning: Failed to load {img_path}: {e}")
        return None


def prefetch_3d_data_generator(cases, num_workers, prefetch_size):
    """
    Generator that prefetches 3D data using a thread pool.

    Args:
        cases: List of (img_path, gt_path, ds_name) tuples
        num_workers: Number of worker threads
        prefetch_size: Number of items to prefetch

    Yields:
        Preprocessed data dictionaries
    """
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit initial batch
        futures = {}
        for i, case in enumerate(cases[:prefetch_size]):
            future = executor.submit(load_3d_case, case)
            futures[future] = i

        next_idx = prefetch_size
        results = {}
        expected_idx = 0

        while futures or expected_idx < len(cases):
            # Wait for any future to complete
            if futures:
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                    done_futures = list(done)

                for future in done_futures:
                    idx = futures.pop(future)
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"Error loading case {idx}: {e}")
                        results[idx] = None

                    # Submit next item if available
                    if next_idx < len(cases):
                        new_future = executor.submit(load_3d_case, cases[next_idx])
                        futures[new_future] = next_idx
                        next_idx += 1

            # Yield results in order
            while expected_idx in results:
                result = results.pop(expected_idx)
                expected_idx += 1
                if result is not None:
                    yield result


@torch.no_grad()
def evaluate_3d_training_samples(
    predictor,
    data_paths: str,
    prompt_mode: str = "box",
    num_workers: int = 4,
) -> List[Dict]:
    """
    Evaluate all 3D training samples and return list of results.

    Uses parallel data loading for improved throughput while keeping
    GPU inference sequential.

    Args:
        predictor: SAM2 video predictor
        data_paths: Root path containing dataset folders
        prompt_mode: Prompt type ("box" or "mask")
        num_workers: Number of workers for parallel data loading

    Returns:
        List of dicts with: img_path, mask_path, label, dataset, dice, voxel_count, voxel_percentage
    """
    if not HAS_3D_DEPS:
        raise RuntimeError("SimpleITK and PIL are required for 3D mode")

    results: List[Dict] = []
    training_folders = get_training_folders_3d(data_paths)

    # Collect all cases with dataset names
    all_cases = []
    for train_folder in training_folders:
        tf = Path(train_folder).resolve()
        ds_name = tf.parent.name if tf.name == "training" else tf.name

        pairs = scan_pairs_same_folder(train_folder)
        if not pairs:
            print(f"No pairs found in {train_folder}")
            continue

        for img_path, gt_path in pairs:
            all_cases.append((img_path, gt_path, ds_name))

    if not all_cases:
        print("No cases found!")
        return results

    print(f"\nTotal cases to process: {len(all_cases)}")
    print(f"Parallel workers for data loading: {num_workers}")

    prefetch_size = min(num_workers * 2, len(all_cases))

    # Process with parallel data loading
    pbar = tqdm(total=len(all_cases), desc="Processing 3D cases")

    for data in prefetch_3d_data_generator(all_cases, num_workers, prefetch_size):
        img_path = data['img_path']
        gt_path = data['gt_path']
        ds_name = data['ds_name']
        gt_zyx = data['gt_zyx']
        vol0 = data['vol0']
        gt_cut = data['gt_cut']
        v1_slices = data['v1_slices']
        v2_slices = data['v2_slices']
        labels = data['labels']
        primary_axis = data['primary_axis']

        total_voxels = int(gt_zyx.size)

        # Process each label
        for lb in labels:
            # Skip if label not present on cut slice
            if np.sum(gt_cut == lb) == 0:
                continue

            try:
                # Predict using video predictor
                pred_zyx_bin = predict_one_label_3d(
                    predictor=predictor,
                    vol0=vol0,
                    gt_cut=gt_cut,
                    v1_slices=v1_slices,
                    v2_slices=v2_slices,
                    prompt_mode=prompt_mode,
                    label=lb,
                    primary_axis=primary_axis,
                )

                if pred_zyx_bin is None:
                    continue

                # Calculate dice
                gt_lb = (gt_zyx == lb).astype(np.uint8)
                dsc = dice_3d(pred_zyx_bin, gt_lb)

                vc = int(gt_lb.sum())

                results.append({
                    "img_path": img_path,
                    "mask_path": gt_path,
                    "label": int(lb),
                    "dataset": ds_name,
                    "dice": float(dsc),
                    "voxel_count": vc,
                    "voxel_percentage": vc / float(total_voxels),
                })

            except Exception as e:
                print(f"  Warning: Failed to process label {lb} for {img_path}: {e}")
                continue

        pbar.update(1)

    pbar.close()
    return results


def predict_one_label_3d(
    predictor,
    vol0: np.ndarray,
    gt_cut: np.ndarray,
    v1_slices: List[int],
    v2_slices: List[int],
    prompt_mode: str,
    label: int,
    primary_axis: int,
) -> Optional[np.ndarray]:
    """Run 3D prediction for one label."""
    import inspect

    lb = int(label)
    mask_lb = (gt_cut == lb).astype(np.uint8)

    if mask_lb.sum() == 0:
        return None

    # Get box prompt
    box = mask2d_to_bbox_xyxy(mask_lb)
    if box is None:
        return None

    N, H, W = vol0.shape

    # Create video loaders
    loader1, num1 = make_pil_loader_from_volume(vol0, v1_slices)
    loader2, num2 = make_pil_loader_from_volume(vol0, v2_slices)

    # Initialize and propagate video 1
    sig = inspect.signature(predictor.init_state)
    params = sig.parameters

    kwargs = {"video_path": loader1}
    if "frame_load_config" in params:
        kwargs["frame_load_config"] = {"max_frames": int(num1), "first_frame_num": 0}
    elif "max_frames" in params:
        kwargs["max_frames"] = int(num1)

    st1 = predictor.init_state(**kwargs)

    # Add prompt
    try:
        predictor.add_new_points_or_box(
            inference_state=st1,
            frame_idx=0,
            obj_id=lb,
            box=box,
        )
    except TypeError:
        predictor.add_new_points_or_box(
            inference_state=st1,
            frame_idx=0,
            obj_id=lb,
            points=None,
            labels=None,
            box=box,
        )

    # Propagate video 1
    pred1: Dict[int, np.ndarray] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(st1):
        fi = int(out_frame_idx)
        for j, o in enumerate(out_obj_ids):
            if int(o) == lb:
                logits = out_mask_logits[j]
                if hasattr(logits, "detach"):
                    logits = logits.detach()
                m = (logits > 0.0)
                if hasattr(m, "cpu"):
                    m = m.cpu().numpy()
                pred1[fi] = np.squeeze(m).astype(np.uint8)
                break

    # Initialize and propagate video 2
    kwargs2 = {"video_path": loader2}
    if "frame_load_config" in params:
        kwargs2["frame_load_config"] = {"max_frames": int(num2), "first_frame_num": 0}
    elif "max_frames" in params:
        kwargs2["max_frames"] = int(num2)

    st2 = predictor.init_state(**kwargs2)

    try:
        predictor.add_new_points_or_box(
            inference_state=st2,
            frame_idx=0,
            obj_id=lb,
            box=box,
        )
    except TypeError:
        predictor.add_new_points_or_box(
            inference_state=st2,
            frame_idx=0,
            obj_id=lb,
            points=None,
            labels=None,
            box=box,
        )

    # Propagate video 2
    pred2: Dict[int, np.ndarray] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(st2):
        fi = int(out_frame_idx)
        for j, o in enumerate(out_obj_ids):
            if int(o) == lb:
                logits = out_mask_logits[j]
                if hasattr(logits, "detach"):
                    logits = logits.detach()
                m = (logits > 0.0)
                if hasattr(m, "cpu"):
                    m = m.cpu().numpy()
                pred2[fi] = np.squeeze(m).astype(np.uint8)
                break

    # Merge predictions
    pred0 = np.zeros((N, H, W), dtype=np.uint8)
    written = set()

    for fidx, m in pred1.items():
        z = v1_slices[fidx]
        pred0[z] = (m > 0).astype(np.uint8)
        written.add(z)

    for fidx, m in pred2.items():
        z = v2_slices[fidx]
        if z not in written:
            pred0[z] = (m > 0).astype(np.uint8)
            written.add(z)

    # Restore to original axis order
    pred_zyx = np.moveaxis(pred0, 0, primary_axis)
    return pred_zyx


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training sample lists with dice scores for filtered training"
    )

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["2d", "3d", "both"], required=True,
                        help="Evaluation mode: 2d, 3d, or both")

    # Model configuration
    parser.add_argument("--sam2-config", type=str, required=True,
                        help="Path to SAM2 config file")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to SAM2 checkpoint")

    # Data paths
    parser.add_argument("--data-paths", type=str, required=True,
                        help="Root path containing dataset folders with /training/ subdirectories")

    # Output options
    parser.add_argument("--save-path", type=str, default=None,
                        help="Output pickle file path (for single mode)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output directory (for 'both' mode, creates training_list_2d.pkl and training_list_3d.pkl)")

    # Processing options
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers for parallel data loading (2D and 3D)")
    parser.add_argument("--prompt-mode", type=str, choices=["box", "mask"], default="box",
                        help="Prompt mode for 3D evaluation (default: box)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate output arguments
    if args.mode == "both":
        if args.save_dir is None:
            print("Error: --save-dir is required when mode is 'both'")
            return
    else:
        if args.save_path is None:
            print(f"Error: --save-path is required when mode is '{args.mode}'")
            return

    print("\n" + "=" * 60)
    print("Generate Training List with Dice Scores")
    print("=" * 60)
    print(f"Mode          : {args.mode}")
    print(f"SAM2 config   : {args.sam2_config}")
    print(f"Checkpoint    : {args.ckpt_path}")
    print(f"Data paths    : {args.data_paths}")
    print(f"Device        : {args.device}")
    print("=" * 60 + "\n")

    t0 = time.time()

    # Run 2D evaluation
    if args.mode in ["2d", "both"]:
        print("\n=== Running 2D Evaluation ===\n")

        # Build SAM2 image model
        sam2_model = build_sam2(args.sam2_config, args.ckpt_path, device=args.device)
        sam2_model.eval()

        results_2d = evaluate_2d_training_samples(
            sam2_model=sam2_model,
            data_paths=args.data_paths,
            num_workers=args.num_workers,
        )

        # Save 2D results
        if args.mode == "2d":
            save_path_2d = args.save_path
        else:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path_2d = os.path.join(args.save_dir, "training_list_2d.pkl")

        Path(os.path.dirname(save_path_2d) or ".").mkdir(parents=True, exist_ok=True)
        with open(save_path_2d, "wb") as f:
            pickle.dump(results_2d, f)

        print(f"\n2D Results saved to: {save_path_2d}")
        print(f"  Total samples: {len(results_2d)}")

        # Print summary statistics
        if results_2d:
            dices = [r["dice"] for r in results_2d]
            print(f"  Dice - mean: {np.mean(dices):.4f}, std: {np.std(dices):.4f}")
            print(f"  Dice - min: {np.min(dices):.4f}, max: {np.max(dices):.4f}")
            for thresh in [0.2, 0.4, 0.6, 0.8]:
                count = sum(1 for d in dices if d > thresh)
                print(f"  Samples with dice > {thresh}: {count} ({100*count/len(dices):.1f}%)")

        # Free memory
        del sam2_model
        torch.cuda.empty_cache()

    # Run 3D evaluation
    if args.mode in ["3d", "both"]:
        print("\n=== Running 3D Evaluation ===\n")

        if not HAS_3D_DEPS:
            print("Error: SimpleITK and PIL are required for 3D mode")
            print("Install with: pip install SimpleITK Pillow")
            return

        # Build SAM2 video predictor
        predictor = build_sam2_video_predictor(args.sam2_config, args.ckpt_path, device=args.device)

        results_3d = evaluate_3d_training_samples(
            predictor=predictor,
            data_paths=args.data_paths,
            prompt_mode=args.prompt_mode,
            num_workers=args.num_workers,
        )

        # Save 3D results
        if args.mode == "3d":
            save_path_3d = args.save_path
        else:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path_3d = os.path.join(args.save_dir, "training_list_3d.pkl")

        Path(os.path.dirname(save_path_3d) or ".").mkdir(parents=True, exist_ok=True)
        with open(save_path_3d, "wb") as f:
            pickle.dump(results_3d, f)

        print(f"\n3D Results saved to: {save_path_3d}")
        print(f"  Total samples: {len(results_3d)}")

        # Print summary statistics
        if results_3d:
            dices = [r["dice"] for r in results_3d]
            print(f"  Dice - mean: {np.mean(dices):.4f}, std: {np.std(dices):.4f}")
            print(f"  Dice - min: {np.min(dices):.4f}, max: {np.max(dices):.4f}")
            for thresh in [0.2, 0.4, 0.6, 0.8]:
                count = sum(1 for d in dices if d > thresh)
                print(f"  Samples with dice > {thresh}: {count} ({100*count/len(dices):.1f}%)")

    print(f"\nTotal elapsed time: {time.time() - t0:.2f}s")
    print("\nDone!")


if __name__ == "__main__":
    main()
