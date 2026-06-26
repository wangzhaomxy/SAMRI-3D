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
python utils/gen_training_list.py \
--mode 2d \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_small.pt \
--data-paths /scratch/user/s4670484/Datasets/SAMRI_train_test/ \
--save-path /scratch/user/s4670484/training_lists/training_list_2d.pkl \
--num-workers 12

2. Generate 3D training list (for volume training):
python utils/gen_training_list.py \
--mode 3d \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_small.pt \
--data-paths /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ \
--save-path /scratch/user/s4670484/training_lists/training_list_3d.pkl \
--prompt-mode box \
--num-workers 16

3. Generate both lists:
python utils/gen_training_list.py \
--mode both \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_small.pt \
--data-paths /scratch/user/s4670484/Datasets/SAMRI_train_test/ \
--save-dir /scratch/user/s4670484/training_lists/ \
--num-workers 16

=== Training with Filtered Samples ===

After generating the lists, use them with train_unified.py (the 2d_img mode):
   python train_unified.py \
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
       --pretrained-ckpt /scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_small.pt \
       --training-list-2d /scratch/user/s4670484/training_lists/training_list_2d.pkl \
       --training-list-3d /scratch/user/s4670484/training_lists/training_list_3d.pkl \
       --dice-threshold-2d 0.4 --dice-threshold-3d 0.2 \
       --save-path /scratch/user/s4670484/Model_save/samri2_filtered/

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
import contextlib
import tempfile
import shutil
import inspect
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# This script lives in utils/; add the repo root (its parent) to sys.path so the
# top-level `sam2` and `utils` packages resolve regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

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
# 3D Evaluation Functions (adapted from test_vis3d_multi.py)
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


def normalize_to_uint8(slice2d: np.ndarray) -> np.ndarray:
    """Normalize slice to uint8 range using plain min-max scaling.

    This matches SAM2's expected preprocessing: images should be normalized
    to [0, 255] uint8 before being passed through SAM2Transforms.
    """
    sl = slice2d.astype(np.float32)
    sl_min, sl_max = sl.min(), sl.max()
    if sl_max - sl_min < 1e-8:
        sl_max = sl_min + 1e-8
    sl = (sl - sl_min) / (sl_max - sl_min)
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


# =============================================================================
# SAM2 Video Predictor Compatibility Helpers
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


def init_state_compat(predictor, video_source: Any, num_frames: int, prefer_streaming: bool = True):
    """
    Version-compatible init_state:
    - Newer SAM2: may accept frame_load_config
    - Mid SAM2: may accept callable video_path but no frame_load_config
    - Older SAM2: requires directory of jpg frames
    """
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
        except Exception:
            # Catch all exceptions (TypeError, ValueError, RuntimeError, etc.)
            # when streaming fails, fallback to JPEG directory mode
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
    """
    Prepare video sources ONCE per case:
    - Try streaming callables first
    - If SAM2 rejects callables, write both videos to jpg dirs ONCE and reuse
    """
    loader1, num1 = make_pil_loader_from_volume(vol0, v1_slices)
    loader2, num2 = make_pil_loader_from_volume(vol0, v2_slices)

    # Try streaming sanity check (init_state once and immediately discard state)
    try:
        with suppress_sam2_output(True):
            _st = init_state_compat(predictor, loader1, num1, prefer_streaming=True)
        # If it succeeded, assume streaming works
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
        # Fallback to jpg dirs (write once)
        fast_tmp = get_fast_tmp_root()
        tmp_base = tempfile.mkdtemp(prefix="sam2_3d_gen_", dir=fast_tmp)
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

        # Prepare video sources ONCE per case (with fallback to JPEG if needed)
        try:
            prepared = prepare_videos_once(predictor, vol0, v1_slices, v2_slices)
        except Exception as e:
            print(f"  Warning: Failed to prepare videos for {img_path}: {e}")
            pbar.update(1)
            continue

        # Multi-object prediction: process all labels at once
        try:
            pred0_semantic, added_labels = predict_multi_object_semantic(
                predictor=predictor,
                prepared=prepared,
                gt_cut_slice=gt_cut,
                prompt_mode=prompt_mode,
                labels=labels,
            )

            if pred0_semantic is not None and added_labels:
                # Restore to original axis order
                pred_zyx_semantic = np.moveaxis(pred0_semantic, 0, primary_axis)

                # Calculate dice for each added label
                for lb in added_labels:
                    gt_lb = (gt_zyx == lb).astype(np.uint8)
                    pred_lb = (pred_zyx_semantic == lb).astype(np.uint8)
                    dsc = dice_3d(pred_lb, gt_lb)

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
            print(f"  Warning: Failed to process {img_path}: {e}")

        # Clean up temp files for this case
        prepared.cleanup()
        pbar.update(1)

    pbar.close()
    return results


def sample_positive_points(mask2d: np.ndarray, num_pos: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Sample positive points from a binary mask for mask prompt mode."""
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int32)
    k = min(num_pos, xs.size)
    idx = np.random.choice(xs.size, size=k, replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)  # (x,y)
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
    """
    Adds one SAM2 object per label with obj_id=label.
    Returns labels actually added (present on cut slice).
    """
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
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    return added


@torch.no_grad()
def propagate_video_multiobj(
    predictor,
    inference_state,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Multi-object propagation.
    Returns: frame_idx -> {obj_id -> mask2d_uint8}
    """
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
    """
    Merge predictions back to a semantic label volume [S,H,W] (int32), background=0.
    Overlap resolution: later writes override earlier ones.
    """
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


def predict_multi_object_semantic(
    predictor,
    prepared: PreparedVideos,
    gt_cut_slice: np.ndarray,
    prompt_mode: str,
    labels: List[int],
) -> Tuple[Optional[np.ndarray], List[int]]:
    """
    Multi-object inference once per video; returns:
      pred0_semantic: [S,H,W] int32 (axis-0 orientation)
      added_labels: labels that were actually prompted (present in cut slice)
    """
    # Suppress all SAM2 output (tqdm bars, prints) to keep dataset-level tqdm clean
    with suppress_sam2_output(True):
        # video1
        st1 = init_state_compat(predictor, prepared.v1_source, prepared.v1_len, prefer_streaming=prepared.use_streaming)
        added = add_prompts_for_all_labels(predictor, st1, prompt_mode, gt_cut_slice, labels, ann_frame_idx=0)
        if not added:
            return None, []
        pred1 = propagate_video_multiobj(predictor, st1)

        # video2
        st2 = init_state_compat(predictor, prepared.v2_source, prepared.v2_len, prefer_streaming=prepared.use_streaming)
        _ = add_prompts_for_all_labels(predictor, st2, prompt_mode, gt_cut_slice, added, ann_frame_idx=0)
        pred2 = propagate_video_multiobj(predictor, st2)

    N = len(prepared.v1_slices) + (len(prepared.v2_slices) - 1)
    H, W = gt_cut_slice.shape
    pred0 = merge_two_videos_to_semantic_volume(
        N=N, H=H, W=W,
        v1_slices=prepared.v1_slices,
        v2_slices=prepared.v2_slices,
        v1_pred=pred1,
        v2_pred=pred2,
        labels=added,
    )
    return pred0, added


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
