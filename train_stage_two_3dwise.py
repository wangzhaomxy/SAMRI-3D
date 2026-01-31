#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_stage_two_3dwise.py

Stage 2 training: 3D-wise training with memory bank on SAM2.

This script trains SAM2 on a mixture of 2D slices and 3D volumes, using the memory
bank mechanism to improve temporal/spatial consistency across slices.

=== Training Strategy ===

Trainable Components:
  - Memory Encoder: Encodes predicted masks into memory features
  - Memory Attention: Conditions current frame on past memories
  - Mask Decoder: Generates segmentation masks from prompts

Frozen Components:
  - Image Encoder: Pre-trained Hiera backbone (frozen to preserve features)
  - Prompt Encoder: Spatial encoding for points/boxes (domain-invariant)

Sample Mix:
  - ~20% 2D samples: Single slices processed without memory (like stage 1)
  - ~80% 3D samples: Full volumes processed slice-by-slice with memory updates

=== Workflow ===

Step 1: Generate training lists with dice scores (if not already done)
  # 2D training list
  python gen_training_list.py \\
      --mode 2d \\
      --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
      --ckpt-path /path/to/sam2.1_hiera_small.pt \\
      --data-paths /path/to/Datasets/ \\
      --save-path /path/to/training_list_2d.pkl

  # 3D training list
  python gen_training_list.py \\
      --mode 3d \\
      --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
      --ckpt-path /path/to/sam2.1_hiera_small.pt \\
      --data-paths /path/to/Datasets/ \\
      --save-path /path/to/training_list_3d.pkl

Step 2: Run stage 2 training
  python train_stage_two_3dwise.py \\
      --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
      --stage1-ckpt /path/to/stage1_checkpoint.pth \\
      --training-list-2d /path/to/training_list_2d.pkl \\
      --training-list-3d /path/to/training_list_3d.pkl \\
      --save-path /path/to/stage2_checkpoints/ \\
      --batch-size 2 --num-epochs 50

=== CLI Examples ===

1. Basic stage 2 training (2x H100 80G GPUs):
   python train_stage_two_3dwise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --stage1-ckpt /path/to/stage1_best.pth \\
       --training-list-2d /path/to/training_list_2d.pkl \\
       --training-list-3d /path/to/training_list_3d.pkl \\
       --dice-threshold-2d 0.4 --dice-threshold-3d 0.2 \\
       --mix-ratio-3d 0.8 \\
       --samples-per-dataset-2d 500 --samples-per-dataset-3d 100 \\
       --save-path /path/to/stage2_checkpoints/ \\
       --batch-size 2 --num-epochs 50 --lr 5e-6 \\
       --num-workers 32

2. With gradient accumulation for larger effective batch:
   python train_stage_two_3dwise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --stage1-ckpt /path/to/stage1_best.pth \\
       --training-list-2d /path/to/training_list_2d.pkl \\
       --training-list-3d /path/to/training_list_3d.pkl \\
       --save-path /path/to/stage2_checkpoints/ \\
       --batch-size 1 --grad-accum-steps 4 \\
       --num-epochs 50 --lr 5e-6

Key Arguments:
  --stage1-ckpt           Path to stage 1 checkpoint (mask decoder fine-tuned)
  --training-list-2d      Path to 2D training list pickle file
  --training-list-3d      Path to 3D training list pickle file
  --dice-threshold-2d     Min dice for 2D samples (default: 0.4)
  --dice-threshold-3d     Min dice for 3D samples (default: 0.2)
  --mix-ratio-3d          Proportion of 3D samples per epoch (default: 0.8)
  --samples-per-dataset-2d  2D samples per dataset per epoch (default: 500)
  --samples-per-dataset-3d  3D volumes per dataset per epoch (default: 100)
  --batch-size            Batch size per GPU (default: 2)
  --grad-accum-steps      Gradient accumulation steps (default: 1)
  --max-slices            Max slices per 3D volume for memory (default: 64)

Logging:
  - Training loss logged to training_log.csv in --save-path every epoch
  - CSV format: epoch, train_loss, loss_2d, loss_3d
  
Hparameters may need adjustment based on GPU memory and dataset size.
Example:
python train_stage_two_3dwise.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--stage1-ckpt /path/to/stage1_checkpoint.pth \
--training-list-2d /path/to/training_list_2d.pkl \
--training-list-3d /path/to/training_list_3d.pkl \
--dice-threshold-2d 0.4 --dice-threshold-3d 0.2 \
--mix-ratio-3d 0.8 \
--samples-per-dataset-2d 500 --samples-per-dataset-3d 100 \
--batch-size 2 --grad-accum-steps 2 \
--lr 5e-6 --num-epochs 50 \
--num-workers 32 --use-amp \
--save-path /path/to/stage2_checkpoints/
"""

import os
import sys
import csv
import time
import shutil
import inspect
import argparse
import pickle
import tempfile
import contextlib
from glob import glob
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from PIL import Image

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

# Prefer local ./sam2
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from utils.losses import DiceFocalLoss


# =============================================================================
# Utility Functions
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


def get_latest_checkpoint(save_path: str, initial_ckpt: str) -> Tuple[str, int]:
    """
    Find the latest checkpoint in save_path and return its path and epoch number.
    If no checkpoint found, return initial_ckpt and epoch 0.

    Args:
        save_path: Directory containing checkpoint files
        initial_ckpt: Fallback checkpoint path (e.g., stage 1 checkpoint)

    Returns:
        Tuple of (checkpoint_path, start_epoch)
    """
    # Look for stage 2 checkpoints
    pattern = os.path.join(save_path, "sam2_stage2_epoch_*.pth")
    ckpt_files = glob(pattern)

    if not ckpt_files:
        return initial_ckpt, 0

    # Extract epoch numbers and find the latest
    epochs = []
    for f in ckpt_files:
        try:
            # Extract epoch from filename like "sam2_stage2_epoch_30.pth"
            basename = os.path.basename(f)
            epoch_str = basename.replace("sam2_stage2_epoch_", "").replace(".pth", "")
            epochs.append((int(epoch_str), f))
        except (ValueError, IndexError):
            continue

    if not epochs:
        return initial_ckpt, 0

    # Sort by epoch and get the latest
    epochs.sort(key=lambda x: x[0], reverse=True)
    latest_epoch, latest_ckpt = epochs[0]

    return latest_ckpt, latest_epoch


def normalize_to_uint8(img: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Normalize image to uint8 range using percentile clipping."""
    img = img.astype(np.float32)
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo + 1e-6:
        lo, hi = float(img.min()), float(img.max() + 1e-6)
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def get_fast_tmp_root() -> Optional[str]:
    """Get fast tmp directory (/dev/shm if available)."""
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return None


def mask2d_to_bbox_xyxy(mask2d: np.ndarray) -> Optional[np.ndarray]:
    """Convert binary mask to bounding box in [x0, y0, x1, y1] format."""
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.sum(pred & gt)
    return 2.0 * intersection / max(np.sum(pred) + np.sum(gt), 1e-8)


# =============================================================================
# 3D Volume Processing Helpers
# =============================================================================

def read_3d_image(path: str) -> Tuple[np.ndarray, Any]:
    """Read 3D volume using SimpleITK. Returns array in [z, y, x] order."""
    if not HAS_SITK:
        raise RuntimeError("SimpleITK is required for 3D processing")

    p = Path(path)
    if p.is_dir():
        # DICOM series
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(p))
        if not series_ids:
            raise RuntimeError(f"No DICOM series found in directory: {path}")
        files = reader.GetGDCMSeriesFileNames(str(p), series_ids[0])
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(str(p))

    arr = sitk.GetArrayFromImage(img)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D image, got shape {arr.shape} from {path}")
    return arr, img


def find_cut_slice_and_labels(gt_arr: np.ndarray) -> Tuple[int, List[int], int]:
    """
    Find the cut slice (slice with largest foreground) and unique labels.

    Returns:
        cut_idx: Index of the cut slice
        labels: List of unique non-zero labels
        primary_axis: Axis with most slices (for reorientation)
    """
    # Find axis with most slices
    primary_axis = int(np.argmax(gt_arr.shape))

    # Move primary axis to front
    gt_reoriented = np.moveaxis(gt_arr, primary_axis, 0)

    # Find unique labels
    labels = sorted([int(lb) for lb in np.unique(gt_reoriented) if lb > 0])

    # Find slice with largest foreground
    fg_counts = (gt_reoriented > 0).sum(axis=(1, 2))
    cut_idx = int(np.argmax(fg_counts))

    return cut_idx, labels, primary_axis


def build_two_videos_indices(N: int, cut_idx: int) -> Tuple[List[int], List[int]]:
    """
    Build slice indices for two videos around the cut slice.

    Video 1: cut_idx -> 0 (reverse order)
    Video 2: cut_idx -> N-1 (forward order)
    """
    v1 = list(range(cut_idx, -1, -1))  # cut to 0
    v2 = [cut_idx] + list(range(cut_idx + 1, N))  # cut to end
    return v1, v2


def make_pil_loader_from_volume(
    vol_shw: np.ndarray,
    slice_indices: List[int],
) -> Tuple[Any, int]:
    """
    Create a callable frame loader for SAM2 video predictor.

    Returns:
        loader: Callable that returns PIL Image for frame index
        num_frames: Number of frames
    """
    num_frames = len(slice_indices)

    def loader(frame_idx: int) -> Image.Image:
        z = slice_indices[frame_idx]
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        return Image.fromarray(rgb)

    return loader, num_frames


class PreparedVideos:
    """Container for prepared video sources for a 3D case."""

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


def write_video_frames_to_dir(vol_shw: np.ndarray, slice_indices: List[int], out_dir: str) -> str:
    """Write volume slices as JPEG frames to a directory."""
    os.makedirs(out_dir, exist_ok=True)
    for i, z in enumerate(slice_indices):
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"{i:05d}.jpg"), quality=95)
    return out_dir


def init_state_compat(predictor, video_source: Any, num_frames: int, prefer_streaming: bool = True):
    """Version-compatible init_state for different SAM2 versions."""
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


def prepare_videos_once(
    predictor,
    vol0: np.ndarray,
    v1_slices: List[int],
    v2_slices: List[int],
) -> PreparedVideos:
    """
    Prepare video sources ONCE per case.
    Tries streaming first, falls back to JPEG directories if needed.
    """
    loader1, num1 = make_pil_loader_from_volume(vol0, v1_slices)
    loader2, num2 = make_pil_loader_from_volume(vol0, v2_slices)

    # Try streaming
    try:
        with suppress_sam2_output(True):
            _ = init_state_compat(predictor, loader1, num1, prefer_streaming=True)
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
        # Fallback to JPEG directories
        fast_tmp = get_fast_tmp_root()
        tmp_base = tempfile.mkdtemp(prefix="sam2_3d_train_", dir=fast_tmp)
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

def volume_sample_factory():
    return {"labels": [], "dices": []}

# =============================================================================
# Stratified Datasets
# =============================================================================

class Stratified2DDataset(Dataset):
    """
    Dataset for stratified 2D slice training with dice filtering.
    Same as StratifiedTrainingDataset in stage 1, but optimized for mixed training.
    """

    def __init__(
        self,
        training_list_path: str,
        dice_threshold: float = 0.4,
        samples_per_dataset: int = 500,
        seed: int = 42,
    ):
        super().__init__()
        self.dice_threshold = dice_threshold
        self.samples_per_dataset = samples_per_dataset
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Load training list
        with open(training_list_path, "rb") as f:
            all_samples = pickle.load(f)

        # Filter by dice threshold and group by dataset
        self.dataset_samples = defaultdict(list)
        for sample in all_samples:
            if sample["dice"] >= dice_threshold:
                ds_name = sample["dataset"]
                self.dataset_samples[ds_name].append(sample)

        self.dataset_names = sorted(self.dataset_samples.keys())

        # Print statistics
        total_filtered = sum(len(v) for v in self.dataset_samples.values())
        print(f"Stratified 2D Dataset:")
        print(f"  Dice threshold: {dice_threshold}")
        print(f"  Total samples after filtering: {total_filtered}")
        print(f"  Number of datasets: {len(self.dataset_names)}")
        print(f"  Samples per dataset per epoch: {samples_per_dataset}")

        self._build_epoch_indices()

    def _build_epoch_indices(self):
        """Build stratified indices for one epoch."""
        self.epoch_indices = []
        for ds_name in self.dataset_names:
            samples = self.dataset_samples[ds_name]
            n = len(samples)
            if n == 0:
                continue
            if n >= self.samples_per_dataset:
                indices = self.rng.choice(n, size=self.samples_per_dataset, replace=False)
            else:
                indices = self.rng.choice(n, size=self.samples_per_dataset, replace=True)

            for idx in indices:
                self.epoch_indices.append((ds_name, idx))

        self.rng.shuffle(self.epoch_indices)

    def reshuffle(self, epoch: int = None):
        """Reshuffle for a new epoch."""
        if epoch is not None:
            self.rng = np.random.RandomState(self.seed + epoch)
        self._build_epoch_indices()

    def __len__(self):
        return len(self.epoch_indices)

    def __getitem__(self, idx):
        ds_name, sample_idx = self.epoch_indices[idx]
        sample = self.dataset_samples[ds_name][sample_idx]

        img_path = sample["img_path"]
        mask_path = sample["mask_path"]
        label = sample["label"]
        slice_idx = sample.get("slice_idx", None)

        # Read images
        img_sitk = sitk.ReadImage(img_path)
        mask_sitk = sitk.ReadImage(mask_path)

        img_arr = sitk.GetArrayFromImage(img_sitk)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)

        # Extract 2D slice
        if img_arr.ndim == 3:
            if slice_idx is not None:
                z = slice_idx
            else:
                # Find slice with most label pixels
                label_mask = (mask_arr == label)
                slice_counts = label_mask.sum(axis=(1, 2))
                z = int(np.argmax(slice_counts))
            img_2d = img_arr[z]
            mask_2d = (mask_arr[z] == label).astype(np.uint8)
        else:
            img_2d = img_arr
            mask_2d = (mask_arr == label).astype(np.uint8)

        # Normalize to uint8 RGB
        img_2d = normalize_to_uint8(img_2d)
        img_rgb = np.stack([img_2d, img_2d, img_2d], axis=-1)

        return {
            "image": img_rgb,
            "mask": mask_2d,
            "label": label,
            "dataset": ds_name,
            "is_3d": False,
        }


class Stratified3DDataset(Dataset):
    """
    Dataset for stratified 3D volume training with dice filtering.
    Returns full volumes for memory-based training.
    """

    def __init__(
        self,
        training_list_path: str,
        dice_threshold: float = 0.2,
        samples_per_dataset: int = 100,
        max_slices: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.dice_threshold = dice_threshold
        self.samples_per_dataset = samples_per_dataset
        self.max_slices = max_slices
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Load training list
        with open(training_list_path, "rb") as f:
            all_samples = pickle.load(f)

        # Group samples by (img_path, dataset) and filter by dice
        # For 3D, we group all labels of a volume together
        self.volume_samples = defaultdict(volume_sample_factory)
        for sample in all_samples:
            if sample["dice"] >= dice_threshold:
                key = (sample["img_path"], sample["mask_path"], sample["dataset"])
                self.volume_samples[key]["labels"].append(sample["label"])
                self.volume_samples[key]["dices"].append(sample["dice"])
                self.volume_samples[key]["img_path"] = sample["img_path"]
                self.volume_samples[key]["mask_path"] = sample["mask_path"]
                self.volume_samples[key]["dataset"] = sample["dataset"]

        # Group by dataset
        self.dataset_volumes = defaultdict(list)
        for key, vol_info in self.volume_samples.items():
            ds_name = vol_info["dataset"]
            self.dataset_volumes[ds_name].append(vol_info)

        self.dataset_names = sorted(self.dataset_volumes.keys())

        # Print statistics
        total_volumes = sum(len(v) for v in self.dataset_volumes.values())
        print(f"Stratified 3D Dataset:")
        print(f"  Dice threshold: {dice_threshold}")
        print(f"  Total volumes after filtering: {total_volumes}")
        print(f"  Number of datasets: {len(self.dataset_names)}")
        print(f"  Volumes per dataset per epoch: {samples_per_dataset}")
        print(f"  Max slices per volume: {max_slices}")

        self._build_epoch_indices()

    def _build_epoch_indices(self):
        """Build stratified indices for one epoch."""
        self.epoch_volumes = []
        for ds_name in self.dataset_names:
            volumes = self.dataset_volumes[ds_name]
            n = len(volumes)
            if n == 0:
                continue
            if n >= self.samples_per_dataset:
                indices = self.rng.choice(n, size=self.samples_per_dataset, replace=False)
            else:
                indices = self.rng.choice(n, size=self.samples_per_dataset, replace=True)

            for idx in indices:
                self.epoch_volumes.append(volumes[idx])

        self.rng.shuffle(self.epoch_volumes)

    def reshuffle(self, epoch: int = None):
        """Reshuffle for a new epoch."""
        if epoch is not None:
            self.rng = np.random.RandomState(self.seed + epoch)
        self._build_epoch_indices()

    def __len__(self):
        return len(self.epoch_volumes)

    def __getitem__(self, idx):
        vol_info = self.epoch_volumes[idx]

        img_path = vol_info["img_path"]
        mask_path = vol_info["mask_path"]
        labels = vol_info["labels"]
        ds_name = vol_info["dataset"]

        # Read 3D volumes
        img_arr, _ = read_3d_image(img_path)
        mask_arr, _ = read_3d_image(mask_path)

        # Find cut slice and reorient
        cut_idx, valid_labels, primary_axis = find_cut_slice_and_labels(mask_arr)

        # Use only labels that passed the filter
        valid_labels = [lb for lb in valid_labels if lb in labels]
        if not valid_labels:
            valid_labels = labels[:1]  # Fallback to first label

        # Reorient volumes
        img_reoriented = np.moveaxis(img_arr, primary_axis, 0)
        mask_reoriented = np.moveaxis(mask_arr, primary_axis, 0)

        # Subsample if too many slices
        N = img_reoriented.shape[0]
        if N > self.max_slices:
            # Keep cut slice, subsample others
            step = max(1, N // self.max_slices)
            keep_indices = list(range(0, N, step))
            if cut_idx not in keep_indices:
                keep_indices.append(cut_idx)
            keep_indices = sorted(keep_indices)[:self.max_slices]

            # Remap cut_idx
            cut_idx = keep_indices.index(cut_idx) if cut_idx in keep_indices else len(keep_indices) // 2

            img_reoriented = img_reoriented[keep_indices]
            mask_reoriented = mask_reoriented[keep_indices]
            N = len(keep_indices)

        # Build video indices
        v1_slices, v2_slices = build_two_videos_indices(N, cut_idx)

        return {
            "image_volume": img_reoriented,  # [N, H, W]
            "mask_volume": mask_reoriented,  # [N, H, W]
            "labels": valid_labels,
            "cut_idx": cut_idx,
            "v1_slices": v1_slices,
            "v2_slices": v2_slices,
            "dataset": ds_name,
            "primary_axis": primary_axis,
            "is_3d": True,
        }


class MixedDataset(Dataset):
    """
    Combined dataset that mixes 2D and 3D samples with configurable ratio.
    """

    def __init__(
        self,
        dataset_2d: Stratified2DDataset,
        dataset_3d: Stratified3DDataset,
        mix_ratio_3d: float = 0.8,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_2d = dataset_2d
        self.dataset_3d = dataset_3d
        self.mix_ratio_3d = mix_ratio_3d
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Calculate epoch size based on mix ratio
        len_2d = len(dataset_2d)
        len_3d = len(dataset_3d)

        # Total samples = max(len_2d / (1-ratio), len_3d / ratio)
        if mix_ratio_3d > 0:
            total_from_3d = int(len_3d / mix_ratio_3d)
        else:
            total_from_3d = 0
        if mix_ratio_3d < 1:
            total_from_2d = int(len_2d / (1 - mix_ratio_3d))
        else:
            total_from_2d = 0

        self.epoch_size = max(total_from_2d, total_from_3d, len_2d + len_3d)
        self.n_3d = int(self.epoch_size * mix_ratio_3d)
        self.n_2d = self.epoch_size - self.n_3d

        print(f"Mixed Dataset:")
        print(f"  2D samples per epoch: {self.n_2d}")
        print(f"  3D samples per epoch: {self.n_3d}")
        print(f"  Mix ratio (3D): {mix_ratio_3d:.1%}")

        self._build_epoch_indices()

    def _build_epoch_indices(self):
        """Build mixed indices for one epoch."""
        # Sample from 2D dataset
        idx_2d = self.rng.choice(len(self.dataset_2d), size=self.n_2d, replace=True)
        # Sample from 3D dataset
        idx_3d = self.rng.choice(len(self.dataset_3d), size=self.n_3d, replace=True)

        # Create mixed indices: (is_3d, idx)
        self.epoch_indices = []
        self.epoch_indices.extend([(False, i) for i in idx_2d])
        self.epoch_indices.extend([(True, i) for i in idx_3d])

        self.rng.shuffle(self.epoch_indices)

    def reshuffle(self, epoch: int = None):
        """Reshuffle both datasets and rebuild mixed indices."""
        if epoch is not None:
            self.rng = np.random.RandomState(self.seed + epoch)
        self.dataset_2d.reshuffle(epoch)
        self.dataset_3d.reshuffle(epoch)
        self._build_epoch_indices()

    def __len__(self):
        return len(self.epoch_indices)

    def __getitem__(self, idx):
        is_3d, dataset_idx = self.epoch_indices[idx]
        if is_3d:
            return self.dataset_3d[dataset_idx]
        else:
            return self.dataset_2d[dataset_idx]


def mixed_collate_fn(batch):
    """
    Custom collate function for mixed dataset.
    Separates 2D and 3D samples since they have different structures.
    """
    samples_2d = [b for b in batch if not b["is_3d"]]
    samples_3d = [b for b in batch if b["is_3d"]]
    return {"samples_2d": samples_2d, "samples_3d": samples_3d}


# =============================================================================
# Model Component Freezing
# =============================================================================

def freeze_stage2_components(model):
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

def train_2d_sample(
    model,
    sample: Dict,
    loss_fn,
    transforms,
    args,
    device,
) -> torch.Tensor:
    """
    Train on a single 2D sample (no memory).
    Similar to stage 1 training.
    """
    image = sample["image"]  # [H, W, 3] numpy
    mask = sample["mask"]  # [H, W] numpy

    # Get original size
    orig_hw = (image.shape[0], image.shape[1])

    # Transform image
    image_tensor = transforms(image).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Prepare mask
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    mask_resized = F.interpolate(mask_tensor, size=(256, 256), mode="nearest")

    # Generate prompt (box)
    box = mask2d_to_bbox_xyxy(mask)
    if box is None:
        return torch.tensor(0.0, device=device)

    # Scale box to transformed coordinates
    scale_x = transforms.resolution / orig_hw[1]
    scale_y = transforms.resolution / orig_hw[0]
    box_scaled = box.copy()
    box_scaled[0] *= scale_x
    box_scaled[2] *= scale_x
    box_scaled[1] *= scale_y
    box_scaled[3] *= scale_y
    box_tensor = torch.from_numpy(box_scaled).unsqueeze(0).to(device)  # [1, 4]

    # Forward pass
    base_model = model.module if hasattr(model, 'module') else model

    with torch.amp.autocast("cuda", enabled=args.use_amp):
        # Get image embeddings
        backbone_out = base_model.forward_image(image_tensor)
        _, vision_feats, vision_pos_embeds, feat_sizes = base_model._prepare_backbone_features(backbone_out)

        # Get prompt embeddings
        sparse_embeddings, dense_embeddings = base_model.sam_prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=None,
        )

        # Decode mask
        low_res_masks, iou_predictions, _, _ = base_model.sam_mask_decoder(
            image_embeddings=backbone_out["vision_features"],
            image_pe=base_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[
                feat.permute(1, 2, 0).view(1, -1, *sz)
                for feat, sz in zip(vision_feats[:-1], feat_sizes[:-1])
            ] if base_model.use_high_res_features_in_sam else None,
        )

        # Compute loss
        loss = loss_fn(low_res_masks, mask_resized)

    return loss


def process_video_with_memory(
    model,
    transforms,
    img_vol: np.ndarray,
    mask_vol: np.ndarray,
    slice_indices: List[int],
    label: int,
    loss_fn,
    device,
    use_amp: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Process a sequence of slices with memory bank, computing loss at each frame.

    This function uses the model directly (not video_predictor) to maintain gradients.

    Args:
        model: SAM2 model (unwrapped from DDP)
        transforms: SAM2Transforms for image preprocessing
        img_vol: [N, H, W] normalized uint8 volume
        mask_vol: [N, H, W] label mask volume
        slice_indices: List of slice indices to process in order
        label: Target label to segment
        loss_fn: Loss function
        device: Torch device
        use_amp: Whether to use automatic mixed precision

    Returns:
        total_loss: Accumulated loss over all slices
        num_slices: Number of slices processed
    """
    if len(slice_indices) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # Accumulate losses as a list, then sum at the end
    # This is more memory efficient than += on tensors
    losses = []
    num_slices = 0

    # Memory bank for tracking
    output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }

    num_frames = len(slice_indices)

    for frame_idx, slice_z in enumerate(slice_indices):
        is_init_cond_frame = (frame_idx == 0)

        # Get current slice image and mask
        img_2d = img_vol[slice_z]
        mask_2d = (mask_vol[slice_z] == label).astype(np.float32)

        # Skip if no foreground
        if mask_2d.sum() == 0 and not is_init_cond_frame:
            continue

        # Prepare image tensor
        img_rgb = np.stack([img_2d, img_2d, img_2d], axis=-1)
        img_tensor = transforms(img_rgb).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Prepare ground truth mask
        gt_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0).unsqueeze(0).to(device)
        gt_resized = F.interpolate(gt_tensor, size=(256, 256), mode="nearest")

        # Get prompt (box) for initial frame
        point_inputs = None
        mask_inputs = None

        if is_init_cond_frame:
            # Get bounding box prompt
            box = mask2d_to_bbox_xyxy(mask_2d.astype(np.uint8))
            if box is None:
                return torch.tensor(0.0, device=device), 0

            # Scale box to image coordinates
            orig_h, orig_w = img_2d.shape
            scale_x = transforms.resolution / orig_w
            scale_y = transforms.resolution / orig_h

            # Convert box to points format for SAM2
            # Box is [x0, y0, x1, y1], convert to corner points
            box_scaled = box.copy()
            box_scaled[0] *= scale_x
            box_scaled[2] *= scale_x
            box_scaled[1] *= scale_y
            box_scaled[3] *= scale_y

            # SAM2 uses point format: top-left (label=2) and bottom-right (label=3) for box
            points = torch.tensor([[
                [box_scaled[0], box_scaled[1]],  # top-left
                [box_scaled[2], box_scaled[3]],  # bottom-right
            ]], dtype=torch.float32, device=device)
            labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)  # box corner labels

            point_inputs = {
                "point_coords": points,
                "point_labels": labels,
            }

        with torch.amp.autocast("cuda", enabled=use_amp):
            # Forward image through backbone
            backbone_out = model.forward_image(img_tensor)

            # Prepare backbone features
            _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)

            # Get high-res features for SAM head
            if len(vision_feats) > 1:
                high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
                ]
            else:
                high_res_features = None

            # Prepare memory-conditioned features
            pix_feat_with_mem = model._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=vision_feats[-1:],
                current_vision_pos_embeds=vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
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
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=False,
            )

            # Compute loss
            pred_resized = F.interpolate(low_res_masks, size=(256, 256), mode="bilinear", align_corners=False)
            loss = loss_fn(pred_resized, gt_resized)

            if not torch.isnan(loss):
                losses.append(loss)
                num_slices += 1

            # Encode memory for next frames
            # IMPORTANT: Detach memory features to prevent gradient accumulation
            # through the entire video. We only train on current frame's prediction,
            # using memory as frozen context. This prevents OOM on long sequences.
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

    # Sum all losses at the end (more memory efficient)
    if losses:
        total_loss = torch.stack(losses).sum()
        return total_loss, num_slices
    return torch.tensor(0.0, device=device, requires_grad=True), 0


def train_3d_sample_with_memory(
    model,
    sample: Dict,
    loss_fn,
    transforms,
    args,
    device,
) -> torch.Tensor:
    """
    Train on a 3D volume using memory bank.
    Processes slices sequentially with proper gradient flow.

    The volume is split into two "videos":
    - Video 1: cut_idx -> 0 (reverse order)
    - Video 2: cut_idx -> N-1 (forward order)

    Each video is processed independently with its own memory bank.
    """
    img_vol = sample["image_volume"]  # [N, H, W] numpy
    mask_vol = sample["mask_volume"]  # [N, H, W] numpy
    labels = sample["labels"]
    cut_idx = sample["cut_idx"]
    v1_slices = sample["v1_slices"]
    v2_slices = sample["v2_slices"]

    N, H, W = img_vol.shape

    # Normalize volume to uint8
    vol_normalized = np.stack([normalize_to_uint8(img_vol[z]) for z in range(N)])

    # Get base model (unwrap DDP if needed)
    base_model = model.module if hasattr(model, 'module') else model

    # Get max slices per video (default to 32 if not set)
    max_slices_per_video = getattr(args, 'max_slices_per_video', 32)

    # Subsample video slices if too long to prevent OOM
    def subsample_slices(slices: List[int], max_len: int) -> List[int]:
        if len(slices) <= max_len:
            return slices
        # Always keep first slice (prompted frame), subsample the rest
        step = max(1, (len(slices) - 1) // (max_len - 1))
        subsampled = [slices[0]] + slices[1::step][:max_len - 1]
        return subsampled

    v1_slices_train = subsample_slices(v1_slices, max_slices_per_video)
    v2_slices_train = subsample_slices(v2_slices, max_slices_per_video)

    # Accumulate losses as a list for memory efficiency
    all_losses = []
    total_slices = 0

    # Process each label
    for label in labels:
        # Check if label exists in cut slice
        gt_cut = mask_vol[cut_idx]
        if (gt_cut == label).sum() == 0:
            continue

        # Process video 1 (cut -> 0)
        if len(v1_slices_train) > 0:
            loss1, n1 = process_video_with_memory(
                model=base_model,
                transforms=transforms,
                img_vol=vol_normalized,
                mask_vol=mask_vol,
                slice_indices=v1_slices_train,
                label=label,
                loss_fn=loss_fn,
                device=device,
                use_amp=args.use_amp,
            )
            if n1 > 0 and loss1.requires_grad:
                all_losses.append(loss1)
                total_slices += n1

        # Process video 2 (cut -> end), skip cut slice (already in v1)
        if len(v2_slices_train) > 1:
            v2_remaining = v2_slices_train[1:]  # Skip cut slice
            loss2, n2 = process_video_with_memory(
                model=base_model,
                transforms=transforms,
                img_vol=vol_normalized,
                mask_vol=mask_vol,
                slice_indices=v2_remaining,
                label=label,
                loss_fn=loss_fn,
                device=device,
                use_amp=args.use_amp,
            )
            if n2 > 0 and loss2.requires_grad:
                all_losses.append(loss2)
                total_slices += n2

    # Clear GPU cache after processing each 3D volume
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
    transforms,
    args,
    device,
    rank,
    scaler=None,
):
    """Train for one epoch on mixed 2D/3D data."""
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

        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Process 2D samples
        for sample in samples_2d:
            loss = train_2d_sample(model, sample, loss_fn, transforms, args, device)
            if not torch.isnan(loss) and loss.item() > 0:
                batch_loss = batch_loss + loss
                loss_2d_total += loss.item()
                num_2d += 1

        # Process 3D samples
        for sample in samples_3d:
            loss = train_3d_sample_with_memory(
                model, sample, loss_fn, transforms, args, device
            )
            if not torch.isnan(loss) and loss.item() > 0:
                batch_loss = batch_loss + loss
                loss_3d_total += loss.item()
                num_3d += 1

        # Average loss over batch
        n_samples = len(samples_2d) + len(samples_3d)
        if n_samples > 0:
            batch_loss = batch_loss / n_samples

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
    parser = argparse.ArgumentParser(description="Stage 2: 3D-wise SAM2 training with memory")

    # Model configuration
    parser.add_argument("--sam2-config", type=str, required=True,
                        help="Path to SAM2 config file")
    parser.add_argument("--stage1-ckpt", type=str, required=True,
                        help="Path to stage 1 checkpoint")

    # Training lists
    parser.add_argument("--training-list-2d", type=str, required=True,
                        help="Path to 2D training list pickle")
    parser.add_argument("--training-list-3d", type=str, required=True,
                        help="Path to 3D training list pickle")

    # Filtering thresholds
    parser.add_argument("--dice-threshold-2d", type=float, default=0.4,
                        help="Min dice for 2D samples (default: 0.4)")
    parser.add_argument("--dice-threshold-3d", type=float, default=0.2,
                        help="Min dice for 3D samples (default: 0.2)")

    # Sampling configuration
    parser.add_argument("--mix-ratio-3d", type=float, default=0.8,
                        help="Proportion of 3D samples per epoch (default: 0.8)")
    parser.add_argument("--samples-per-dataset-2d", type=int, default=500,
                        help="2D samples per dataset per epoch (default: 500)")
    parser.add_argument("--samples-per-dataset-3d", type=int, default=100,
                        help="3D volumes per dataset per epoch (default: 100)")
    parser.add_argument("--max-slices", type=int, default=64,
                        help="Max slices per 3D volume (default: 64)")
    parser.add_argument("--max-slices-per-video", type=int, default=32,
                        help="Max slices per video sequence during training (default: 32). "
                             "Reduces memory usage by subsampling long sequences.")

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
    parser.add_argument("--num-workers", type=int, default=32,
                        help="Number of data loading workers (default: 32)")
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
    ckpt_path, start_epoch = get_latest_checkpoint(args.save_path, args.stage1_ckpt)
    resume_training = start_epoch > 0

    if rank == 0:
        if resume_training:
            print(f"Resuming from epoch {start_epoch}, checkpoint: {ckpt_path}")
        else:
            print(f"Starting fresh training from stage 1 checkpoint: {ckpt_path}")

    # Build SAM2 model for training
    # Note: We use the model directly for both 2D and 3D training
    # (not video_predictor, which is for inference only with @torch.no_grad())
    model = build_sam2(
        config_file=args.sam2_config,
        ckpt_path=ckpt_path,
        device=device,
        mode="train",
        apply_postprocessing=False,
    )

    # Freeze appropriate components
    freeze_stage2_components(model)

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

    # Setup transforms
    transforms = SAM2Transforms(
        resolution=model.module.image_size if hasattr(model, 'module') else model.image_size,
        mask_threshold=0.0,
    )

    # Setup AMP scaler (use new API to avoid deprecation warning)
    scaler = torch.amp.GradScaler(device) if args.use_amp else None

    # Setup datasets
    if rank == 0:
        print("\nLoading 2D dataset...")
    dataset_2d = Stratified2DDataset(
        training_list_path=args.training_list_2d,
        dice_threshold=args.dice_threshold_2d,
        samples_per_dataset=args.samples_per_dataset_2d,
        seed=args.seed,
    )

    if rank == 0:
        print("\nLoading 3D dataset...")
    dataset_3d = Stratified3DDataset(
        training_list_path=args.training_list_3d,
        dice_threshold=args.dice_threshold_3d,
        samples_per_dataset=args.samples_per_dataset_3d,
        max_slices=args.max_slices,
        seed=args.seed,
    )

    if rank == 0:
        print("\nCreating mixed dataset...")
    mixed_dataset = MixedDataset(
        dataset_2d=dataset_2d,
        dataset_3d=dataset_3d,
        mix_ratio_3d=args.mix_ratio_3d,
        seed=args.seed,
    )

    # Dataloader (no sampler for stratified dataset)
    train_loader = DataLoader(
        mixed_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=mixed_collate_fn,
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
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"{'='*60}")

        # Reshuffle dataset
        mixed_dataset.reshuffle(epoch)

        # Train epoch
        avg_loss, avg_loss_2d, avg_loss_3d = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            transforms=transforms,
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
                    f"sam2_stage2_epoch_{epoch + 1}.pth"
                )
                torch.save({
                    "model": model.module.state_dict(),
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                }, save_name)
                print(f"Checkpoint saved: {save_name}")

    # Save final checkpoint
    if rank == 0:
        save_name = os.path.join(args.save_path, "sam2_stage2_final.pth")
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

    print(f"Starting Stage 2 training with {world_size} GPUs")
    print(f"Using MASTER_PORT: {os.environ['MASTER_PORT']}")

    mp.spawn(main, args=(world_size, args), nprocs=world_size)
