#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_3d_embeddings.py

Generate SAM2-compatible embeddings for 3D volumes.

This script pre-computes SAM2 image embeddings for all slices in 3D volumes,
storing them in a format optimized for 3D training with memory bank.

Output NPZ format (one file per volume):
- image_embeds: (N, 256, 64, 64) - embeddings for all slices
- high_res_feat_0s: (N, 32, 256, 256) - high-res features level 0
- high_res_feat_1s: (N, 64, 128, 128) - high-res features level 1
- masks: (N, H, W) - ground truth masks for all slices
- ori_h: int - original height
- ori_w: int - original width
- n_slices: int - number of slices
- cut_idx: int - cut slice index (largest foreground)
- labels: array - valid non-zero labels
- v1_slices: array - video 1 slice indices (cut→0, reverse order)
- v2_slices: array - video 2 slice indices (cut→end, forward order)
- primary_axis: int - the axis used for slicing
- img_path: str - original image path
- mask_path: str - original mask path
- dataset: str - dataset name

=== Usage ===

1. Generate 3D embeddings from training list:
   python generate_3d_embeddings.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /path/to/sam2.1_hiera_small.pt \\
       --training-list /path/to/training_list_3d.pkl \\
       --output-dir /path/to/3D_embeddings/ \\
       --dice-threshold 0.2 \\
       --num-workers 16

2. Generate from raw data paths (no filtering):
   python generate_3d_embeddings.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /path/to/sam2.1_hiera_small.pt \\
       --data-paths /path/to/Datasets/*/training/ \\
       --output-dir /path/to/3D_embeddings/ \\
       --num-workers 8

3. Resume after crash (automatically skips existing embeddings):
   python generate_3d_embeddings.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /path/to/sam2.1_hiera_small.pt \\
       --training-list /path/to/training_list_3d.pkl \\
       --output-dir /path/to/3D_embeddings/ \\
       --num-workers 8

=== Train with 3D embeddings ===

   python train_embedding_3dwise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --stage1-ckpt /path/to/stage1_checkpoint.pth \\
       --embedding-dir /path/to/3D_embeddings/ \\
       --save-path /path/to/stage2_checkpoints/ \\
       --batch-size 2 --num-epochs 50

Example command to generate 3D embeddings:       
python generate_3d_embeddings.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_small.pt \
--training-list /scratch/user/s4670484/training_lists/training_list_3d.pkl \
--output-dir /scratch/user/s4670484/SAM2_embeddings/ \
--dice-threshold 0.2 \
--num-workers 16

python generate_3d_embeddings.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_small.pt \
--data-paths /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/*/validation/ \
--output-dir /scratch/user/s4670484/SAM2_val_embeddings/ \
--num-workers 16       

"""

import os
import sys
import argparse
import pickle
from glob import glob
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

# Prefer local ./sam2
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 range using plain min-max scaling.

    This matches SAM2's expected preprocessing: images should be normalized
    to [0, 255] uint8 before being passed through SAM2Transforms.
    """
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min < 1e-8:
        img_max = img_min + 1e-8
    img = (img - img_min) / (img_max - img_min)
    return (img * 255.0).astype(np.uint8)


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


def choose_primary_axis(shape: Tuple[int, int, int]) -> int:
    """
    Choose primary axis as the shortest dimension.
    This matches test_vis3d_multi.py behavior for consistent video ordering.
    """
    m = min(shape)
    axes = [i for i, s in enumerate(shape) if s == m]
    return axes[0] if axes else 0


def find_cut_slice_and_labels(gt_arr: np.ndarray) -> Tuple[int, List[int], int]:
    """
    Find the cut slice (slice with largest foreground) and unique labels.

    Returns:
        cut_idx: Index of the cut slice
        labels: List of unique non-zero labels
        primary_axis: Shortest axis (for slicing, matches test_vis3d_multi.py)
    """
    # Find shortest axis (matches test_vis3d_multi.py)
    primary_axis = choose_primary_axis(gt_arr.shape)

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


def get_bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Get bounding box from binary mask in XYXY format."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2, y2])


def dice_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-10) -> float:
    """Calculate dice similarity coefficient between two binary masks."""
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def extract_dataset_name(file_path: str, data_paths: List[str]) -> str:
    """Extract dataset name from file path."""
    for dp in data_paths:
        if file_path.startswith(dp):
            parts = dp.rstrip('/').split('/')
            for i, part in enumerate(parts):
                if part in ['training', 'testing', 'validation']:
                    if i > 0:
                        return parts[i - 1]
            return parts[-1] if parts[-1] else parts[-2]
    return os.path.basename(os.path.dirname(os.path.dirname(file_path)))


def get_existing_embeddings(output_dir: str) -> set:
    """Scan output directory for existing embedding files."""
    existing = set()
    if not os.path.exists(output_dir):
        return existing

    npz_files = glob(os.path.join(output_dir, "**", "*_3demb.npz"), recursive=True)
    for npz_path in npz_files:
        dataset_name = os.path.basename(os.path.dirname(npz_path))
        filename = os.path.basename(npz_path)
        base_name = filename.replace("_3demb.npz", "")
        existing.add((dataset_name, base_name))

    return existing


# =============================================================================
# Parallel I/O Helpers
# =============================================================================

def load_volume_pair(vol: Dict) -> Optional[Dict]:
    """Load a volume pair (for parallel prefetching)."""
    try:
        img_arr, _ = read_3d_image(vol["img_path"])
        mask_arr, _ = read_3d_image(vol["mask_path"])

        if img_arr.shape != mask_arr.shape:
            return None

        return {
            "img_arr": img_arr,
            "mask_arr": mask_arr,
            "img_path": vol["img_path"],
            "mask_path": vol["mask_path"],
            "dataset": vol["dataset"],
        }
    except Exception:
        return None


def prefetch_volumes(volumes: List[Dict], num_workers: int, prefetch_size: int):
    """
    Generator that prefetches volumes using a thread pool.

    Args:
        volumes: List of volume dicts with img_path, mask_path, dataset
        num_workers: Number of worker threads
        prefetch_size: Number of volumes to prefetch

    Yields:
        Loaded volume dicts with img_arr, mask_arr
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit initial batch
        futures = {}
        for i, vol in enumerate(volumes[:prefetch_size]):
            future = executor.submit(load_volume_pair, vol)
            futures[future] = i

        next_idx = prefetch_size
        results = {}
        expected_idx = 0

        while futures or expected_idx < len(volumes):
            # Wait for any future to complete
            if futures:
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    import concurrent.futures
                    done, _ = concurrent.futures.wait(
                        futures.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    done_futures = list(done)

                for future in done_futures:
                    idx = futures.pop(future)
                    try:
                        results[idx] = future.result()
                    except Exception:
                        results[idx] = None

                    # Submit next item if available
                    if next_idx < len(volumes):
                        new_future = executor.submit(load_volume_pair, volumes[next_idx])
                        futures[new_future] = next_idx
                        next_idx += 1

            # Yield results in order
            while expected_idx in results:
                result = results.pop(expected_idx)
                expected_idx += 1
                yield result


def save_embeddings_async(save_args: Dict) -> str:
    """Save embeddings to NPZ file (for async saving)."""
    output_path = save_args["output_path"]
    compress = save_args["compress"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        output_path,
        image_embeds=save_args["image_embeds"],
        high_res_feat_0s=save_args["high_res_feat_0s"],
        high_res_feat_1s=save_args["high_res_feat_1s"],
        masks=save_args["masks"],
        ori_h=save_args["ori_h"],
        ori_w=save_args["ori_w"],
        n_slices=save_args["n_slices"],
        cut_idx=save_args["cut_idx"],
        labels=save_args["labels"],
        v1_slices=save_args["v1_slices"],
        v2_slices=save_args["v2_slices"],
        primary_axis=save_args["primary_axis"],
        img_path=save_args["img_path"],
        mask_path=save_args["mask_path"],
        dataset=save_args["dataset"],
    )
    return output_path


def get_base_name_from_image_path(img_path: str) -> str:
    """Extract base name from image path."""
    base_name = os.path.basename(img_path)
    base_name = os.path.splitext(base_name)[0]
    if base_name.endswith('.nii'):
        base_name = base_name[:-4]
    return base_name


# =============================================================================
# Dataset Scanning
# =============================================================================

SUPPORTED_EXTS = {".nii", ".mhd", ".mha", ".nrrd", ".dcm", ".gz"}


def _has_med_ext_or_dir(p: Path) -> bool:
    if p.is_dir():
        return True
    if p.name.lower().endswith(".nii.gz"):
        return True
    return p.suffix.lower() in SUPPORTED_EXTS


def is_img_candidate(p: Path) -> bool:
    return ("_img_" in p.name) and _has_med_ext_or_dir(p) and (p.is_file() or p.is_dir())


def try_find_gt_same_folder(img_path: Path) -> Optional[Path]:
    """Find corresponding ground truth file."""
    gt_name = img_path.name.replace("_img_", "_gt_")
    gt_path = img_path.with_name(gt_name)
    if gt_path.exists():
        return gt_path

    # Try _seg_ pattern
    gt_name = img_path.name.replace("_img_", "_seg_")
    gt_path = img_path.with_name(gt_name)
    if gt_path.exists():
        return gt_path

    return None


def scan_pairs_from_paths(data_paths: List[str]) -> List[Tuple[str, str, str]]:
    """Scan data paths for image-mask pairs. Returns (img_path, mask_path, dataset_name)."""
    pairs = []
    for data_path in data_paths:
        rootp = Path(data_path)
        dataset_name = rootp.parent.name if rootp.name in ['training', 'testing'] else rootp.name

        for p in rootp.rglob("*"):
            if not is_img_candidate(p):
                continue
            gt = try_find_gt_same_folder(p)
            if gt is None or not gt.exists():
                continue
            pairs.append((str(p), str(gt), dataset_name))

    pairs.sort()
    return pairs


def collect_volumes_from_training_list(
    training_list_path: str,
    dice_threshold: float = 0.2,
) -> List[Dict]:
    """
    Collect unique volumes from 3D training list with dice filtering.

    Returns list of dicts with: img_path, mask_path, dataset, labels, dices
    """
    with open(training_list_path, "rb") as f:
        all_samples = pickle.load(f)

    # Group by (img_path, mask_path, dataset)
    volume_dict = defaultdict(lambda: {"labels": [], "dices": []})
    for sample in all_samples:
        if sample["dice"] >= dice_threshold:
            key = (sample["img_path"], sample["mask_path"], sample["dataset"])
            volume_dict[key]["labels"].append(sample["label"])
            volume_dict[key]["dices"].append(sample["dice"])
            volume_dict[key]["img_path"] = sample["img_path"]
            volume_dict[key]["mask_path"] = sample["mask_path"]
            volume_dict[key]["dataset"] = sample["dataset"]

    return list(volume_dict.values())


# =============================================================================
# Embedding Generation
# =============================================================================

def generate_volume_embeddings(
    predictor: SAM2ImagePredictor,
    img_path: str,
    mask_path: str,
    dataset_name: str,
    output_dir: str,
    max_slices: int = 128,
    compress: bool = True,
    dice_threshold: float = 0.2,
    img_arr: Optional[np.ndarray] = None,
    mask_arr: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    Generate embeddings for a single 3D volume.

    Args:
        predictor: SAM2ImagePredictor instance
        img_path: Path to 3D image
        mask_path: Path to 3D mask
        dataset_name: Name of the dataset
        output_dir: Output directory
        max_slices: Maximum slices to keep (subsample if exceeded)
        compress: Whether to compress NPZ files
        dice_threshold: Minimum dice for any label to save volume
        img_arr: Pre-loaded image array (optional, for prefetching)
        mask_arr: Pre-loaded mask array (optional, for prefetching)

    Returns:
        Dict with save_args if successful, None if skipped
    """
    # Read 3D volumes if not pre-loaded
    if img_arr is None:
        img_arr, _ = read_3d_image(img_path)
    if mask_arr is None:
        mask_arr, _ = read_3d_image(mask_path)

    if img_arr.shape != mask_arr.shape:
        print(f"Warning: Shape mismatch {img_arr.shape} vs {mask_arr.shape}, skipping {img_path}")
        return None

    # Find cut slice and labels
    cut_idx, labels, primary_axis = find_cut_slice_and_labels(mask_arr)

    if not labels:
        return None

    # Reorient volumes (move primary axis to front)
    img_reoriented = np.moveaxis(img_arr, primary_axis, 0)
    mask_reoriented = np.moveaxis(mask_arr, primary_axis, 0)

    N, H, W = img_reoriented.shape

    # Subsample if too many slices
    keep_indices = list(range(N))
    if N > max_slices:
        step = max(1, N // max_slices)
        keep_indices = list(range(0, N, step))
        if cut_idx not in keep_indices:
            keep_indices.append(cut_idx)
        keep_indices = sorted(keep_indices)[:max_slices]

        # Remap cut_idx
        cut_idx = keep_indices.index(cut_idx) if cut_idx in keep_indices else len(keep_indices) // 2

        img_reoriented = img_reoriented[keep_indices]
        mask_reoriented = mask_reoriented[keep_indices]
        N = len(keep_indices)

    # Build video indices
    v1_slices, v2_slices = build_two_videos_indices(N, cut_idx)

    # Check if any label passes dice threshold on cut slice
    cut_mask = mask_reoriented[cut_idx]
    cut_img = img_reoriented[cut_idx]
    cut_img_uint8 = normalize_to_uint8(cut_img)
    cut_img_rgb = np.stack([cut_img_uint8, cut_img_uint8, cut_img_uint8], axis=-1)

    predictor.set_image(cut_img_rgb)
    any_label_passes = False

    for label in labels:
        gt_mask = (cut_mask == label).astype(np.float32)
        if gt_mask.sum() == 0:
            continue
        bbox = get_bbox_from_mask(gt_mask)
        if bbox is None:
            continue

        masks, _, _ = predictor.predict(box=bbox, multimask_output=False)
        pred_mask = masks[0] > 0
        dsc = dice_score(gt_mask, pred_mask.astype(np.float32))

        if dsc >= dice_threshold:
            any_label_passes = True
            break

    if not any_label_passes:
        return None

    # Generate embeddings for all slices
    image_embeds = []
    high_res_feat_0s = []
    high_res_feat_1s = []

    for z in range(N):
        slice_img = img_reoriented[z]
        slice_img_uint8 = normalize_to_uint8(slice_img)
        slice_img_rgb = np.stack([slice_img_uint8, slice_img_uint8, slice_img_uint8], axis=-1)

        predictor.set_image(slice_img_rgb)

        # Extract features
        image_embed = predictor._features['image_embed'].cpu().numpy().squeeze(0)
        high_res_feat_0 = predictor._features['high_res_feats'][0].cpu().numpy().squeeze(0)
        high_res_feat_1 = predictor._features['high_res_feats'][1].cpu().numpy().squeeze(0)

        image_embeds.append(image_embed)
        high_res_feat_0s.append(high_res_feat_0)
        high_res_feat_1s.append(high_res_feat_1)

    # Stack embeddings
    image_embeds = np.stack(image_embeds, axis=0)  # (N, 256, 64, 64)
    high_res_feat_0s = np.stack(high_res_feat_0s, axis=0)  # (N, 32, 256, 256)
    high_res_feat_1s = np.stack(high_res_feat_1s, axis=0)  # (N, 64, 128, 128)

    # Create output path
    base_name = get_base_name_from_image_path(img_path)
    output_name = f"{base_name}_3demb.npz"
    out_subdir = os.path.join(output_dir, dataset_name)
    output_path = os.path.join(out_subdir, output_name)

    # Return save args for async saving
    save_args = {
        "output_path": output_path,
        "compress": compress,
        "image_embeds": image_embeds.astype(np.float16),
        "high_res_feat_0s": high_res_feat_0s.astype(np.float16),
        "high_res_feat_1s": high_res_feat_1s.astype(np.float16),
        "masks": mask_reoriented.astype(np.int16),
        "ori_h": H,
        "ori_w": W,
        "n_slices": N,
        "cut_idx": cut_idx,
        "labels": np.array(labels, dtype=np.int16),
        "v1_slices": np.array(v1_slices, dtype=np.int16),
        "v2_slices": np.array(v2_slices, dtype=np.int16),
        "primary_axis": primary_axis,
        "img_path": img_path,
        "mask_path": mask_path,
        "dataset": dataset_name,
    }

    return save_args


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SAM2 3D volume embeddings")

    # Model configuration
    parser.add_argument("--sam2-config", type=str, required=True,
                        help="Path to SAM2 config file")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to SAM2 checkpoint")

    # Data source (either training list or data paths)
    parser.add_argument("--training-list", type=str, default=None,
                        help="Path to 3D training list pickle (from gen_training_list.py)")
    parser.add_argument("--data-paths", type=str, nargs="+", default=None,
                        help="List of data paths (alternative to --training-list)")

    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for embeddings")

    # Processing options
    parser.add_argument("--dice-threshold", type=float, default=0.2,
                        help="Minimum dice score to save volume (default: 0.2)")
    parser.add_argument("--max-slices", type=int, default=128,
                        help="Maximum slices per volume (default: 128)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for parallel I/O (default: 4)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable NPZ compression")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")

    # Resume options
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip volumes with existing embeddings (default: enabled)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Force reprocessing all volumes")

    args = parser.parse_args()

    if args.training_list is None and args.data_paths is None:
        parser.error("Either --training-list or --data-paths is required")

    return args


def main():
    args = parse_args()

    if not HAS_SITK:
        raise RuntimeError("SimpleITK is required. Install with: pip install SimpleITK")

    # Build SAM2 predictor
    print(f"Loading SAM2 model from {args.ckpt_path}...")
    model = build_sam2(
        config_file=args.sam2_config,
        ckpt_path=args.ckpt_path,
        device=args.device,
    )
    predictor = SAM2ImagePredictor(model)

    # Collect volumes to process
    if args.training_list:
        print(f"Loading volumes from training list: {args.training_list}")
        volumes = collect_volumes_from_training_list(
            args.training_list,
            dice_threshold=args.dice_threshold,
        )
        print(f"Found {len(volumes)} volumes after dice filtering (threshold: {args.dice_threshold})")
    else:
        print(f"Scanning data paths for volumes...")
        pairs = scan_pairs_from_paths(args.data_paths)
        volumes = [
            {"img_path": p[0], "mask_path": p[1], "dataset": p[2]}
            for p in pairs
        ]
        print(f"Found {len(volumes)} volumes")

    # Check for existing embeddings
    skip_existing = args.skip_existing and not args.no_skip_existing
    already_exist_count = 0

    if skip_existing:
        print("Scanning for existing embeddings...")
        existing = get_existing_embeddings(args.output_dir)
        print(f"  Found {len(existing)} existing embeddings")

        filtered_volumes = []
        for vol in volumes:
            dataset_name = vol["dataset"]
            base_name = get_base_name_from_image_path(vol["img_path"])
            if (dataset_name, base_name) not in existing:
                filtered_volumes.append(vol)
            else:
                already_exist_count += 1

        volumes = filtered_volumes
        print(f"  Skipping {already_exist_count} already-processed volumes")

    print(f"\nProcessing {len(volumes)} volumes...")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Max slices per volume: {args.max_slices}")
    print(f"  Compression: {'enabled' if not args.no_compress else 'disabled'}")
    print(f"  Parallel workers: {args.num_workers}")

    # Process volumes with prefetching and async saving
    saved_count = 0
    skipped_count = 0

    # Create thread pool for async saving
    save_executor = ThreadPoolExecutor(max_workers=args.num_workers)
    save_futures = []

    # Use prefetch size proportional to num_workers
    prefetch_size = max(4, args.num_workers * 2)

    pbar = tqdm(total=len(volumes), desc="Generating embeddings")

    for loaded_vol in prefetch_volumes(volumes, args.num_workers, prefetch_size):
        if loaded_vol is None:
            skipped_count += 1
            pbar.update(1)
            continue

        try:
            save_args = generate_volume_embeddings(
                predictor=predictor,
                img_path=loaded_vol["img_path"],
                mask_path=loaded_vol["mask_path"],
                dataset_name=loaded_vol["dataset"],
                output_dir=args.output_dir,
                max_slices=args.max_slices,
                compress=not args.no_compress,
                dice_threshold=args.dice_threshold,
                img_arr=loaded_vol["img_arr"],
                mask_arr=loaded_vol["mask_arr"],
            )

            if save_args:
                # Submit async save
                save_future = save_executor.submit(save_embeddings_async, save_args)
                save_futures.append(save_future)
                saved_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            print(f"\nError processing {loaded_vol['img_path']}: {e}")
            skipped_count += 1

        pbar.update(1)

    pbar.close()

    # Wait for all saves to complete
    print(f"\nWaiting for {len(save_futures)} file saves to complete...")
    for future in tqdm(as_completed(save_futures), total=len(save_futures), desc="Saving"):
        try:
            future.result()
        except Exception as e:
            print(f"Error saving file: {e}")
            saved_count -= 1
            skipped_count += 1

    save_executor.shutdown(wait=True)

    print(f"\n{'='*60}")
    print(f"Embedding generation complete!")
    print(f"  Saved: {saved_count} volumes")
    print(f"  Skipped (low dice or error): {skipped_count}")
    print(f"  Already existed: {already_exist_count}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
