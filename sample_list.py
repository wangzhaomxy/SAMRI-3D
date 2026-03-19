#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_list.py

Scan one or more dataset root directories, enumerate all image/mask pairs,
and write a CSV with per-sample metadata.

Columns
-------
dataset_name       : leaf folder name of the individual dataset
split              : data split subfolder ("training", "testing", "validation", or "all")
sample_name        : image file stem (without extension)
image_path         : absolute path to image file
mask_path          : absolute path to mask file
shape_z/y/x        : volume dimensions in SITK [z, y, x] order
spacing_z/y/x_mm   : voxel spacing in mm
origin_x/y/z       : image origin
primary_axis       : axis used for slicing (shortest dim: 0=z,1=y,2=x)
n_slices           : size along primary_axis
n_labels           : number of unique non-background labels
labels             : semicolon-separated label ids  (e.g. "1;2;3")
label_voxel_counts : semicolon-separated voxel count per label
label_volumes_mm3  : semicolon-separated physical volume (mm³) per label
image_min          : image intensity minimum
image_max          : image intensity maximum
image_mean         : image intensity mean
image_std          : image intensity standard deviation
warnings           : pipe-separated non-fatal issues detected (empty = clean)
                     possible values:
                       shape_mismatch         — image and mask shapes differ
                       spacing_may_be_default — all spacings are 1.0 (header may be missing)
                       empty_mask             — mask has no foreground labels
                       blank_image            — image intensity is all near-zero

Usage
-----
python sample_list.py \
    --data-paths /path/to/dataset1/ /path/to/dataset2/ \
    --save-path  /path/to/sample_list.csv \
    --num-workers 8
    
e.g.
python sample_list.py \
--data-paths /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ /scratch/user/s4670484/Datasets/Zeroshot3D/ \
--save-path /scratch/user/s4670484/sample_detail.csv \
--num-workers 12
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset scanning (mirrors test_vis3d_multi.py conventions)
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".nii", ".mhd", ".mha", ".nrrd", ".dcm", ".gz"}
PREFERRED_EXTS = [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]


def _is_nii_gz(p: Path) -> bool:
    return p.name.lower().endswith(".nii.gz")


def _has_med_ext(p: Path) -> bool:
    if p.is_dir():
        return True
    if _is_nii_gz(p):
        return True
    return p.suffix.lower() in SUPPORTED_EXTS


def _try_find_gt(img_path: Path) -> Optional[Path]:
    gt_name = img_path.name.replace("_img_", "_gt_")
    gt_path = img_path.with_name(gt_name)
    if gt_path.exists():
        return gt_path

    base = gt_name[:-7] if gt_name.lower().endswith(".nii.gz") else Path(gt_name).stem
    for ext in PREFERRED_EXTS:
        cand = img_path.with_name(base + ext)
        if cand.exists():
            return cand
    cands = sorted(img_path.parent.glob(base + ".*"))
    return cands[0] if cands else None


def scan_pairs(folder: str) -> List[Tuple[str, str]]:
    """Find all (image, mask) pairs under *folder* (recursive)."""
    rootp = Path(folder)
    pairs: List[Tuple[str, str]] = []
    for p in rootp.rglob("*"):
        if "_img_" not in p.name:
            continue
        if not _has_med_ext(p):
            continue
        if not (p.is_file() or p.is_dir()):
            continue
        gt = _try_find_gt(p)
        if gt is None or not gt.exists():
            continue
        pairs.append((str(p), str(gt)))
    pairs.sort()
    return pairs


SPLIT_NAMES = ["training", "testing", "validation"]


def collect_all_folders(dataset_paths: List[str]) -> List[Tuple[str, str, str]]:
    """
    For each dataset root, enumerate every dataset sub-folder and then every
    split sub-folder (training / testing / validation).  If no recognised split
    sub-folders exist, the dataset folder itself is used with split="all".

    Returns list of (dataset_name, split, folder_path).
    """
    results: List[Tuple[str, str, str]] = []
    for dp in dataset_paths:
        dp = dp.rstrip("/")
        ds_folders = sorted(d for d in glob(dp + "/*") if os.path.isdir(d))
        if not ds_folders:
            # dp itself is the only dataset folder
            ds_folders = [dp]

        for ds in ds_folders:
            ds_name = Path(ds).name
            split_folders = [
                (split, os.path.join(ds, split))
                for split in SPLIT_NAMES
                if os.path.isdir(os.path.join(ds, split))
            ]
            if split_folders:
                for split, folder in split_folders:
                    results.append((ds_name, split, folder))
            else:
                # No recognised split sub-folders — treat whole folder as "all"
                results.append((ds_name, "all", ds))
    return results


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def stem_of(path: str) -> str:
    """File stem, stripping .nii.gz or any single extension."""
    p = Path(path)
    name = p.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return p.stem


def choose_primary_axis(shape: Tuple[int, int, int]) -> int:
    m = min(shape)
    return next(i for i, s in enumerate(shape) if s == m)


def process_pair(
    dataset_name: str,
    split: str,
    img_path: str,
    msk_path: str,
) -> Optional[Dict]:
    """
    Read one image/mask pair and return a metadata dict (or None on fatal error).

    Non-fatal issues are collected in the 'warnings' field of the returned dict
    so they are visible in the CSV without dropping the row entirely.
    Fatal errors (unreadable file, wrong dimensionality) return None and print
    a [ERROR] line to stderr.
    """
    try:
        sitk_img = sitk.ReadImage(img_path)
        sitk_msk = sitk.ReadImage(msk_path)

        arr = sitk.GetArrayFromImage(sitk_img)   # [z, y, x]
        msk = sitk.GetArrayFromImage(sitk_msk)   # [z, y, x]

        # ---- Fatal checks (wrong dimensionality — cannot produce meaningful stats) ----
        if arr.ndim != 3:
            print(f"[ERROR] {img_path}: image is {arr.ndim}D, expected 3D — skipped",
                  file=sys.stderr)
            return None
        if msk.ndim != 3:
            print(f"[ERROR] {msk_path}: mask is {msk.ndim}D, expected 3D — skipped",
                  file=sys.stderr)
            return None

        spacing = sitk_img.GetSpacing()   # (x, y, z) in SimpleITK
        origin  = sitk_img.GetOrigin()    # (x, y, z)

        shape_z, shape_y, shape_x = arr.shape
        spacing_x, spacing_y, spacing_z = spacing
        origin_x, origin_y, origin_z = origin

        # ---- Non-fatal checks: collect warnings, keep the row ----
        warnings: List[str] = []

        # Shape mismatch between image and mask — label stats will be wrong
        if arr.shape != msk.shape:
            warnings.append(
                f"shape_mismatch:image={arr.shape},mask={msk.shape}"
            )
            print(f"[WARN] {img_path}: image/mask shape mismatch "
                  f"{arr.shape} vs {msk.shape}", file=sys.stderr)

        # Default spacing (1, 1, 1) suggests the header is missing physical metadata;
        # label_volumes_mm3 will be in voxels, not mm³
        if all(abs(s - 1.0) < 1e-6 for s in spacing):
            warnings.append("spacing_may_be_default:physical_volumes_unreliable")

        # Empty mask — no foreground labels
        if msk.max() == 0:
            warnings.append("empty_mask:no_foreground_labels")

        # Image appears blank
        if float(np.abs(arr).max()) < 1e-8:
            warnings.append("blank_image")

        primary_axis = choose_primary_axis(arr.shape)
        n_slices = arr.shape[primary_axis]

        # Label statistics — computed on mask as-is even if shape mismatch warned above
        labels = sorted(int(v) for v in np.unique(msk) if v != 0)
        voxel_vol_mm3 = float(spacing_x * spacing_y * spacing_z)
        label_voxel_counts = [int((msk == lb).sum()) for lb in labels]
        label_volumes_mm3  = [round(cnt * voxel_vol_mm3, 4) for cnt in label_voxel_counts]

        # Image intensity statistics (use float32 to avoid overflow on int16)
        arr_f = arr.astype(np.float32)
        img_min  = float(arr_f.min())
        img_max  = float(arr_f.max())
        img_mean = float(arr_f.mean())
        img_std  = float(arr_f.std())

        return {
            "dataset_name":        dataset_name,
            "split":               split,
            "sample_name":         stem_of(img_path),
            "image_path":          img_path,
            "mask_path":           msk_path,
            "shape_z":             shape_z,
            "shape_y":             shape_y,
            "shape_x":             shape_x,
            "spacing_z_mm":        round(spacing_z, 6),
            "spacing_y_mm":        round(spacing_y, 6),
            "spacing_x_mm":        round(spacing_x, 6),
            "origin_x":            round(origin_x, 4),
            "origin_y":            round(origin_y, 4),
            "origin_z":            round(origin_z, 4),
            "primary_axis":        primary_axis,
            "n_slices":            n_slices,
            "n_labels":            len(labels),
            "labels":              ";".join(str(lb) for lb in labels),
            "label_voxel_counts":  ";".join(str(c) for c in label_voxel_counts),
            "label_volumes_mm3":   ";".join(str(v) for v in label_volumes_mm3),
            "image_min":           round(img_min, 4),
            "image_max":           round(img_max, 4),
            "image_mean":          round(img_mean, 4),
            "image_std":           round(img_std, 4),
            "warnings":            "|".join(warnings),
        }

    except Exception as e:
        print(f"[ERROR] Failed to read {img_path}: {e}", file=sys.stderr)
        return None


# worker must be top-level for multiprocessing pickle
def _worker(args):
    return process_pair(*args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "dataset_name", "split", "sample_name", "image_path", "mask_path",
    "shape_z", "shape_y", "shape_x",
    "spacing_z_mm", "spacing_y_mm", "spacing_x_mm",
    "origin_x", "origin_y", "origin_z",
    "primary_axis", "n_slices",
    "n_labels", "labels", "label_voxel_counts", "label_volumes_mm3",
    "image_min", "image_max", "image_mean", "image_std",
    "warnings",
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CSV listing all sample metadata in the dataset."
    )
    parser.add_argument(
        "--data-paths", nargs="+", required=True,
        help="One or more dataset root directories."
    )
    parser.add_argument(
        "--save-path", type=str, default="sample_list.csv",
        help="Output CSV file path."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Parallel worker processes (0 = single-threaded)."
    )
    args = parser.parse_args()

    # Collect all (dataset_name, split, folder) tuples
    ds_folders = collect_all_folders(args.data_paths)
    print(f"Found {len(ds_folders)} dataset/split folder(s).")

    # Scan all image/mask pairs
    tasks = []   # (dataset_name, split, img_path, msk_path)
    for ds_name, split, folder in ds_folders:
        pairs = scan_pairs(folder)
        for img, msk in pairs:
            tasks.append((ds_name, split, img, msk))

    print(f"Found {len(tasks)} image/mask pair(s) total.")
    if not tasks:
        print("Nothing to process. Check --data-paths.", file=sys.stderr)
        sys.exit(1)

    # Process
    rows = []
    if args.num_workers <= 1:
        for task in tqdm(tasks, desc="Processing"):
            row = process_pair(*task)
            if row is not None:
                rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                row = fut.result()
                if row is not None:
                    rows.append(row)

    # Sort by dataset_name, sample_name for deterministic output
    rows.sort(key=lambda r: (r["dataset_name"], r["split"], r["sample_name"]))

    # Write CSV
    out_path = Path(args.save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to: {out_path}")

    # Quick summary
    if rows:
        import collections
        by_ds = collections.Counter((r["dataset_name"], r["split"]) for r in rows)
        print("\nSamples per dataset/split:")
        for (ds, split), cnt in sorted(by_ds.items()):
            print(f"  {ds:40s}  {split:12s}  {cnt:5d}")
        total_labels = sum(r["n_labels"] for r in rows)
        print(f"\nTotal labels across all samples: {total_labels}")
        warned = [r for r in rows if r["warnings"]]
        if warned:
            print(f"\nSamples with warnings: {len(warned)}")
            for r in warned:
                print(f"  [{r['dataset_name']} / {r['split']}] {r['sample_name']}: {r['warnings']}")


if __name__ == "__main__":
    main()
