#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_vis3d.py

3D SAM2 VideoPredictor evaluation script (fast: NO temp JPG frames).
- Reads 3D MRI volumes + 3D semantic GT masks from common medical formats
- Splits volume into two “videos” around a cut slice (largest foreground slice)
- Runs SAM2 video propagation (streaming frame loader)
- Merges back to a 3D semantic prediction
- Auto-detects ALL labels from GT (excluding 0)
- Computes per-label 3D DSC / HD / MSD
- Saves pickle record in the SAME style as test_vis2d.py:
    final_record: dict[str, list[dict]] = {ds_name: [case_record, ...], ...}
  where each case_record includes:
    img_name, mask_name, labels, pixel_count, area_percentage,
    p_dice/b_dice/bp_dice, p_hd/b_hd/bp_hd, p_msd/b_msd/bp_msd
  (We fill only the list corresponding to prompt-mode; other lists are empty.)

Dataset layout:
- Image and GT are in the SAME folder with different keys:
    "_img_" for image, "_gt_" for GT
- Typically under each dataset folder’s /testing/ subfolder.
- This script expects: --dataset-path points to the root that contains multiple dataset folders.
  For each dataset folder, it will try:
    <ds>/testing/
  If that doesn't exist, it will scan <ds>/ directly.
  (If your layout differs, adjust `collect_test_folders()`.)

CLI:
python test_vis3d.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --ckpt-path /path/to/sam2.1_hiera_large.pt \
  --dataset-path /path/to/SAMRI_train_test/ \
  --save-path /path/to/save/results_3d.pkl \
  --prompt-mode box \
  --device cuda
  
python test_vis3d.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /home/s4670484/Desktop/Scratch/samri/pretrained_ckpt/sam2.1_hiera_small.pt \
--dataset-path /home/s4670484/Desktop/Scratch/samri/Datasets/Test_3d/ \
--save-path /home/s4670484/Desktop/Scratch/samri/Eval_results/SAM2/sam2_hiera_s_3d.pkl \
--prompt-mode box \
--device cuda
  
Available SAM2 configs/checkpoints:
    sam2_hiera_t.yaml     sam2.1_hiera_tiny.pt
    sam2_hiera_s.yaml     sam2.1_hiera_small.pt
    sam2_hiera_b+.yaml    sam2.1_hiera_base_plus.pt
    sam2_hiera_l.yaml     sam2.1_hiera_large.pt
    SAMRI_train_test/     Test_3d/
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import shutil
import inspect
import argparse
import tempfile
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
import SimpleITK as sitk
from PIL import Image

import torch

# Ensure local ./sam2 is importable (preferred over site-packages)
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2_video_predictor


# ----------------------------
# IO: 3D volume reader
# ----------------------------
SUPPORTED_EXTS = {".nii", ".mhd", ".mha", ".nrrd", ".dcm", ".gz"}  # .nii.gz handled specially
PREFERRED_EXTS = [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]


def _is_nii_gz(p: Path) -> bool:
    return p.name.lower().endswith(".nii.gz")


def _has_med_ext_or_dir(p: Path) -> bool:
    if p.is_dir():
        return True
    if _is_nii_gz(p):
        return True
    return p.suffix.lower() in SUPPORTED_EXTS


def read_3d_image(path: str) -> Tuple[np.ndarray, sitk.Image]:
    """
    Read 3D volume using SimpleITK.
    Returns array in SITK order: [z, y, x].
    If path is a directory -> reads first DICOM series found.
    """
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
            raise RuntimeError(f"Expected 3D DICOM series, got shape {arr.shape} from {path}")
        return arr, img

    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D image, got shape {arr.shape} from {path}")
    return arr, img


def ensure_same_shape(img: np.ndarray, msk: np.ndarray, img_path: str, msk_path: str) -> None:
    if img.shape != msk.shape:
        raise ValueError(
            f"Shape mismatch:\n  image: {img.shape} ({img_path})\n  mask : {msk.shape} ({msk_path})"
        )


# ----------------------------
# Dataset scanning: same folder pairing _img_ -> _gt_
# ----------------------------
def is_img_candidate(p: Path) -> bool:
    return ("_img_" in p.name) and _has_med_ext_or_dir(p) and (p.is_file() or p.is_dir())


def replace_img_to_gt_name(name: str) -> str:
    return name.replace("_img_", "_gt_")


def try_find_gt_same_folder(img_path: Path) -> Optional[Path]:
    """
    Find GT in SAME directory:
    1) direct name replacement (same extension)
    2) try common medical extensions with same base name
    3) fallback: first glob match base.* in same folder
    """
    gt_name = replace_img_to_gt_name(img_path.name)
    gt_path = img_path.with_name(gt_name)
    if gt_path.exists():
        return gt_path

    # Match by base (handle .nii.gz)
    if gt_name.lower().endswith(".nii.gz"):
        base = gt_name[:-7]
    else:
        base = Path(gt_name).stem

    for ext in PREFERRED_EXTS:
        cand = img_path.with_name(base + ext)
        if cand.exists():
            return cand

    cands = sorted(img_path.parent.glob(base + ".*"))
    return cands[0] if cands else None


def scan_pairs_same_folder(folder: str) -> List[Tuple[str, str]]:
    """
    Recursively find all *_img_* items and pair with *_gt_* in the same folder.
    """
    rootp = Path(folder)
    pairs: List[Tuple[str, str]] = []
    for p in rootp.rglob("*"):
        if not is_img_candidate(p):
            continue
        gt = try_find_gt_same_folder(p)
        if gt is None or not gt.exists():
            print(f"[Warn] GT not found for image: {p}")
            continue
        pairs.append((str(p), str(gt)))
    pairs.sort()
    return pairs


def collect_test_folders(dataset_path: str) -> List[str]:
    """
    dataset-path contains multiple dataset folders.
    For each dataset folder, prefer <ds>/testing if exists; else scan <ds>.
    If dataset_path itself contains *_img_* files, we just scan dataset_path.
    """
    dataset_path = dataset_path.rstrip("/")

    # If dataset_path already looks like a test folder (contains any _img_), just use it.
    if any("_img_" in p.name for p in Path(dataset_path).rglob("*")):
        # It might still be a multi-dataset root; we’ll handle both cases below.
        pass

    ds_folders = [d for d in sorted(glob(dataset_path + "/*")) if os.path.isdir(d)]
    if not ds_folders:
        return [dataset_path]

    test_folders: List[str] = []
    for ds in ds_folders:
        cand = os.path.join(ds, "testing")
        test_folders.append(cand if os.path.isdir(cand) else ds)
    return test_folders


# ----------------------------
# Cut-slice selection
# ----------------------------
def shortest_axes(shape: Tuple[int, int, int]) -> List[int]:
    m = min(shape)
    return [i for i, s in enumerate(shape) if s == m]


def choose_primary_axis(shape: Tuple[int, int, int]) -> int:
    ax = shortest_axes(shape)
    return ax[0] if ax else 0


def largest_mask_slice_index(mask: np.ndarray, axis: int, label: Optional[int] = None) -> int:
    """
    Slice index along `axis` with maximum foreground area.
    If label is provided: area of (mask==label); else area of (mask>0).
    """
    moved = np.moveaxis(mask, axis, 0)  # [S, ...]
    fg = (moved == label) if (label is not None) else (moved > 0)
    sums = fg.sum(axis=tuple(range(1, moved.ndim)))
    return int(np.argmax(sums))


def build_two_videos_indices(N: int, n: int) -> Tuple[List[int], List[int]]:
    """
    N slices total (0..N-1). Cut slice = n.
    video1: n, n-1, ..., 0
    video2: n, n+1, ..., N-1
    """
    v1 = list(range(n, -1, -1))
    v2 = [n] + list(range(n + 1, N))
    return v1, v2


# ----------------------------
# Frame adapters: streaming callable (preferred) + jpg fallback
# ----------------------------
def normalize_to_uint8(slice2d: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    sl = slice2d.astype(np.float32)
    lo = np.percentile(sl, p_low)
    hi = np.percentile(sl, p_high)
    if hi <= lo + 1e-6:
        lo, hi = float(sl.min()), float(sl.max() + 1e-6)
    sl = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
    return (sl * 255.0).astype(np.uint8)


def make_pil_loader_from_volume(
    vol_shw: np.ndarray,        # [S,H,W]
    slice_indices: List[int],   # maps frame_idx -> slice_idx
) -> Tuple[Callable[[int], Image.Image], int, int, int]:
    """
    Returns loader_func(frame_idx)->PIL.Image RGB, and (num_frames, H, W).
    """
    _, H, W = vol_shw.shape

    def loader_func(frame_idx: int) -> Image.Image:
        z = slice_indices[frame_idx]
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        return Image.fromarray(rgb)

    return loader_func, len(slice_indices), H, W


def get_fast_tmp_root() -> Optional[str]:
    # Prefer RAM-backed tmp on Linux/HPC
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return None


def write_video_frames_to_dir(vol_shw: np.ndarray, slice_indices: List[int], out_dir: str) -> str:
    """
    Writes frames as 00000.jpg, 00001.jpg, ... into out_dir.
    """
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
    - Older SAM2: requires a directory of jpg frames
    """
    sig = inspect.signature(predictor.init_state)
    params = sig.parameters

    # 1) Try streaming callable
    if prefer_streaming and callable(video_source):
        kwargs = {"video_path": video_source}

        # Only pass supported kwargs
        if "frame_load_config" in params:
            kwargs["frame_load_config"] = {"max_frames": int(num_frames), "first_frame_num": 0}
        elif "max_frames" in params:
            kwargs["max_frames"] = int(num_frames)
        elif "num_frames" in params:
            kwargs["num_frames"] = int(num_frames)

        try:
            return predictor.init_state(**kwargs)
        except TypeError as e:
            print(f"[Warn] Streaming init_state failed: {e}")
            # fallthrough to folder mode

    # 2) Folder mode
    if not isinstance(video_source, str):
        raise TypeError("Fallback init_state requires a directory path (str) as video_source.")
    return predictor.init_state(video_path=video_source)


# ----------------------------
# Prompts: box or mask (fallback to points if needed)
# ----------------------------
def mask2d_to_bbox_xyxy(mask2d: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def sample_positive_points(mask2d: np.ndarray, num_pos: int = 128) -> Tuple[np.ndarray, np.ndarray]:
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
    gt_cut_slice: np.ndarray,   # [H,W] label ids
    labels: List[int],
    ann_frame_idx: int = 0,
) -> List[int]:
    """
    Adds one SAM2 object per label with obj_id=label.
    Skips labels absent on cut slice.
    Returns labels actually added.
    """
    added = []
    for lb in labels:
        obj_id = int(lb)
        mask_lb = (gt_cut_slice == lb).astype(np.uint8)
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
            added.append(lb)

        elif prompt_mode == "mask":
            if hasattr(predictor, "add_new_mask"):
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    mask=mask_lb,
                )
                added.append(lb)
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
                added.append(lb)
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    return added


# ----------------------------
# SAM2 propagation + merge
# ----------------------------
@torch.no_grad()
def propagate_video_multiobj(
    predictor,
    inference_state,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
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


def merge_two_videos_to_volume(
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
        nonlocal pred
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


# ----------------------------
# Metrics: 3D DSC, HD, MSD per label
# ----------------------------
def dice_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = (pred & gt).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + gt.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def hd_msd_3d_sitk(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
) -> Tuple[float, float]:
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0, 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("inf"), float("inf")

    pred_img = sitk.GetImageFromArray(pred_bin)  # z,y,x
    gt_img = sitk.GetImageFromArray(gt_bin)

    sx, sy, sz = float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])
    pred_img.SetSpacing((sx, sy, sz))
    gt_img.SetSpacing((sx, sy, sz))

    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(pred_img, gt_img)
    hd = float(hd_filter.GetHausdorffDistance())

    pred_surface = sitk.LabelContour(pred_img)
    gt_surface = sitk.LabelContour(gt_img)

    pred_dt = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_img, squaredDistance=False, useImageSpacing=True))
    gt_dt = sitk.Abs(sitk.SignedMaurerDistanceMap(gt_img, squaredDistance=False, useImageSpacing=True))

    pred_to_gt = sitk.GetArrayFromImage(gt_dt * sitk.Cast(pred_surface, sitk.sitkFloat32))
    gt_to_pred = sitk.GetArrayFromImage(pred_dt * sitk.Cast(gt_surface, sitk.sitkFloat32))

    pred_vals = pred_to_gt[pred_to_gt > 0]
    gt_vals = gt_to_pred[gt_to_pred > 0]

    if pred_vals.size == 0 or gt_vals.size == 0:
        msd = float("inf")
    else:
        msd = float((pred_vals.mean() + gt_vals.mean()) / 2.0)

    return hd, msd


# ----------------------------
# Record saving (same style as test_vis2d.py)
# ----------------------------
def run_two_videos_predict(
    predictor,
    vol0: np.ndarray,   # [S,H,W]
    gt0: np.ndarray,    # [S,H,W] labels
    labels: List[int],
    prompt_mode: str,
    cut_idx: int,
) -> np.ndarray:
    """
    Runs SAM2 on two videos split around cut_idx, returns pred0: [S,H,W] int32 semantic labels.
    Uses streaming callable if possible; else writes jpg frames to /dev/shm for speed.
    """
    N, H, W = vol0.shape
    n = int(cut_idx)
    v1_slices, v2_slices = build_two_videos_indices(N, n)
    gt_cut = gt0[n]  # [H,W]

    loader1, num1, _, _ = make_pil_loader_from_volume(vol0, v1_slices)
    loader2, num2, _, _ = make_pil_loader_from_volume(vol0, v2_slices)

    # We create a per-case tmp base only if we need fallback. (Also safe for folder mode.)
    fast_tmp = get_fast_tmp_root()
    tmp_base = tempfile.mkdtemp(prefix="sam2_3d_", dir=fast_tmp)

    try:
        # Video1 init (try streaming)
        try:
            st1 = init_state_compat(predictor, loader1, num1, prefer_streaming=True)
        except Exception:
            v1_dir = os.path.join(tmp_base, "video1")
            write_video_frames_to_dir(vol0, v1_slices, v1_dir)
            st1 = init_state_compat(predictor, v1_dir, num1, prefer_streaming=False)

        added1 = add_prompts_for_all_labels(predictor, st1, prompt_mode, gt_cut, labels, ann_frame_idx=0)
        if not added1:
            raise RuntimeError("No labels present on cut slice for prompting; cannot proceed.")
        pred1 = propagate_video_multiobj(predictor, st1)

        # Video2 init (try streaming)
        try:
            st2 = init_state_compat(predictor, loader2, num2, prefer_streaming=True)
        except Exception:
            v2_dir = os.path.join(tmp_base, "video2")
            write_video_frames_to_dir(vol0, v2_slices, v2_dir)
            st2 = init_state_compat(predictor, v2_dir, num2, prefer_streaming=False)

        _ = add_prompts_for_all_labels(predictor, st2, prompt_mode, gt_cut, labels, ann_frame_idx=0)
        pred2 = propagate_video_multiobj(predictor, st2)

        pred0 = merge_two_videos_to_volume(N, H, W, v1_slices, v2_slices, pred1, pred2, labels)
        return pred0

    finally:
        # Cleanup (important to avoid filling /dev/shm)
        shutil.rmtree(tmp_base, ignore_errors=True)


def save_test_record_3d(
    dataset_path: str,
    predictor,
    prompt_mode: str,
    save_path: str,
) -> Dict[str, List[Dict]]:
    """
    Evaluate all datasets under dataset_path and pickle:
      {ds_name: record_list}

    Each record_list item mirrors test_vis2d.py keys:
      img_name, mask_name, labels, pixel_count, area_percentage,
      p_dice/b_dice/bp_dice, p_hd/b_hd/bp_hd, p_msd/b_msd/bp_msd
    For this 3D script:
      - if prompt_mode == "box": fill b_* lists; others empty
      - if prompt_mode == "mask": fill p_* lists; others empty
    """
    final_record: Dict[str, List[Dict]] = {}

    test_folders = collect_test_folders(dataset_path)
    for test_folder in test_folders:
        tf = Path(test_folder).resolve()
        # dataset name: if scanning .../<ds>/testing then ds is parent name, else folder name
        ds_name = tf.parent.name if tf.name == "testing" else tf.name

        print(f"\n[Dataset] {ds_name}")
        print(f"[Scan  ] {test_folder}")

        pairs = scan_pairs_same_folder(test_folder)
        if not pairs:
            print(f"[Warn] No *_img_/*_gt_* pairs under: {test_folder}")
            final_record[ds_name] = []
            continue

        ds_records: List[Dict] = []
        for (img_path, gt_path) in pairs:
            img_zyx, img_sitk = read_3d_image(img_path)
            gt_zyx, _ = read_3d_image(gt_path)
            ensure_same_shape(img_zyx, gt_zyx, img_path, gt_path)

            labels = sorted([int(x) for x in np.unique(gt_zyx) if int(x) != 0])
            if not labels:
                continue

            shape = img_zyx.shape  # (Z,Y,X)
            primary_axis = choose_primary_axis(shape)
            cut_idx = largest_mask_slice_index(gt_zyx, primary_axis, label=None)

            # axis-0 view
            vol0 = np.moveaxis(img_zyx, primary_axis, 0)  # [S,H,W]
            gt0 = np.moveaxis(gt_zyx, primary_axis, 0)    # [S,H,W]

            # Predict in axis-0, then restore
            pred0 = run_two_videos_predict(
                predictor=predictor,
                vol0=vol0,
                gt0=gt0,
                labels=labels,
                prompt_mode=prompt_mode,
                cut_idx=cut_idx,
            )
            pred_zyx = np.moveaxis(pred0, 0, primary_axis)  # [Z,Y,X]

            # spacing
            spacing_xyz = img_sitk.GetSpacing()  # (x,y,z)
            spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

            # voxel stats
            total_voxels = int(gt_zyx.size)
            pixel_count = [int(np.sum(gt_zyx == lb)) for lb in labels]
            area_percentage = [pc / float(total_voxels) for pc in pixel_count]

            # 2D-style lists
            p_dice, b_dice, bp_dice = [], [], []
            p_hd, b_hd, bp_hd = [], [], []
            p_msd, b_msd, bp_msd = [], [], []

            for lb in labels:
                gt_lb = (gt_zyx == lb).astype(np.uint8)
                pr_lb = (pred_zyx == lb).astype(np.uint8)

                dsc = dice_3d(pr_lb, gt_lb)
                hd, msd = hd_msd_3d_sitk(pr_lb, gt_lb, spacing_zyx=spacing_zyx)

                if prompt_mode == "box":
                    b_dice.append(dsc); b_hd.append(hd); b_msd.append(msd)
                else:  # "mask"
                    p_dice.append(dsc); p_hd.append(hd); p_msd.append(msd)

            single_data_result = {
                "img_name": img_path,
                "mask_name": gt_path,
                "labels": labels,
                "pixel_count": pixel_count,
                "area_percentage": area_percentage,

                # Keep 2D-style keys
                "p_dice": p_dice,
                "b_dice": b_dice,
                "bp_dice": bp_dice,
                "p_hd": p_hd,
                "b_hd": b_hd,
                "bp_hd": bp_hd,
                "p_msd": p_msd,
                "b_msd": b_msd,
                "bp_msd": bp_msd,

                # Extra debug info (harmless for old readers)
                "cut_axis": int(primary_axis),
                "cut_idx": int(cut_idx),
                "prompt_mode": prompt_mode,
            }
            ds_records.append(single_data_result)

        final_record[ds_name] = ds_records

    # Save pickle
    save_path = str(save_path)
    Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(final_record, f)

    return final_record


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--prompt-mode", type=str, choices=["box", "mask"], default="box")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    print("\n======================================")
    print(f"SAM2 config   : {args.sam2_config}")
    print(f"Checkpoint    : {args.ckpt_path}")
    print(f"Dataset path  : {args.dataset_path}")
    print(f"Save path     : {args.save_path}")
    print(f"Prompt mode   : {args.prompt_mode}")
    print(f"Device        : {args.device}")
    print("======================================\n")

    t0 = time.time()
    predictor = build_sam2_video_predictor(args.sam2_config, args.ckpt_path, device=args.device)

    save_test_record_3d(
        dataset_path=args.dataset_path,
        predictor=predictor,
        prompt_mode=args.prompt_mode,
        save_path=args.save_path,
    )

    print(f"\nSaved: {args.save_path}")
    print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

