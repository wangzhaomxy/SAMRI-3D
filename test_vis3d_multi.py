#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_vis3d_multi.py

Multi-object version of test_vis3d.py for comparison:
- For each case, split volume into two "videos" around cut slice.
- Add prompts for ALL labels (obj_id = label) at frame 0.
- Propagate in video ONCE to get all objects simultaneously.
- Merge video1 + video2 back to 3D semantic prediction.
- Evaluate per label, and save per-(case,label) record rows.

Keeps:
- same dataset scanning: _img_ / _gt_ in same folder
- same streaming loader / /dev/shm JPG fallback
- suppress SAM2 internal tqdm prints so only dataset-level tqdm shows

CLI:
python test_vis3d_multi.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/sam2.1_hiera_small.pt \
  --dataset-path /path/to/Datasets/Test_3d/ \
  --save-path /path/to/Eval_results/SAM2/sam2_hiera_s_3d_box_multi.pkl \
  --prompt-mode box \
  --device cuda
  
python test_vis3d_multi.py \
--sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
--ckpt-path /home/s4670484/Desktop/Scratch/samri/pretrained_ckpt/sam2.1_hiera_tiny.pt \
--dataset-path /home/s4670484/Desktop/Scratch/samri/Datasets/SAMRI_datasets_3D/ \
--save-path /home/s4670484/Desktop/Scratch/samri/Eval_results/SAM2/sam2_hiera_t_3d_multi.pkl \
--prompt-mode box \
--device cuda
  
Available SAM2 configs/checkpoints:
    sam2.1_hiera_t.yaml     sam2.1_hiera_tiny.pt
    sam2.1_hiera_s.yaml     sam2.1_hiera_small.pt
    sam2.1_hiera_b+.yaml    sam2.1_hiera_base_plus.pt
    sam2.1_hiera_l.yaml     sam2.1_hiera_large.pt
    SAMRI_datasets_3D/     Test_3d/
    box / mask          
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
from typing import Dict, List, Tuple, Optional, Callable, Any, Union

import numpy as np
import SimpleITK as sitk
from PIL import Image

import torch

# tqdm optional
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Prefer local ./sam2
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2_video_predictor

import contextlib


@contextlib.contextmanager
def suppress_sam2_output(enabled: bool = True):
    """
    Suppress stdout/stderr (SAM2 prints + internal tqdm bars) inside the context.
    Keeps dataset-level tqdm clean.
    """
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
    gt_name = replace_img_to_gt_name(img_path.name)
    gt_path = img_path.with_name(gt_name)
    if gt_path.exists():
        return gt_path

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


def collect_test_folders(dataset_path: str) -> List[str]:
    dataset_path = dataset_path.rstrip("/")
    ds_folders = [d for d in sorted(glob(dataset_path + "/*")) if os.path.isdir(d)]
    if not ds_folders:
        return [dataset_path]

    test_folders: List[str] = []
    for ds in ds_folders:
        cand = os.path.join(ds, "testing")
        test_folders.append(cand if os.path.isdir(cand) else ds)
    return test_folders


# ----------------------------
# Empty sample filtering
# ----------------------------
def is_image_empty(img_zyx: np.ndarray, eps: float = 1e-8) -> bool:
    return float(np.max(np.abs(img_zyx))) <= eps


def get_labels_from_mask(gt_zyx: np.ndarray) -> List[int]:
    labels = [int(x) for x in np.unique(gt_zyx)]
    return sorted([x for x in labels if x != 0])


# ----------------------------
# Cut-slice selection + split videos
# ----------------------------
def shortest_axes(shape: Tuple[int, int, int]) -> List[int]:
    m = min(shape)
    return [i for i, s in enumerate(shape) if s == m]


def choose_primary_axis(shape: Tuple[int, int, int]) -> int:
    ax = shortest_axes(shape)
    return ax[0] if ax else 0


def largest_mask_slice_index(mask: np.ndarray, axis: int, label: Optional[int] = None) -> int:
    moved = np.moveaxis(mask, axis, 0)  # [S, ...]
    fg = (moved == label) if (label is not None) else (moved > 0)
    sums = fg.sum(axis=tuple(range(1, moved.ndim)))
    return int(np.argmax(sums))


def build_two_videos_indices(N: int, n: int) -> Tuple[List[int], List[int]]:
    v1 = list(range(n, -1, -1))
    v2 = [n] + list(range(n + 1, N))
    return v1, v2


# ----------------------------
# Frame adapters: streaming callable + jpg fallback (reused per case)
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
    vol_shw: np.ndarray,
    slice_indices: List[int],
) -> Tuple[Callable[[int], Image.Image], int]:
    def loader_func(frame_idx: int) -> Image.Image:
        z = slice_indices[frame_idx]
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        return Image.fromarray(rgb)

    return loader_func, len(slice_indices)


def get_fast_tmp_root() -> Optional[str]:
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return None


def write_video_frames_to_dir(vol_shw: np.ndarray, slice_indices: List[int], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    for i, z in enumerate(slice_indices):
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, rgb := s2d], axis=-1)  # small trick to avoid extra var
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
        except TypeError:
            pass

    if not isinstance(video_source, str):
        raise TypeError("Fallback init_state requires a directory path (str) as video_source.")
    return predictor.init_state(video_path=video_source)


class PreparedVideos:
    """
    Prepared per-case video sources for video1 and video2.
    """
    def __init__(
        self,
        v1_source: Union[Callable[[int], Image.Image], str],
        v2_source: Union[Callable[[int], Image.Image], str],
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
    loader1, num1 = make_pil_loader_from_volume(vol0, v1_slices)
    loader2, num2 = make_pil_loader_from_volume(vol0, v2_slices)

    # streaming sanity check
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
        fast_tmp = get_fast_tmp_root()
        tmp_base = tempfile.mkdtemp(prefix="sam2_3d_case_", dir=fast_tmp)
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


# ----------------------------
# Prompts: multi-object
# ----------------------------
def mask2d_to_bbox_xyxy(mask2d: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def sample_positive_points(mask2d: np.ndarray, num_pos: int = 256) -> Tuple[np.ndarray, np.ndarray]:
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


# ----------------------------
# SAM2 propagation (multi-object) + merge
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
    # video1
    with suppress_sam2_output(True):
        st1 = init_state_compat(predictor, prepared.v1_source, prepared.v1_len, prefer_streaming=prepared.use_streaming)
        added = add_prompts_for_all_labels(predictor, st1, prompt_mode, gt_cut_slice, labels, ann_frame_idx=0)
        if not added:
            return None, []
        pred1 = propagate_video_multiobj(predictor, st1)

    # video2
    with suppress_sam2_output(True):
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


# ----------------------------
# Metrics (3D): DSC, HD, MSD
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
# Main record function (per-case, per-label rows)
# ----------------------------
def save_test_record_3d_multi_object(
    dataset_path: str,
    predictor,
    prompt_mode: str,
    save_path: str,
) -> Dict[str, List[Dict]]:
    """
    Saves pickle:
      final_record = {ds_name: [row, row, ...], ...}

    Each row corresponds to ONE (case, label):
      img_name, mask_name, label,
      voxel_count, voxel_percentage,
      dice, hd, msd,
      cut_axis, cut_idx
    """
    final_record: Dict[str, List[Dict]] = {}

    test_folders = collect_test_folders(dataset_path)
    for test_folder in test_folders:
        tf = Path(test_folder).resolve()
        ds_name = tf.parent.name if tf.name == "testing" else tf.name

        pairs = scan_pairs_same_folder(test_folder)
        if not pairs:
            final_record[ds_name] = []
            continue

        iterator = pairs
        if tqdm is not None:
            iterator = tqdm(pairs, desc=f"{ds_name}", leave=True)

        ds_rows: List[Dict] = []
        skipped_empty = 0
        skipped_unpromptable_labels = 0
        skipped_errors = 0
        kept_labels = 0

        for (img_path, gt_path) in iterator:
            try:
                img_zyx, img_sitk = read_3d_image(img_path)
                gt_zyx, _ = read_3d_image(gt_path)
                ensure_same_shape(img_zyx, gt_zyx, img_path, gt_path)

                if is_image_empty(img_zyx):
                    skipped_empty += 1
                    continue
                labels = get_labels_from_mask(gt_zyx)
                if not labels:
                    skipped_empty += 1
                    continue

                # choose cut slice on shortest axis
                primary_axis = choose_primary_axis(img_zyx.shape)
                cut_idx = largest_mask_slice_index(gt_zyx, primary_axis, label=None)

                vol0 = np.moveaxis(img_zyx, primary_axis, 0)  # [S,H,W]
                gt0 = np.moveaxis(gt_zyx, primary_axis, 0)
                N, H, W = vol0.shape
                n = int(cut_idx)
                v1_slices, v2_slices = build_two_videos_indices(N, n)
                gt_cut = gt0[n]  # [H,W]

                prepared = prepare_videos_once(predictor, vol0, v1_slices, v2_slices)

                spacing_xyz = img_sitk.GetSpacing()  # (x,y,z)
                spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
                total_voxels = int(gt_zyx.size)

                try:
                    pred0_sem, added = predict_multi_object_semantic(
                        predictor=predictor,
                        prepared=prepared,
                        gt_cut_slice=gt_cut,
                        prompt_mode=prompt_mode,
                        labels=labels,
                    )
                    if pred0_sem is None or not added:
                        # no promptable labels on cut slice => treat as empty-like
                        skipped_empty += 1
                        continue

                    # restore to zyx
                    pred_zyx_sem = np.moveaxis(pred0_sem, 0, primary_axis)  # [Z,Y,X]

                    # record per label
                    for lb in labels:
                        if lb not in added:
                            skipped_unpromptable_labels += 1
                            continue

                        gt_lb = (gt_zyx == lb).astype(np.uint8)
                        pr_lb = (pred_zyx_sem == lb).astype(np.uint8)

                        dsc = dice_3d(pr_lb, gt_lb)
                        hd, msd = hd_msd_3d_sitk(pr_lb, gt_lb, spacing_zyx=spacing_zyx)

                        vc = int(gt_lb.sum())
                        vp = vc / float(total_voxels)

                        ds_rows.append({
                            "img_name": img_path,
                            "mask_name": gt_path,
                            "label": int(lb),
                            "voxel_count": vc,
                            "voxel_percentage": vp,
                            "dice": float(dsc),
                            "hd": float(hd),
                            "msd": float(msd),
                            "cut_axis": int(primary_axis),
                            "cut_idx": int(cut_idx),
                        })
                        kept_labels += 1

                finally:
                    prepared.cleanup()

            except Exception:
                skipped_errors += 1
                continue

        final_record[ds_name] = ds_rows
        print(
            f"[Done] {ds_name}: cases={len(pairs)} | kept_labels={kept_labels} | "
            f"skip_empty_cases={skipped_empty}, skip_unpromptable_labels={skipped_unpromptable_labels}, errors={skipped_errors}"
        )

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

    save_test_record_3d_multi_object(
        dataset_path=args.dataset_path,
        predictor=predictor,
        prompt_mode=args.prompt_mode,
        save_path=args.save_path,
    )

    print(f"\nSaved: {args.save_path}")
    print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
