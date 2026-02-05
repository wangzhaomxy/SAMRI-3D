#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_vis3d.py

3D SAM2 VideoPredictor evaluation (single-object, iterative propagation).
Processes each label separately, running video propagation once per label.
For faster multi-object evaluation, use test_vis3d_multi.py instead.

Features:
- Reads 3D MRI volumes + 3D semantic GT masks from common medical formats
- Splits volume into two "videos" around cut slice (largest foreground)
- Runs SAM2 video propagation per label (streaming frame loader)
- Merges back to 3D semantic prediction
- Auto-detects ALL labels from GT (excluding 0)
- Computes per-label 3D DSC / HD / MSD metrics
- Parallel data loading for improved throughput

=== Usage ===

1. Basic evaluation with box prompts:
   python test_vis3d.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \\
       --ckpt-path /path/to/sam2.1_hiera_large.pt \\
       --dataset-path /path/to/SAMRI_train_test/ \\
       --save-path /path/to/results_3d.pkl \\
       --prompt-mode box

2. With parallel data loading (faster I/O):
   python test_vis3d.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \\
       --ckpt-path /path/to/sam2.1_hiera_large.pt \\
       --dataset-path /path/to/SAMRI_train_test/ \\
       --save-path /path/to/results_3d.pkl \\
       --num-workers 8

=== Options ===

  --sam2-config     SAM2 config file (yaml)
  --ckpt-path       SAM2 checkpoint file (.pt)
  --dataset-path    Root path containing dataset folders with /testing/ subdirs
  --save-path       Output pickle file path
  --prompt-mode     Prompt type: box or mask (default: box)
  --num-workers     Workers for parallel data loading (default: 4)
  --device          Device to use: cuda or cpu (default: cuda)

=== Available Configs/Checkpoints ===

    sam2.1_hiera_t.yaml     sam2.1_hiera_tiny.pt
    sam2.1_hiera_s.yaml     sam2.1_hiera_small.pt
    sam2.1_hiera_b+.yaml    sam2.1_hiera_base_plus.pt
    sam2.1_hiera_l.yaml     sam2.1_hiera_large.pt
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
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any, Union, Generator

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
    Suppress stdout/stderr (SAM2 prints + tqdm bars) inside the context.
    Keeps your dataset-level tqdm clean.
    """
    if not enabled:
        yield
        return

    # Also helps some tqdm-based libs
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


def make_pil_loader_from_volume(
    vol_shw: np.ndarray,        # [S,H,W]
    slice_indices: List[int],   # maps frame_idx -> slice_idx
) -> Tuple[Callable[[int], Image.Image], int]:
    # Pre-compute all PIL frames so the loader is a simple lookup
    frames: List[Image.Image] = []
    for z in slice_indices:
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        frames.append(Image.fromarray(rgb))

    def loader_func(frame_idx: int) -> Image.Image:
        return frames[frame_idx]

    return loader_func, len(slice_indices)


def get_fast_tmp_root() -> Optional[str]:
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return None


def write_video_frames_to_dir(vol_shw: np.ndarray, slice_indices: List[int], out_dir: str) -> str:
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
    """
    Prepared per-case video sources for video1 and video2.
    - If streaming works: sources are callables (no disk IO).
    - Else: sources are directories with jpgs (written once and reused).
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


# Cached across cases: None = untested, True/False = result from first probe
_streaming_works: Optional[bool] = None


def prepare_videos_once(
    predictor,
    vol0: np.ndarray,            # [S,H,W]
    v1_slices: List[int],
    v2_slices: List[int],
) -> PreparedVideos:
    """
    Prepare video sources ONCE per case:
    - Try streaming callables first (probe only on the very first case;
      result is cached for the rest of the run)
    - If SAM2 rejects callables, write both videos to jpg dirs and reuse
    """
    global _streaming_works

    loader1, num1 = make_pil_loader_from_volume(vol0, v1_slices)
    loader2, num2 = make_pil_loader_from_volume(vol0, v2_slices)

    # Fast path: streaming decision already cached from a previous case
    if _streaming_works is True:
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

    if _streaming_works is None:
        # First case only: probe streaming with one init_state call
        try:
            with suppress_sam2_output(True):
                _ = init_state_compat(predictor, loader1, num1, prefer_streaming=True)
            _streaming_works = True
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
            _streaming_works = False

    # _streaming_works is False: fallback to jpg dirs
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
# Prompts: single-object
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


def add_prompt_single_label(
    predictor,
    inference_state,
    prompt_mode: str,
    gt_cut_slice: np.ndarray,   # [H,W] label ids
    label: int,
    ann_frame_idx: int = 0,
) -> bool:
    """
    Add prompt for exactly ONE label (obj_id=label).
    Returns True if prompt added, False if label not present on cut slice.
    """
    lb = int(label)
    mask_lb = (gt_cut_slice == lb).astype(np.uint8)
    if mask_lb.sum() == 0:
        return False

    if prompt_mode == "box":
        box = mask2d_to_bbox_xyxy(mask_lb)
        if box is None:
            return False
        try:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=lb,
                box=box,
            )
        except TypeError:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=lb,
                points=None,
                labels=None,
                box=box,
            )
        return True

    if prompt_mode == "mask":
        if hasattr(predictor, "add_new_mask"):
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=lb,
                mask=mask_lb,
            )
            return True
        pts, lbls = sample_positive_points(mask_lb, num_pos=256)
        if pts.shape[0] == 0:
            return False
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=lb,
            points=pts,
            labels=lbls,
        )
        return True

    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


# ----------------------------
# SAM2 propagation (single-object) + merge
# ----------------------------
@torch.no_grad()
def propagate_video_singleobj(
    predictor,
    inference_state,
    obj_id: int,
) -> Dict[int, np.ndarray]:
    """
    Returns: frame_idx -> mask2d_uint8 for a single obj_id.
    """
    out: Dict[int, np.ndarray] = {}
    oid = int(obj_id)

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        fi = int(out_frame_idx)
        # find index of this obj_id
        idx = None
        for j, o in enumerate(out_obj_ids):
            if int(o) == oid:
                idx = j
                break
        if idx is None:
            continue
        logits = out_mask_logits[idx]
        if hasattr(logits, "detach"):
            logits = logits.detach()
        m = (logits > 0.0)
        if hasattr(m, "cpu"):
            m = m.cpu().numpy()
        out[fi] = np.squeeze(m).astype(np.uint8)

    return out


def merge_two_videos_single_label(
    N: int,
    H: int,
    W: int,
    v1_slices: List[int],
    v2_slices: List[int],
    v1_pred: Dict[int, np.ndarray],
    v2_pred: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Merge into [S,H,W] uint8 binary mask for ONE label.
    """
    pred = np.zeros((N, H, W), dtype=np.uint8)
    written = set()

    for fidx, m in v1_pred.items():
        z = v1_slices[fidx]
        pred[z] = (m > 0).astype(np.uint8)
        written.add(z)

    for fidx, m in v2_pred.items():
        z = v2_slices[fidx]
        if z in written:
            continue
        pred[z] = (m > 0).astype(np.uint8)
        written.add(z)

    return pred


@torch.no_grad()
def prefill_image_features(predictor, inference_state) -> None:
    """Pre-fill backbone feature cache for all frames.

    SAM2's _get_image_feature uses a 1-entry cache that is replaced on every
    miss.  Pre-computing all frames here means every propagate call hits the
    cache and forward_image is never called again.  reset_state() does not
    touch cached_features, so the pre-fill persists across labels.
    """
    num_frames = inference_state["num_frames"]
    device = inference_state["device"]
    cache = {}
    for i in range(num_frames):
        image = inference_state["images"][i].to(device).float().unsqueeze(0)
        backbone_out = predictor.forward_image(image)
        cache[i] = (image, backbone_out)
    inference_state["cached_features"] = cache


def predict_all_labels_reuse_state(
    predictor,
    prepared: PreparedVideos,
    gt_cut_slice: np.ndarray,   # [H,W]
    prompt_mode: str,
    labels: List[int],
) -> Dict[int, Optional[np.ndarray]]:
    """
    Run single-object inference for every label, reusing inference states.

    init_state is called once per video.  When K > 1 image features are
    pre-filled so that reset_state (which preserves cached_features) lets each
    subsequent label propagate without re-encoding.

    Returns {label: pred0 [S,H,W] uint8 | None}.  Labels not present on the
    cut slice are absent from the dict.
    """
    results: Dict[int, Optional[np.ndarray]] = {}
    N = len(prepared.v1_slices) + (len(prepared.v2_slices) - 1)
    H, W = gt_cut_slice.shape

    # Filter to labels visible on cut slice
    promptable = [int(lb) for lb in labels if np.sum(gt_cut_slice == int(lb)) > 0]
    if not promptable:
        return results

    with suppress_sam2_output(True):
        # --- init states once per case ---
        st1 = init_state_compat(predictor, prepared.v1_source, prepared.v1_len,
                                prefer_streaming=prepared.use_streaming)
        st2 = init_state_compat(predictor, prepared.v2_source, prepared.v2_len,
                                prefer_streaming=prepared.use_streaming)

        # Pre-fill image features when multiple labels and API is available
        if len(promptable) > 1 and hasattr(predictor, "forward_image"):
            prefill_image_features(predictor, st1)
            prefill_image_features(predictor, st2)

        for lb in promptable:
            # --- video1 ---
            ok1 = add_prompt_single_label(predictor, st1, prompt_mode,
                                          gt_cut_slice, lb, ann_frame_idx=0)
            if ok1:
                pred1 = propagate_video_singleobj(predictor, st1, obj_id=lb)
            else:
                pred1 = {}
            predictor.reset_state(st1)

            if not ok1:
                results[lb] = None
                continue

            # --- video2 ---
            ok2 = add_prompt_single_label(predictor, st2, prompt_mode,
                                          gt_cut_slice, lb, ann_frame_idx=0)
            if ok2:
                pred2 = propagate_video_singleobj(predictor, st2, obj_id=lb)
            else:
                pred2 = {}
            predictor.reset_state(st2)

            if not ok2:
                results[lb] = None
                continue

            # --- merge ---
            results[lb] = merge_two_videos_single_label(
                N=N, H=H, W=W,
                v1_slices=prepared.v1_slices,
                v2_slices=prepared.v2_slices,
                v1_pred=pred1,
                v2_pred=pred2,
            )

    return results


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
# Parallel data loading
# ----------------------------
def load_3d_case(pair: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    """
    Load a single 3D case (image + mask) from disk.
    Returns None if case should be skipped (empty or error).
    """
    img_path, gt_path = pair
    try:
        img_zyx, img_sitk = read_3d_image(img_path)
        gt_zyx, _ = read_3d_image(gt_path)
        ensure_same_shape(img_zyx, gt_zyx, img_path, gt_path)

        if is_image_empty(img_zyx):
            return None
        labels = get_labels_from_mask(gt_zyx)
        if not labels:
            return None

        spacing_xyz = img_sitk.GetSpacing()
        spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

        return {
            "img_path": img_path,
            "gt_path": gt_path,
            "img_zyx": img_zyx,
            "gt_zyx": gt_zyx,
            "labels": labels,
            "spacing_zyx": spacing_zyx,
        }
    except Exception:
        return None


def prefetch_data_generator(
    pairs: List[Tuple[str, str]],
    num_workers: int = 4,
    prefetch_size: int = 8,
) -> Generator[Tuple[int, Optional[Dict[str, Any]]], None, None]:
    """
    Generator that prefetches 3D data in parallel using a thread pool.
    Yields (index, data_dict) pairs in order, where data_dict is None for skipped cases.
    """
    if num_workers <= 1:
        for i, pair in enumerate(pairs):
            yield i, load_3d_case(pair)
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        next_submit = 0
        next_yield = 0
        n = len(pairs)

        # Submit initial batch
        while next_submit < n and len(futures) < prefetch_size:
            future = executor.submit(load_3d_case, pairs[next_submit])
            futures[next_submit] = future
            next_submit += 1

        # Yield results in order and submit more
        while next_yield < n:
            if next_yield in futures:
                result = futures.pop(next_yield).result()
                yield next_yield, result
                next_yield += 1

                # Submit next task
                if next_submit < n:
                    future = executor.submit(load_3d_case, pairs[next_submit])
                    futures[next_submit] = future
                    next_submit += 1
            else:
                time.sleep(0.001)


# ----------------------------
# Main record function (per-case, per-label records)
# ----------------------------
def save_test_record_3d_single_object(
    dataset_path: str,
    predictor,
    prompt_mode: str,
    save_path: str,
    num_workers: int = 4,
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

        ds_rows: List[Dict] = []
        skipped_empty = 0
        skipped_prompt = 0
        skipped_errors = 0
        kept_labels = 0

        # Use prefetch generator for parallel data loading
        data_gen = prefetch_data_generator(pairs, num_workers=num_workers)
        if tqdm is not None:
            data_gen = tqdm(data_gen, total=len(pairs), desc=f"{ds_name}", leave=True)

        for idx, case_data in data_gen:
            if case_data is None:
                skipped_empty += 1
                continue

            try:
                img_path = case_data["img_path"]
                gt_path = case_data["gt_path"]
                img_zyx = case_data["img_zyx"]
                gt_zyx = case_data["gt_zyx"]
                labels = case_data["labels"]
                spacing_zyx = case_data["spacing_zyx"]

                # choose cut slice on shortest axis
                primary_axis = choose_primary_axis(img_zyx.shape)
                cut_idx = largest_mask_slice_index(gt_zyx, primary_axis, label=None)

                # axis-0 view for splitting
                vol0 = np.moveaxis(img_zyx, primary_axis, 0)  # [S,H,W]
                gt0 = np.moveaxis(gt_zyx, primary_axis, 0)    # [S,H,W]
                N, H, W = vol0.shape
                n = int(cut_idx)
                v1_slices, v2_slices = build_two_videos_indices(N, n)
                gt_cut = gt0[n]  # [H,W]

                # prepare reusable video sources ONCE per case
                prepared = prepare_videos_once(predictor, vol0, v1_slices, v2_slices)
                total_voxels = int(gt_zyx.size)

                try:
                    # single-object inference for all labels, states reused
                    label_preds = predict_all_labels_reuse_state(
                        predictor=predictor,
                        prepared=prepared,
                        gt_cut_slice=gt_cut,
                        prompt_mode=prompt_mode,
                        labels=labels,
                    )

                    for lb in labels:
                        if lb not in label_preds:
                            skipped_prompt += 1
                            continue
                        pred0 = label_preds[lb]
                        if pred0 is None:
                            skipped_prompt += 1
                            continue

                        # restore to zyx orientation
                        pred_zyx_bin = np.moveaxis(pred0, 0, primary_axis)  # [Z,Y,X]

                        gt_lb = (gt_zyx == lb).astype(np.uint8)
                        pr_lb = (pred_zyx_bin > 0).astype(np.uint8)

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
            f"skip_empty_cases={skipped_empty}, skip_unpromptable_labels={skipped_prompt}, errors={skipped_errors}"
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
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers for data loading")
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
    print(f"Num workers   : {args.num_workers}")
    print("======================================\n")

    t0 = time.time()
    predictor = build_sam2_video_predictor(args.sam2_config, args.ckpt_path, device=args.device)

    save_test_record_3d_single_object(
        dataset_path=args.dataset_path,
        predictor=predictor,
        prompt_mode=args.prompt_mode,
        save_path=args.save_path,
        num_workers=args.num_workers,
    )

    print(f"\nSaved: {args.save_path}")
    print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

