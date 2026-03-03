#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_vis3d_twopass.py

3D SAM2 VideoPredictor evaluation with two-pass auto-context refinement.
Extends test_vis3d_multi.py: Pass 1 runs standard inference, then Pass 2
uses Pass 1's predictions to provide additional prompts / context.

=== Two-Pass Variants ===

  topk      (+2Pass-TopK)
    Select the K slices with highest Pass-1 confidence (object_score_logits).
    Add their Pass-1 masks as additional prompted frames in Pass 2.
    More anchors → smaller "unguided" gaps → better propagation.

  allconf   (+2Pass-AllConf)
    Add ALL Pass-1 slices above a minimum confidence threshold as prompts.
    More aggressive than TopK; essentially bootstraps from the full volume.

  revstyle  (+2Pass-RevStyle)
    Identify contiguous low-quality regions (below threshold).
    Bookend each region with its neighboring high-quality masks as prompts.
    Targets re-prompting exactly where Pass 1 is weak.

  none      (baseline, identical to test_vis3d_multi.py)

=== Usage ===

Baseline (no two-pass):
  python test_vis3d_twopass.py \
      --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
      --ckpt-path /path/to/checkpoint.pt \
      --dataset-path /path/to/Test_3d/ /path/to/ZeroShot/ \
      --save-path /path/to/results.pkl

TopK (K=5):
  python test_vis3d_twopass.py ... \
      --two-pass topk --two-pass-k 5 \
      --save-path /path/to/results_topk5.pkl

AllConf (confidence ≥ 0.6):
  python test_vis3d_twopass.py ... \
      --two-pass allconf --two-pass-conf-threshold 0.6 \
      --save-path /path/to/results_allconf.pkl

RevStyle (quality threshold 0.7):
  python test_vis3d_twopass.py ... \
      --two-pass revstyle --two-pass-conf-threshold 0.7 \
      --save-path /path/to/results_revstyle.pkl

Examples:
python test_vis3d_twopass.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/user/s4670484/Model_save/samri2_5x_3d_emb_256/sam2_emb3dloss_epoch_48.pth \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ /scratch/user/s4670484/Datasets/Zeroshot3D/ \
--save-path /scratch/user/s4670484/Eval_results/SAMRI2/sam2_5x_3dloss_256_epoch_48_topk5.pkl \
--num-workers 12 \
--two-pass topk --two-pass-k 5

python test_vis3d_twopass.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/user/s4670484/Model_save/samri2_5x_3d_emb_256/sam2_emb3dloss_epoch_48.pth \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ /scratch/user/s4670484/Datasets/Zeroshot3D/ \
--save-path /scratch/user/s4670484/Eval_results/SAMRI2/sam2_5x_3dloss_256_epoch_48_allconf.pkl \
--num-workers 12 \
--two-pass allconf --two-pass-conf-threshold 0.6

python test_vis3d_twopass.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/user/s4670484/Model_save/samri2_5x_3d_emb_256/sam2_emb3dloss_epoch_48.pth \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ /scratch/user/s4670484/Datasets/Zeroshot3D/ \
--save-path /scratch/user/s4670484/Eval_results/SAMRI2/sam2_5x_3dloss_256_epoch_48_revstyle.pkl \
--num-workers 12 \
--two-pass revstyle --two-pass-conf-threshold 0.7
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

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2_video_predictor

import contextlib


# ----------------------------
# Shared IO / utilities (verbatim from test_vis3d_multi.py)
# ----------------------------
@contextlib.contextmanager
def suppress_sam2_output(enabled: bool = True):
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


SUPPORTED_EXTS = {".nii", ".mhd", ".mha", ".nrrd", ".dcm", ".gz"}
PREFERRED_EXTS = [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]


def _is_nii_gz(p: Path) -> bool:
    return p.name.lower().endswith(".nii.gz")


def _has_med_ext_or_dir(p: Path) -> bool:
    if p.is_dir(): return True
    if _is_nii_gz(p): return True
    return p.suffix.lower() in SUPPORTED_EXTS


def read_3d_image(path: str) -> Tuple[np.ndarray, sitk.Image]:
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
            raise RuntimeError(f"Expected 3D, got {arr.shape}")
        return arr, img
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D, got {arr.shape} from {path}")
    return arr, img


def ensure_same_shape(img, msk, ip, mp):
    if img.shape != msk.shape:
        raise ValueError(f"Shape mismatch: {img.shape} vs {msk.shape}")


def is_img_candidate(p: Path) -> bool:
    return ("_img_" in p.name) and _has_med_ext_or_dir(p) and (p.is_file() or p.is_dir())


def replace_img_to_gt_name(name: str) -> str:
    return name.replace("_img_", "_gt_")


def try_find_gt_same_folder(img_path: Path) -> Optional[Path]:
    gt_name = replace_img_to_gt_name(img_path.name)
    gt_path = img_path.with_name(gt_name)
    if gt_path.exists(): return gt_path
    base = gt_name[:-7] if gt_name.lower().endswith(".nii.gz") else Path(gt_name).stem
    for ext in PREFERRED_EXTS:
        cand = img_path.with_name(base + ext)
        if cand.exists(): return cand
    cands = sorted(img_path.parent.glob(base + ".*"))
    return cands[0] if cands else None


def scan_pairs_same_folder(folder: str) -> List[Tuple[str, str]]:
    rootp = Path(folder)
    pairs: List[Tuple[str, str]] = []
    for p in rootp.rglob("*"):
        if not is_img_candidate(p): continue
        gt = try_find_gt_same_folder(p)
        if gt is None or not gt.exists(): continue
        pairs.append((str(p), str(gt)))
    pairs.sort()
    return pairs


def collect_test_folders(dataset_paths: List[str]) -> List[str]:
    test_folders: List[str] = []
    for dataset_path in dataset_paths:
        dataset_path = dataset_path.rstrip("/")
        ds_folders = [d for d in sorted(glob(dataset_path + "/*")) if os.path.isdir(d)]
        if not ds_folders:
            test_folders.append(dataset_path)
            continue
        for ds in ds_folders:
            cand = os.path.join(ds, "testing")
            test_folders.append(cand if os.path.isdir(cand) else ds)
    return test_folders


def is_image_empty(img_zyx, eps=1e-8):
    return float(np.max(np.abs(img_zyx))) <= eps


def get_labels_from_mask(gt_zyx):
    return sorted([int(x) for x in np.unique(gt_zyx) if x != 0])


def shortest_axes(shape):
    m = min(shape)
    return [i for i, s in enumerate(shape) if s == m]


def choose_primary_axis(shape) -> int:
    ax = shortest_axes(shape)
    return ax[0] if ax else 0


def largest_mask_slice_index(mask, axis, label=None):
    moved = np.moveaxis(mask, axis, 0)
    fg = (moved == label) if (label is not None) else (moved > 0)
    sums = fg.sum(axis=tuple(range(1, moved.ndim)))
    return int(np.argmax(sums))


def build_two_videos_indices(N, n):
    v1 = list(range(n, -1, -1))
    v2 = [n] + list(range(n + 1, N))
    return v1, v2


def normalize_to_uint8(slice2d):
    sl = slice2d.astype(np.float32)
    sl_min, sl_max = sl.min(), sl.max()
    if sl_max - sl_min < 1e-8:
        sl_max = sl_min + 1e-8
    return ((sl - sl_min) / (sl_max - sl_min) * 255.0).astype(np.uint8)


def make_pil_loader_from_volume(vol_shw, slice_indices):
    def loader_func(frame_idx):
        z = slice_indices[frame_idx]
        s2d = normalize_to_uint8(vol_shw[z])
        return Image.fromarray(np.stack([s2d, s2d, s2d], axis=-1))
    return loader_func, len(slice_indices)


def get_fast_tmp_root():
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return None


def write_video_frames_to_dir(vol_shw, slice_indices, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, z in enumerate(slice_indices):
        s2d = normalize_to_uint8(vol_shw[z])
        Image.fromarray(np.stack([s2d, s2d, s2d], axis=-1)).save(
            os.path.join(out_dir, f"{i:05d}.jpg"), quality=95)
    return out_dir


def init_state_compat(predictor, video_source, num_frames, prefer_streaming=True):
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
            pass
    if not isinstance(video_source, str):
        raise TypeError("Fallback requires directory path (str).")
    return predictor.init_state(video_path=video_source)


class PreparedVideos:
    def __init__(self, v1_source, v2_source, v1_len, v2_len, v1_slices, v2_slices, tmp_base, use_streaming):
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


def prepare_videos_once(predictor, vol0, v1_slices, v2_slices):
    loader1, num1 = make_pil_loader_from_volume(vol0, v1_slices)
    loader2, num2 = make_pil_loader_from_volume(vol0, v2_slices)
    try:
        with suppress_sam2_output(True):
            _ = init_state_compat(predictor, loader1, num1, prefer_streaming=True)
        return PreparedVideos(loader1, loader2, num1, num2, v1_slices, v2_slices, None, True)
    except Exception:
        fast_tmp = get_fast_tmp_root()
        tmp_base = tempfile.mkdtemp(prefix="sam2_3d_case_", dir=fast_tmp)
        v1_dir = os.path.join(tmp_base, "video1")
        v2_dir = os.path.join(tmp_base, "video2")
        write_video_frames_to_dir(vol0, v1_slices, v1_dir)
        write_video_frames_to_dir(vol0, v2_slices, v2_dir)
        return PreparedVideos(v1_dir, v2_dir, num1, num2, v1_slices, v2_slices, tmp_base, False)


def mask2d_to_bbox_xyxy(mask2d):
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0: return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def sample_positive_points(mask2d, num_pos=256):
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int32)
    k = min(num_pos, xs.size)
    idx = np.random.choice(xs.size, size=k, replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32), np.ones((k,), dtype=np.int32)


def add_prompts_for_all_labels(predictor, inference_state, prompt_mode, gt_cut_slice, labels, ann_frame_idx=0):
    added = []
    for lb in labels:
        obj_id = int(lb)
        mask_lb = (gt_cut_slice == obj_id).astype(np.uint8)
        if mask_lb.sum() == 0: continue
        if prompt_mode == "box":
            box = mask2d_to_bbox_xyxy(mask_lb)
            if box is None: continue
            try:
                predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=obj_id, box=box)
            except TypeError:
                predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=obj_id, points=None, labels=None, box=box)
            added.append(obj_id)
        elif prompt_mode == "mask":
            if hasattr(predictor, "add_new_mask"):
                predictor.add_new_mask(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=obj_id, mask=mask_lb)
                added.append(obj_id)
            else:
                pts, lbls = sample_positive_points(mask_lb)
                if pts.shape[0] == 0: continue
                predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=obj_id, points=pts, labels=lbls)
                added.append(obj_id)
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")
    return added


# ----------------------------
# Propagation
# ----------------------------
@torch.no_grad()
def propagate_video_multiobj(predictor, inference_state) -> Dict[int, Dict[int, np.ndarray]]:
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        fi = int(out_frame_idx)
        out[fi] = {}
        for i, oid in enumerate(out_obj_ids):
            logits = out_mask_logits[i]
            if hasattr(logits, "detach"): logits = logits.detach()
            m = (logits > 0.0)
            if hasattr(m, "cpu"): m = m.cpu().numpy()
            out[fi][int(oid)] = np.squeeze(m).astype(np.uint8)
    return out


@torch.no_grad()
def _prefill_features(predictor, inference_state: dict) -> None:
    """Pre-populate cached_features for every frame so features survive reset_state.

    Without prefill, SAM2's _get_image_feature uses a 1-entry dict cache that gets
    replaced on every cache miss.  After propagation only the last frame's features
    remain, so a subsequent reset_state + re-propagation would re-encode all frames.

    With prefill, the full dict {frame_idx: (image, backbone_out)} is written before
    propagation begins.  _get_image_feature finds every frame there (cache hit) and
    never replaces the dict, so features survive intact through reset_state.
    """
    images = inference_state["images"]
    device = inference_state["device"]
    cache: Dict[int, Any] = {}
    for i in range(len(images)):
        image = images[i].to(device).float().unsqueeze(0)
        backbone_out = predictor.forward_image(image)
        cache[i] = (image, backbone_out)
    inference_state["cached_features"] = cache


def _select_extra_frames(
    mode: str,
    frame_scores: Dict[int, float],
    k: int,
    threshold: float,
    excl: set,
) -> List[int]:
    """Dispatch to the appropriate frame-selection strategy."""
    if mode == "topk":
        return select_topk_frames(frame_scores, k, excl)
    elif mode == "allconf":
        return select_allconf_frames(frame_scores, threshold, excl)
    elif mode == "revstyle":
        return select_revstyle_frames(frame_scores, threshold, excl)
    return []


def merge_two_videos_to_semantic_volume(N, H, W, v1_slices, v2_slices, v1_pred, v2_pred, labels):
    pred = np.zeros((N, H, W), dtype=np.int32)
    written = set()

    def write_slice(z, obj_to_mask):
        for lb in labels:
            if int(lb) not in obj_to_mask: continue
            pred[z][obj_to_mask[int(lb)] > 0] = int(lb)

    for fidx, objm in v1_pred.items():
        z = v1_slices[fidx]
        write_slice(z, objm)
        written.add(z)
    for fidx, objm in v2_pred.items():
        z = v2_slices[fidx]
        if z in written: continue
        write_slice(z, objm)
        written.add(z)
    return pred


# ----------------------------
# Two-pass support functions
# ----------------------------
def extract_frame_scores(inference_state, added_labels: List[int]) -> Dict[int, Dict[int, float]]:
    """
    After propagation, read object_score_logits per frame per object from the inference state.
    Returns: {obj_id: {frame_idx: score_float}}
    The score is object_score_logits (higher = model more confident object is present & well-segmented).
    """
    scores: Dict[int, Dict[int, float]] = {}
    num_obj = len(inference_state.get("obj_idx_to_id", {}))
    for obj_idx in range(num_obj):
        obj_id = int(inference_state["obj_idx_to_id"][obj_idx])
        if obj_id not in [int(x) for x in added_labels]:
            continue
        obj_scores: Dict[int, float] = {}
        obj_out = inference_state["output_dict_per_obj"].get(obj_idx, {})
        for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
            for frame_idx, out in obj_out.get(storage_key, {}).items():
                score_t = out.get("object_score_logits", None)
                if score_t is None:
                    continue
                try:
                    val = float(score_t.item()) if hasattr(score_t, "item") else float(score_t)
                except Exception:
                    continue
                if frame_idx not in obj_scores:  # cond_frame takes priority (iterated first)
                    obj_scores[frame_idx] = val
        scores[obj_id] = obj_scores
    return scores


def select_topk_frames(
    frame_scores: Dict[int, float],
    k: int,
    exclude_frames: Optional[set] = None,
) -> List[int]:
    """Return indices of top-K frames by score, excluding specified frames."""
    excl = exclude_frames or {0}
    candidates = [(fi, sc) for fi, sc in frame_scores.items() if fi not in excl]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [fi for fi, _ in candidates[:k]]


def select_allconf_frames(
    frame_scores: Dict[int, float],
    threshold: float,
    exclude_frames: Optional[set] = None,
) -> List[int]:
    """Return all frames with sigmoid(score) >= threshold, excluding specified frames."""
    excl = exclude_frames or {0}
    selected = []
    for fi, sc in frame_scores.items():
        if fi in excl:
            continue
        conf = 1.0 / (1.0 + np.exp(-sc))  # sigmoid
        if conf >= threshold:
            selected.append(fi)
    return selected


def select_revstyle_frames(
    frame_scores: Dict[int, float],
    threshold: float,
    exclude_frames: Optional[set] = None,
) -> List[int]:
    """
    RevStyle selection: bookend low-quality regions.
    Identify contiguous runs of frames below the confidence threshold.
    For each such run, add the nearest high-quality frame on each side as prompts.
    Returns the set of high-quality anchor frame indices to use as prompts.
    """
    excl = exclude_frames or {0}
    # Sort frames by index (temporal order)
    all_frames = sorted(frame_scores.keys())
    if not all_frames:
        return []

    # Compute per-frame sigmoid confidence
    conf = {fi: 1.0 / (1.0 + np.exp(-frame_scores[fi])) for fi in all_frames}

    # Partition into high/low quality
    high_q = {fi for fi in all_frames if conf[fi] >= threshold and fi not in excl}
    low_q  = {fi for fi in all_frames if conf[fi] <  threshold and fi not in excl}

    if not low_q:
        return []

    # Find contiguous low-quality runs
    low_sorted = sorted(low_q)
    runs: List[List[int]] = []
    current_run = [low_sorted[0]]
    for fi in low_sorted[1:]:
        if fi == current_run[-1] + 1:
            current_run.append(fi)
        else:
            runs.append(current_run)
            current_run = [fi]
    runs.append(current_run)

    # For each run, find the nearest high-quality frame on each side
    anchors = set()
    high_sorted = sorted(high_q)
    for run in runs:
        run_min = run[0]
        run_max = run[-1]
        # Nearest high-quality frame before the run
        before = [fi for fi in high_sorted if fi < run_min]
        after  = [fi for fi in high_sorted if fi > run_max]
        if before:
            anchors.add(before[-1])  # closest before
        if after:
            anchors.add(after[0])    # closest after

    return list(anchors)


def add_pass2_mask_prompts(
    predictor,
    inference_state,
    added_labels: List[int],
    pass1_masks: Dict[int, Dict[int, np.ndarray]],  # frame_idx -> {obj_id -> binary mask}
    extra_frame_indices: Dict[int, List[int]],       # obj_id -> [frame_idx, ...]
) -> None:
    """
    Add Pass-1 mask predictions as additional prompts to a fresh inference state.
    `extra_frame_indices`: per-object list of frame indices to re-prompt.
    """
    for obj_id in added_labels:
        oid = int(obj_id)
        frames_to_add = extra_frame_indices.get(oid, [])
        for frame_idx in frames_to_add:
            frame_masks = pass1_masks.get(frame_idx, {})
            mask = frame_masks.get(oid, None)
            if mask is None:
                continue
            if mask.sum() == 0:
                continue  # skip empty masks (object not visible in this slice)
            try:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=oid,
                    mask=mask.astype(bool),
                )
            except Exception:
                pass  # if frame index is out of range for this video, skip


# ----------------------------
# Main prediction function
# ----------------------------
def predict_multi_object_semantic(
    predictor,
    prepared: PreparedVideos,
    gt_cut_slice: np.ndarray,
    prompt_mode: str,
    labels: List[int],
    two_pass_mode: str = "none",
    two_pass_k: int = 5,
    two_pass_conf_threshold: float = 0.7,
) -> Tuple[Optional[np.ndarray], List[int]]:
    """
    Multi-object inference with optional two-pass refinement.

    two_pass_mode:
        "none"     - standard single-pass (identical to test_vis3d_multi.py)
        "topk"     - re-prompt Pass 2 with top-K confidence frames from Pass 1
        "allconf"  - re-prompt Pass 2 with all frames above confidence threshold
        "revstyle" - re-prompt Pass 2 with anchors bookending low-quality regions
    """
    do_two_pass = two_pass_mode in ("topk", "allconf", "revstyle")
    excl_frame0 = {0}  # frame 0 is always the original prompt; never re-prompt it

    # ---- VIDEO 1: Pass 1 then (optionally) Pass 2 in the same state ----
    # Prefill caches all frame features before Pass 1 so they survive reset_state,
    # allowing Pass 2 to reuse them without re-running the image encoder.
    with suppress_sam2_output(True):
        st1 = init_state_compat(predictor, prepared.v1_source, prepared.v1_len,
                                prefer_streaming=prepared.use_streaming)
        # Prefill before adding prompts: prompt processing also calls _get_image_feature
        # on frame 0, so prefilling first avoids encoding frame 0 twice.
        if do_two_pass:
            _prefill_features(predictor, st1)
        added = add_prompts_for_all_labels(predictor, st1, prompt_mode, gt_cut_slice, labels, ann_frame_idx=0)
        if not added:
            return None, []
        p1_v1 = propagate_video_multiobj(predictor, st1)

    if do_two_pass:
        scores_v1 = extract_frame_scores(st1, added)
        extra_v1: Dict[int, List[int]] = {
            int(oid): _select_extra_frames(
                two_pass_mode, scores_v1.get(int(oid), {}),
                two_pass_k, two_pass_conf_threshold, excl_frame0,
            )
            for oid in added
        }
        with suppress_sam2_output(True):
            predictor.reset_state(st1)          # clears objects; keeps cached_features
            _ = add_prompts_for_all_labels(predictor, st1, prompt_mode, gt_cut_slice, added, ann_frame_idx=0)
            add_pass2_mask_prompts(predictor, st1, added, p1_v1, extra_v1)
            final_v1 = propagate_video_multiobj(predictor, st1)
    else:
        final_v1 = p1_v1

    del st1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- VIDEO 2: Pass 1 then (optionally) Pass 2 in the same state ----
    with suppress_sam2_output(True):
        st2 = init_state_compat(predictor, prepared.v2_source, prepared.v2_len,
                                prefer_streaming=prepared.use_streaming)
        if do_two_pass:
            _prefill_features(predictor, st2)
        _ = add_prompts_for_all_labels(predictor, st2, prompt_mode, gt_cut_slice, added, ann_frame_idx=0)
        p1_v2 = propagate_video_multiobj(predictor, st2)

    if do_two_pass:
        scores_v2 = extract_frame_scores(st2, added)
        extra_v2: Dict[int, List[int]] = {
            int(oid): _select_extra_frames(
                two_pass_mode, scores_v2.get(int(oid), {}),
                two_pass_k, two_pass_conf_threshold, excl_frame0,
            )
            for oid in added
        }
        with suppress_sam2_output(True):
            predictor.reset_state(st2)          # clears objects; keeps cached_features
            _ = add_prompts_for_all_labels(predictor, st2, prompt_mode, gt_cut_slice, added, ann_frame_idx=0)
            add_pass2_mask_prompts(predictor, st2, added, p1_v2, extra_v2)
            final_v2 = propagate_video_multiobj(predictor, st2)
    else:
        final_v2 = p1_v2

    del st2

    N = len(prepared.v1_slices) + (len(prepared.v2_slices) - 1)
    H, W = gt_cut_slice.shape
    pred0 = merge_two_videos_to_semantic_volume(
        N=N, H=H, W=W,
        v1_slices=prepared.v1_slices, v2_slices=prepared.v2_slices,
        v1_pred=final_v1, v2_pred=final_v2, labels=added,
    )
    return pred0, added


# ----------------------------
# Metrics
# ----------------------------
def dice_3d(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 0).astype(np.uint8)
    inter = (pred & gt).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + gt.sum(dtype=np.float64)
    if denom == 0: return 1.0
    return float(2.0 * inter / denom)


def hd_msd_3d_sitk(pred, gt, spacing_zyx):
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin   = (gt   > 0).astype(np.uint8)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0: return 0.0, 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:  return float("inf"), float("inf")

    pred_img = sitk.GetImageFromArray(pred_bin)
    gt_img   = sitk.GetImageFromArray(gt_bin)
    sx, sy, sz = float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])
    pred_img.SetSpacing((sx, sy, sz))
    gt_img.SetSpacing((sx, sy, sz))

    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(pred_img, gt_img)
    hd = float(hd_filter.GetHausdorffDistance())

    pred_surface = sitk.LabelContour(pred_img)
    gt_surface   = sitk.LabelContour(gt_img)
    pred_dt = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_img, squaredDistance=False, useImageSpacing=True))
    gt_dt   = sitk.Abs(sitk.SignedMaurerDistanceMap(gt_img,   squaredDistance=False, useImageSpacing=True))

    pred_to_gt = sitk.GetArrayFromImage(gt_dt   * sitk.Cast(pred_surface, sitk.sitkFloat32))
    gt_to_pred = sitk.GetArrayFromImage(pred_dt * sitk.Cast(gt_surface,   sitk.sitkFloat32))

    pred_vals = pred_to_gt[pred_to_gt > 0]
    gt_vals   = gt_to_pred[gt_to_pred > 0]
    if pred_vals.size == 0 or gt_vals.size == 0:
        msd = float("inf")
    else:
        msd = float((pred_vals.mean() + gt_vals.mean()) / 2.0)
    return hd, msd


# ----------------------------
# Parallel data loading
# ----------------------------
def load_3d_case(pair):
    img_path, gt_path = pair
    try:
        img_zyx, img_sitk = read_3d_image(img_path)
        gt_zyx, _ = read_3d_image(gt_path)
        ensure_same_shape(img_zyx, gt_zyx, img_path, gt_path)
        if is_image_empty(img_zyx): return None
        labels = get_labels_from_mask(gt_zyx)
        if not labels: return None
        spacing_xyz = img_sitk.GetSpacing()
        spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
        return {"img_path": img_path, "gt_path": gt_path, "img_zyx": img_zyx,
                "gt_zyx": gt_zyx, "labels": labels, "spacing_zyx": spacing_zyx}
    except Exception:
        return None


def prefetch_data_generator(pairs, num_workers=4, prefetch_size=8):
    if num_workers <= 1:
        for i, pair in enumerate(pairs):
            yield i, load_3d_case(pair)
        return
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        next_submit = next_yield = 0
        n = len(pairs)
        while next_submit < n and len(futures) < prefetch_size:
            futures[next_submit] = executor.submit(load_3d_case, pairs[next_submit])
            next_submit += 1
        while next_yield < n:
            if next_yield in futures:
                result = futures.pop(next_yield).result()
                yield next_yield, result
                next_yield += 1
                if next_submit < n:
                    futures[next_submit] = executor.submit(load_3d_case, pairs[next_submit])
                    next_submit += 1
            else:
                time.sleep(0.001)


# ----------------------------
# Main record function
# ----------------------------
def save_test_record_3d_twopass(
    dataset_paths: List[str],
    predictor,
    prompt_mode: str,
    save_path: str,
    num_workers: int = 4,
    two_pass_mode: str = "none",
    two_pass_k: int = 5,
    two_pass_conf_threshold: float = 0.7,
) -> Dict[str, List[Dict]]:

    final_record: Dict[str, List[Dict]] = {}
    test_folders = collect_test_folders(dataset_paths)

    for test_folder in test_folders:
        tf = Path(test_folder).resolve()
        ds_name = tf.parent.name if tf.name == "testing" else tf.name

        pairs = scan_pairs_same_folder(test_folder)
        if not pairs:
            final_record[ds_name] = []
            continue

        ds_rows: List[Dict] = []
        skipped_empty = skipped_unpromptable = skipped_errors = kept_labels = 0

        data_gen = prefetch_data_generator(pairs, num_workers=num_workers)
        if tqdm is not None:
            data_gen = tqdm(data_gen, total=len(pairs), desc=f"{ds_name}", leave=True)

        for idx, case_data in data_gen:
            if case_data is None:
                skipped_empty += 1
                continue
            try:
                img_path    = case_data["img_path"]
                gt_path     = case_data["gt_path"]
                img_zyx     = case_data["img_zyx"]
                gt_zyx      = case_data["gt_zyx"]
                labels      = case_data["labels"]
                spacing_zyx = case_data["spacing_zyx"]

                primary_axis = choose_primary_axis(img_zyx.shape)
                cut_idx      = largest_mask_slice_index(gt_zyx, primary_axis, label=None)

                vol0 = np.moveaxis(img_zyx, primary_axis, 0)
                gt0  = np.moveaxis(gt_zyx,  primary_axis, 0)
                N, H, W = vol0.shape
                n = int(cut_idx)
                v1_slices, v2_slices = build_two_videos_indices(N, n)
                gt_cut = gt0[n]

                prepared     = prepare_videos_once(predictor, vol0, v1_slices, v2_slices)
                total_voxels = int(gt_zyx.size)

                try:
                    pred0_sem, added = predict_multi_object_semantic(
                        predictor=predictor,
                        prepared=prepared,
                        gt_cut_slice=gt_cut,
                        prompt_mode=prompt_mode,
                        labels=labels,
                        two_pass_mode=two_pass_mode,
                        two_pass_k=two_pass_k,
                        two_pass_conf_threshold=two_pass_conf_threshold,
                    )
                    if pred0_sem is None or not added:
                        skipped_empty += 1
                        continue

                    pred_zyx_sem = np.moveaxis(pred0_sem, 0, primary_axis)

                    for lb in labels:
                        if lb not in added:
                            skipped_unpromptable += 1
                            continue
                        gt_lb = (gt_zyx == lb).astype(np.uint8)
                        pr_lb = (pred_zyx_sem == lb).astype(np.uint8)

                        dsc = dice_3d(pr_lb, gt_lb)
                        hd, msd = hd_msd_3d_sitk(pr_lb, gt_lb, spacing_zyx=spacing_zyx)

                        ds_rows.append({
                            "img_name":         img_path,
                            "mask_name":        gt_path,
                            "label":            int(lb),
                            "voxel_count":      int(gt_lb.sum()),
                            "voxel_percentage": int(gt_lb.sum()) / float(total_voxels),
                            "dice":             float(dsc),
                            "hd":               float(hd),
                            "msd":              float(msd),
                            "cut_axis":         int(primary_axis),
                            "cut_idx":          int(cut_idx),
                            "two_pass_mode":    two_pass_mode,
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
            f"skip_empty={skipped_empty}, skip_unpromptable={skipped_unpromptable}, errors={skipped_errors}"
        )

    save_path = str(save_path)
    Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(final_record, f)
    return final_record


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="3D SAM2 evaluation with two-pass auto-context refinement")
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--ckpt-path",   type=str, required=True)
    parser.add_argument("--dataset-path", type=str, nargs=2, required=True,
                        metavar=("TEST_DATA", "ZEROSHOT_RESULTS"))
    parser.add_argument("--save-path",   type=str, required=True)
    parser.add_argument("--prompt-mode", type=str, choices=["box", "mask"], default="box")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=0)

    # Two-pass arguments
    parser.add_argument("--two-pass", type=str, default="none",
                        choices=["none", "topk", "allconf", "revstyle"],
                        help="Two-pass mode: none | topk | allconf | revstyle (default: none)")
    parser.add_argument("--two-pass-k", type=int, default=5,
                        help="Number of top-K frames for topk mode (default: 5)")
    parser.add_argument("--two-pass-conf-threshold", type=float, default=0.7,
                        help="Confidence threshold for allconf/revstyle modes "
                             "(sigmoid of object_score_logits, default: 0.7)")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    print("\n======================================")
    print(f"SAM2 config   : {args.sam2_config}")
    print(f"Checkpoint    : {args.ckpt_path}")
    print(f"Test data     : {args.dataset_path[0]}")
    print(f"Zero-shot data: {args.dataset_path[1]}")
    print(f"Save path     : {args.save_path}")
    print(f"Prompt mode   : {args.prompt_mode}")
    print(f"Device        : {args.device}")
    print(f"Num workers   : {args.num_workers}")
    print(f"Two-pass mode : {args.two_pass}")
    if args.two_pass == "topk":
        print(f"  K            = {args.two_pass_k}")
    if args.two_pass in ("allconf", "revstyle"):
        print(f"  conf thresh  = {args.two_pass_conf_threshold}")
    print("======================================\n")

    t0 = time.time()
    predictor = build_sam2_video_predictor(args.sam2_config, args.ckpt_path, device=args.device)

    save_test_record_3d_twopass(
        dataset_paths=args.dataset_path,
        predictor=predictor,
        prompt_mode=args.prompt_mode,
        save_path=args.save_path,
        num_workers=args.num_workers,
        two_pass_mode=args.two_pass,
        two_pass_k=args.two_pass_k,
        two_pass_conf_threshold=args.two_pass_conf_threshold,
    )

    print(f"\nSaved: {args.save_path}")
    print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
