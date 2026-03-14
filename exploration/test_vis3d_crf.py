#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_vis3d_crf.py

3D SAM2 VideoPredictor evaluation with Dense CRF post-processing.
Extends test_vis3d_multi.py by adding optional 3D/2D Dense CRF refinement
on SAM2's soft probability outputs AFTER propagation, BEFORE metric computation.

CRF backend priority:
  1. SimpleCRF (pip install SimpleCRF) -- true 3D CRF
  2. pydensecrf (pip install pydensecrf) -- 2D slice-by-slice fallback
  3. Simple sigmoid thresholding -- if neither is available

=== Usage ===

Without CRF (identical to test_vis3d_multi.py):
  python test_vis3d_crf.py \
      --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
      --ckpt-path /path/to/checkpoint.pt \
      --dataset-path /path/to/Test_3d/ /path/to/ZeroShot/ \
      --save-path /path/to/results.pkl \
      --prompt-mode box

With CRF (default hyperparameters):
  python test_vis3d_crf.py \
      --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
      --ckpt-path /path/to/checkpoint.pt \
      --dataset-path /path/to/Test_3d/ /path/to/ZeroShot/ \
      --save-path /path/to/results_crf.pkl \
      --prompt-mode box \
      --use-crf

With CRF (custom hyperparameters):
  python test_vis3d_crf.py ... \
      --use-crf \
      --crf-sigma-spatial 3.0 \
      --crf-sigma-intensity 0.1 \
      --crf-sigma-smooth 1.0 \
      --crf-w-bilateral 5.0 \
      --crf-w-smooth 3.0 \
      --crf-iterations 5

Examples:
python test_vis3d_crf.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/user/s4670484/Model_save/samri2_5x_3d_emb_256/sam2_emb3dloss_epoch_48.pth \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ /scratch/user/s4670484/Datasets/Zeroshot3D/ \
--save-path /scratch/user/s4670484/Eval_results/SAMRI2/sam2_5x_3dloss_256_epoch_48_crf.pkl \
--num-workers 12 \
--use-crf
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
# CRF backend detection
# ----------------------------
_CRF_BACKEND = None  # cached after first call


def _detect_crf_backend() -> str:
    global _CRF_BACKEND
    if _CRF_BACKEND is not None:
        return _CRF_BACKEND
    try:
        import SimpleCRF  # noqa: F401
        _CRF_BACKEND = "simplecrf"
        return _CRF_BACKEND
    except ImportError:
        pass
    try:
        import pydensecrf  # noqa: F401
        _CRF_BACKEND = "pydensecrf"
        return _CRF_BACKEND
    except ImportError:
        pass
    _CRF_BACKEND = "none"
    return _CRF_BACKEND


def apply_3d_crf(
    image_3d: np.ndarray,
    prob_3d: np.ndarray,
    sigma_spatial: float,
    sigma_intensity: float,
    sigma_smooth: float,
    w_bilateral: float,
    w_smooth: float,
    num_iterations: int,
) -> np.ndarray:
    """
    Apply Dense CRF refinement on a 3D probability volume.

    Args:
        image_3d:        (D, H, W) float32, normalized MRI intensity in [0, 1]
        prob_3d:         (D, H, W) float32, sigmoid foreground probability in [0, 1]
        sigma_spatial:   spatial sigma for the bilateral (appearance) kernel (voxels)
        sigma_intensity: intensity sigma for the bilateral kernel (fraction of [0,1])
        sigma_smooth:    spatial sigma for the smoothness (Gaussian) kernel (voxels)
        w_bilateral:     weight of the bilateral pairwise term
        w_smooth:        weight of the smoothness pairwise term
        num_iterations:  number of CRF mean-field iterations

    Returns:
        refined: (D, H, W) uint8 binary mask (0/1)
    """
    eps = 1e-5
    D, H, W = prob_3d.shape
    backend = _detect_crf_backend()

    # --- SimpleCRF 3D ---
    if backend == "simplecrf":
        try:
            from SimpleCRF import denseCRF3D
            img_input = (image_3d * 255.0).clip(0, 255).astype(np.uint8)
            img_input = img_input[..., np.newaxis]  # (D, H, W, 1)
            fg_p = np.clip(prob_3d, eps, 1.0 - eps).astype(np.float32)
            prob_input = np.stack([np.float32(1.0) - fg_p, fg_p], axis=-1)  # (D, H, W, 2) float32
            result = denseCRF3D(
                img_input, prob_input,
                sigma_spatial, sigma_intensity,
                sigma_smooth, num_iterations,
            )
            return result.astype(np.uint8)
        except Exception as e:
            print(f"[CRF] SimpleCRF failed ({e}), falling back to pydensecrf")
            backend = "pydensecrf"

    # --- pydensecrf 2D slice-by-slice ---
    if backend == "pydensecrf":
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax

            result = np.zeros((D, H, W), dtype=np.uint8)
            # Scale intensity sigma from [0,1] range back to uint8 range used by pydensecrf
            srgb = max(1.0, sigma_intensity * 255.0)
            for z in range(D):
                img_slice = (image_3d[z] * 255.0).clip(0, 255).astype(np.uint8)
                img_rgb = np.ascontiguousarray(np.stack([img_slice, img_slice, img_slice], axis=-1))  # (H, W, 3)

                fg_p = np.clip(prob_3d[z], eps, 1.0 - eps).astype(np.float32)
                bg_p = 1.0 - fg_p
                probs = np.stack([bg_p, fg_p], axis=0)  # (2, H, W)

                d = dcrf.DenseCRF2D(W, H, 2)
                U = unary_from_softmax(probs)
                d.setUnaryEnergy(U)
                d.addPairwiseGaussian(sxy=sigma_smooth, compat=w_smooth)
                d.addPairwiseBilateral(
                    sxy=sigma_spatial,
                    srgb=srgb,
                    rgbim=img_rgb,
                    compat=w_bilateral,
                )
                Q = d.inference(num_iterations)
                result[z] = np.argmax(Q, axis=0).reshape(H, W).astype(np.uint8)
            return result
        except Exception as e:
            print(f"[CRF] pydensecrf failed ({e}), falling back to thresholding")

    # --- fallback: simple sigmoid threshold ---
    return (prob_3d > 0.5).astype(np.uint8)


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
    if p.is_dir():
        return True
    if _is_nii_gz(p):
        return True
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
            raise RuntimeError(f"Expected 3D DICOM series, got shape {arr.shape}")
        return arr, img
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D image, got shape {arr.shape} from {path}")
    return arr, img


def ensure_same_shape(img, msk, img_path, msk_path):
    if img.shape != msk.shape:
        raise ValueError(f"Shape mismatch:\n  image: {img.shape} ({img_path})\n  mask : {msk.shape} ({msk_path})")


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


def is_image_empty(img_zyx: np.ndarray, eps: float = 1e-8) -> bool:
    return float(np.max(np.abs(img_zyx))) <= eps


def get_labels_from_mask(gt_zyx: np.ndarray) -> List[int]:
    labels = [int(x) for x in np.unique(gt_zyx)]
    return sorted([x for x in labels if x != 0])


def shortest_axes(shape):
    m = min(shape)
    return [i for i, s in enumerate(shape) if s == m]


def choose_primary_axis(shape) -> int:
    ax = shortest_axes(shape)
    return ax[0] if ax else 0


def largest_mask_slice_index(mask: np.ndarray, axis: int, label=None) -> int:
    moved = np.moveaxis(mask, axis, 0)
    fg = (moved == label) if (label is not None) else (moved > 0)
    sums = fg.sum(axis=tuple(range(1, moved.ndim)))
    return int(np.argmax(sums))


def build_two_videos_indices(N: int, n: int) -> Tuple[List[int], List[int]]:
    v1 = list(range(n, -1, -1))
    v2 = [n] + list(range(n + 1, N))
    return v1, v2


def normalize_to_uint8(slice2d: np.ndarray) -> np.ndarray:
    sl = slice2d.astype(np.float32)
    sl_min, sl_max = sl.min(), sl.max()
    if sl_max - sl_min < 1e-8:
        sl_max = sl_min + 1e-8
    sl = (sl - sl_min) / (sl_max - sl_min)
    return (sl * 255.0).astype(np.uint8)


def normalize_to_float(vol: np.ndarray) -> np.ndarray:
    """Normalize a volume to [0, 1] float32 using plain min-max scaling."""
    v = vol.astype(np.float32)
    v_min, v_max = v.min(), v.max()
    if v_max - v_min < 1e-8:
        v_max = v_min + 1e-8
    return (v - v_min) / (v_max - v_min)


def make_pil_loader_from_volume(vol_shw, slice_indices):
    def loader_func(frame_idx: int) -> Image.Image:
        z = slice_indices[frame_idx]
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        return Image.fromarray(rgb)
    return loader_func, len(slice_indices)


def get_fast_tmp_root():
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return None


def write_video_frames_to_dir(vol_shw, slice_indices, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, z in enumerate(slice_indices):
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"{i:05d}.jpg"), quality=95)
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
        raise TypeError("Fallback init_state requires a directory path (str) as video_source.")
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
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def sample_positive_points(mask2d, num_pos=256):
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int32)
    k = min(num_pos, xs.size)
    idx = np.random.choice(xs.size, size=k, replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    lbls = np.ones((k,), dtype=np.int32)
    return pts, lbls


def add_prompts_for_all_labels(predictor, inference_state, prompt_mode, gt_cut_slice, labels, ann_frame_idx=0):
    added = []
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
                if pts.shape[0] == 0:
                    continue
                predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=obj_id, points=pts, labels=lbls)
                added.append(obj_id)
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")
    return added


# ----------------------------
# Propagation: binary masks (standard path)
# ----------------------------
@torch.no_grad()
def propagate_video_multiobj(predictor, inference_state) -> Dict[int, Dict[int, np.ndarray]]:
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        fi = int(out_frame_idx)
        out[fi] = {}
        for i, oid in enumerate(out_obj_ids):
            logits = out_mask_logits[i]
            if hasattr(logits, "detach"):
                logits = logits.detach()
            m = (logits > 0.0)
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            out[fi][int(oid)] = np.squeeze(m).astype(np.uint8)
    return out


# ----------------------------
# Propagation: raw logits (CRF path)
# ----------------------------
@torch.no_grad()
def propagate_video_multiobj_with_logits(predictor, inference_state) -> Dict[int, Dict[int, np.ndarray]]:
    """Returns: frame_idx -> {obj_id -> float32 logit array (H, W)}"""
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        fi = int(out_frame_idx)
        out[fi] = {}
        for i, oid in enumerate(out_obj_ids):
            logits = out_mask_logits[i]
            if hasattr(logits, "detach"):
                logits = logits.detach().cpu().numpy()
            elif hasattr(logits, "numpy"):
                logits = logits.numpy()
            out[fi][int(oid)] = np.squeeze(logits).astype(np.float32)
    return out


# ----------------------------
# Merge / assemble volumes
# ----------------------------
def merge_two_videos_to_semantic_volume(N, H, W, v1_slices, v2_slices, v1_pred, v2_pred, labels):
    pred = np.zeros((N, H, W), dtype=np.int32)
    written = set()

    def write_slice(z, obj_to_mask):
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


def build_logit_volumes(
    N: int, H: int, W: int,
    v1_slices: List[int], v2_slices: List[int],
    v1_pred: Dict[int, Dict[int, np.ndarray]],
    v2_pred: Dict[int, Dict[int, np.ndarray]],
    labels: List[int],
) -> Dict[int, np.ndarray]:
    """
    Build per-label 3D logit volumes from two-video float predictions.
    Returns: {label_id: (N, H, W) float32 logit volume}
    Unvisited slices default to -10.0 (strong background).
    """
    logit_vols: Dict[int, np.ndarray] = {}
    for lb in labels:
        oid = int(lb)
        vol = np.full((N, H, W), -10.0, dtype=np.float32)
        written: set = set()

        for fidx, objm in v1_pred.items():
            z = v1_slices[fidx]
            if oid in objm:
                s = objm[oid]
                if s.shape != (H, W):
                    from scipy.ndimage import zoom
                    s = zoom(s, (H / s.shape[0], W / s.shape[1]), order=1)
                vol[z] = s
                written.add(z)

        for fidx, objm in v2_pred.items():
            z = v2_slices[fidx]
            if z in written:
                continue
            if oid in objm:
                s = objm[oid]
                if s.shape != (H, W):
                    from scipy.ndimage import zoom
                    s = zoom(s, (H / s.shape[0], W / s.shape[1]), order=1)
                vol[z] = s
                written.add(z)

        logit_vols[lb] = vol
    return logit_vols


# ----------------------------
# Main prediction function
# ----------------------------
def predict_multi_object_semantic(
    predictor,
    prepared: PreparedVideos,
    gt_cut_slice: np.ndarray,
    prompt_mode: str,
    labels: List[int],
    vol0: Optional[np.ndarray] = None,   # (N, H, W) image; required when use_crf=True
    use_crf: bool = False,
    crf_params: Optional[Dict] = None,
) -> Tuple[Optional[np.ndarray], List[int]]:
    """
    Multi-object inference; returns (pred0_semantic [S,H,W] int32, added_labels).
    When use_crf=True, SAM2 logits are post-processed with Dense CRF before thresholding.
    """
    # video1
    with suppress_sam2_output(True):
        st1 = init_state_compat(predictor, prepared.v1_source, prepared.v1_len, prefer_streaming=prepared.use_streaming)
        added = add_prompts_for_all_labels(predictor, st1, prompt_mode, gt_cut_slice, labels, ann_frame_idx=0)
        if not added:
            return None, []
        if use_crf:
            pred1 = propagate_video_multiobj_with_logits(predictor, st1)
        else:
            pred1 = propagate_video_multiobj(predictor, st1)

    # video2
    with suppress_sam2_output(True):
        st2 = init_state_compat(predictor, prepared.v2_source, prepared.v2_len, prefer_streaming=prepared.use_streaming)
        _ = add_prompts_for_all_labels(predictor, st2, prompt_mode, gt_cut_slice, added, ann_frame_idx=0)
        if use_crf:
            pred2 = propagate_video_multiobj_with_logits(predictor, st2)
        else:
            pred2 = propagate_video_multiobj(predictor, st2)

    N = len(prepared.v1_slices) + (len(prepared.v2_slices) - 1)
    H, W = gt_cut_slice.shape

    if not use_crf:
        pred0 = merge_two_videos_to_semantic_volume(
            N=N, H=H, W=W,
            v1_slices=prepared.v1_slices, v2_slices=prepared.v2_slices,
            v1_pred=pred1, v2_pred=pred2, labels=added,
        )
        return pred0, added

    # CRF path: build logit volumes, apply CRF per label, assemble semantic volume
    logit_vols = build_logit_volumes(N, H, W, prepared.v1_slices, prepared.v2_slices, pred1, pred2, added)

    # Normalize the image to [0,1] for CRF appearance kernel
    img_float = normalize_to_float(vol0)  # (N, H, W)

    p = crf_params or {}
    sigma_spatial   = p.get("sigma_spatial",   3.0)
    sigma_intensity = p.get("sigma_intensity", 0.1)
    sigma_smooth    = p.get("sigma_smooth",    1.0)
    w_bilateral     = p.get("w_bilateral",     5.0)
    w_smooth        = p.get("w_smooth",        3.0)
    num_iterations  = p.get("num_iterations",  5)

    pred0 = np.zeros((N, H, W), dtype=np.int32)
    for lb in added:
        logit_vol = logit_vols[lb]           # (N, H, W)
        from scipy.special import expit
        prob_vol = expit(logit_vol.astype(np.float64)).astype(np.float32)  # stable sigmoid

        refined = apply_3d_crf(
            img_float, prob_vol,
            sigma_spatial=sigma_spatial,
            sigma_intensity=sigma_intensity,
            sigma_smooth=sigma_smooth,
            w_bilateral=w_bilateral,
            w_smooth=w_smooth,
            num_iterations=num_iterations,
        )
        pred0[refined > 0] = int(lb)

    return pred0, added


# ----------------------------
# Metrics
# ----------------------------
def dice_3d(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 0).astype(np.uint8)
    inter = (pred & gt).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + gt.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def hd_msd_3d_sitk(pred, gt, spacing_zyx):
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin   = (gt   > 0).astype(np.uint8)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0, 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("inf"), float("inf")

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
        if is_image_empty(img_zyx):
            return None
        labels = get_labels_from_mask(gt_zyx)
        if not labels:
            return None
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
        next_submit = 0
        next_yield = 0
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
def save_test_record_3d_crf(
    dataset_paths: List[str],
    predictor,
    prompt_mode: str,
    save_path: str,
    num_workers: int = 4,
    use_crf: bool = False,
    crf_params: Optional[Dict] = None,
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
                img_path   = case_data["img_path"]
                gt_path    = case_data["gt_path"]
                img_zyx    = case_data["img_zyx"]
                gt_zyx     = case_data["gt_zyx"]
                labels     = case_data["labels"]
                spacing_zyx = case_data["spacing_zyx"]

                primary_axis = choose_primary_axis(img_zyx.shape)
                cut_idx      = largest_mask_slice_index(gt_zyx, primary_axis, label=None)

                vol0 = np.moveaxis(img_zyx, primary_axis, 0)   # [S, H, W]
                gt0  = np.moveaxis(gt_zyx,  primary_axis, 0)
                N, H, W = vol0.shape
                n = int(cut_idx)
                v1_slices, v2_slices = build_two_videos_indices(N, n)
                gt_cut = gt0[n]

                prepared    = prepare_videos_once(predictor, vol0, v1_slices, v2_slices)
                total_voxels = int(gt_zyx.size)

                try:
                    pred0_sem, added = predict_multi_object_semantic(
                        predictor=predictor,
                        prepared=prepared,
                        gt_cut_slice=gt_cut,
                        prompt_mode=prompt_mode,
                        labels=labels,
                        vol0=vol0 if use_crf else None,
                        use_crf=use_crf,
                        crf_params=crf_params,
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
                            "img_name": img_path,
                            "mask_name": gt_path,
                            "label": int(lb),
                            "voxel_count": int(gt_lb.sum()),
                            "voxel_percentage": int(gt_lb.sum()) / float(total_voxels),
                            "dice": float(dsc),
                            "hd": float(hd),
                            "msd": float(msd),
                            "cut_axis": int(primary_axis),
                            "cut_idx": int(cut_idx),
                            "crf_applied": use_crf,
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
    parser = argparse.ArgumentParser(description="3D SAM2 evaluation with optional Dense CRF post-processing")
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, nargs=2, required=True,
                        metavar=("TEST_DATA", "ZEROSHOT_RESULTS"))
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--prompt-mode", type=str, choices=["box", "mask"], default="box")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    # CRF arguments
    parser.add_argument("--use-crf", action="store_true", help="Enable Dense CRF post-processing")
    parser.add_argument("--crf-sigma-spatial",   type=float, default=3.0,
                        help="Bilateral kernel spatial sigma in voxels (default: 3.0)")
    parser.add_argument("--crf-sigma-intensity", type=float, default=0.1,
                        help="Bilateral kernel intensity sigma, fraction of [0,1] (default: 0.1)")
    parser.add_argument("--crf-sigma-smooth",    type=float, default=1.0,
                        help="Smoothness kernel spatial sigma in voxels (default: 1.0)")
    parser.add_argument("--crf-w-bilateral",     type=float, default=5.0,
                        help="Weight of bilateral pairwise term (default: 5.0)")
    parser.add_argument("--crf-w-smooth",        type=float, default=3.0,
                        help="Weight of smoothness pairwise term (default: 3.0)")
    parser.add_argument("--crf-iterations",      type=int,   default=5,
                        help="Number of CRF mean-field iterations (default: 5)")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    crf_params = {
        "sigma_spatial":   args.crf_sigma_spatial,
        "sigma_intensity": args.crf_sigma_intensity,
        "sigma_smooth":    args.crf_sigma_smooth,
        "w_bilateral":     args.crf_w_bilateral,
        "w_smooth":        args.crf_w_smooth,
        "num_iterations":  args.crf_iterations,
    }

    if args.use_crf:
        backend = _detect_crf_backend()
        print(f"\n[CRF] Backend: {backend}")

    print("\n======================================")
    print(f"SAM2 config   : {args.sam2_config}")
    print(f"Checkpoint    : {args.ckpt_path}")
    print(f"Test data     : {args.dataset_path[0]}")
    print(f"Zero-shot data: {args.dataset_path[1]}")
    print(f"Save path     : {args.save_path}")
    print(f"Prompt mode   : {args.prompt_mode}")
    print(f"Device        : {args.device}")
    print(f"Num workers   : {args.num_workers}")
    print(f"CRF enabled   : {args.use_crf}")
    if args.use_crf:
        print(f"  sigma_spatial   = {crf_params['sigma_spatial']}")
        print(f"  sigma_intensity = {crf_params['sigma_intensity']}")
        print(f"  sigma_smooth    = {crf_params['sigma_smooth']}")
        print(f"  w_bilateral     = {crf_params['w_bilateral']}")
        print(f"  w_smooth        = {crf_params['w_smooth']}")
        print(f"  iterations      = {crf_params['num_iterations']}")
    print("======================================\n")

    t0 = time.time()
    predictor = build_sam2_video_predictor(args.sam2_config, args.ckpt_path, device=args.device)

    save_test_record_3d_crf(
        dataset_paths=args.dataset_path,
        predictor=predictor,
        prompt_mode=args.prompt_mode,
        save_path=args.save_path,
        num_workers=args.num_workers,
        use_crf=args.use_crf,
        crf_params=crf_params,
    )

    print(f"\nSaved: {args.save_path}")
    print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
