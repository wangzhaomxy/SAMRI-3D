#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3d_inference.py

Batch 3D inference with SAM2 VideoPredictor (NO evaluation).
- Scans dataset folders automatically (same logic as test_vis3d.py)
- Image + GT in same folder with "_img_" and "_gt_" keys
- Uses GT ONLY to find cut slice (largest foreground slice) and to build prompt:
    prompt-mode = "box": bbox from cut-slice GT for each label
    prompt-mode = "mask": cut-slice GT mask prompt (if add_new_mask exists, else dense points)

- Splits volume into two “videos” around cut slice, runs SAM2 video propagation,
  merges back to a 3D semantic prediction volume.

- Saves predicted 3D masks to target folders (mirrors input structure):
    <save-root>/<dataset>/<relative_path>/<basename with _img_ replaced>_pred.nii.gz
  Uses SimpleITK to write NIfTI (.nii.gz). Spacing/origin/direction copied from input image.

NOTE:
This script is intentionally “almost the same” as test_vis3d.py but without metrics.

CLI:
python 3d_inference.py \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --ckpt-path /path/to/sam2.1_hiera_small.pt \
  --dataset-path /path/to/Datasets/Test_3d/ \
  --save-root /path/to/Infer_results/SAM2/ \
  --prompt-mode box \
  --device cuda
  
python 3d_inference.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /home/s4670484/Desktop/Scratch/samri/pretrained_ckpt/sam2.1_hiera_small.pt \
--dataset-path /home/s4670484/Desktop/Scratch/samri/Datasets/Test_3d/ \
--save-root /home/s4670484/Desktop/Scratch/samri/Infer_results/SAM2/ \
--prompt-mode box \
--device cuda
  
Available SAM2 configs/checkpoints:
    sam2.1_hiera_s.yaml     sam2.1_hiera_small.pt
    sam2.1_hiera_b.yaml     sam2.1_hiera_base.pt
    sam2.1_hiera_l+.yaml    sam2.1_hiera_large_plus.pt
    sam2.1_hiera_m.yaml     sam2.1_hiera_medium.pt
"""

from __future__ import annotations

import os
import sys
import time
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

# Prefer local sam2/ folder
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2_video_predictor


# ----------------------------
# IO: 3D volume reader/writer
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


def write_pred_mask_nifti(pred_zyx: np.ndarray, ref_img: sitk.Image, out_path: str) -> None:
    """
    Save semantic label volume as .nii.gz with metadata copied from reference image.
    pred_zyx should be [z,y,x] int32.
    """
    out_img = sitk.GetImageFromArray(pred_zyx.astype(np.int16))  # int16 enough for labels
    out_img.CopyInformation(ref_img)
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, out_path)


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
            print(f"[Warn] GT not found for image: {p}")
            continue
        pairs.append((str(p), str(gt)))
    pairs.sort()
    return pairs


def collect_test_folders(dataset_path: str) -> List[str]:
    """
    dataset-path contains multiple dataset folders.
    For each dataset folder, prefer <ds>/testing if exists; else scan <ds>.
    If dataset_path itself is already a single dataset folder, it will still work.
    """
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
# Frame adapters: streaming callable (preferred) + jpg fallback
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
    vol_shw: np.ndarray,
    slice_indices: List[int],
) -> Tuple[Callable[[int], Image.Image], int, int, int]:
    _, H, W = vol_shw.shape

    def loader_func(frame_idx: int) -> Image.Image:
        z = slice_indices[frame_idx]
        s2d = normalize_to_uint8(vol_shw[z])
        rgb = np.stack([s2d, s2d, s2d], axis=-1)
        return Image.fromarray(rgb)

    return loader_func, len(slice_indices), H, W


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
        except TypeError as e:
            print(f"[Warn] Streaming init_state failed: {e}")

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
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
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
    gt_cut = gt0[n]

    loader1, num1, _, _ = make_pil_loader_from_volume(vol0, v1_slices)
    loader2, num2, _, _ = make_pil_loader_from_volume(vol0, v2_slices)

    fast_tmp = get_fast_tmp_root()
    tmp_base = tempfile.mkdtemp(prefix="sam2_3d_", dir=fast_tmp)

    try:
        # video1
        try:
            st1 = init_state_compat(predictor, loader1, num1, prefer_streaming=True)
        except Exception:
            v1_dir = os.path.join(tmp_base, "video1")
            write_video_frames_to_dir(vol0, v1_slices, v1_dir)
            st1 = init_state_compat(predictor, v1_dir, num1, prefer_streaming=False)

        added1 = add_prompts_for_all_labels(predictor, st1, prompt_mode, gt_cut, labels, ann_frame_idx=0)
        if not added1:
            raise RuntimeError("No labels present on cut slice for prompting.")
        pred1 = propagate_video_multiobj(predictor, st1)

        # video2
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
        shutil.rmtree(tmp_base, ignore_errors=True)


# ----------------------------
# Output path builder
# ----------------------------
def strip_known_suffixes(name: str) -> str:
    """
    Turn foo.nii.gz -> foo, foo.nii -> foo, foo.mha -> foo, etc.
    """
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def build_out_path(
    save_root: str,
    dataset_name: str,
    test_folder: str,
    img_path: str,
) -> str:
    """
    Save to:
      <save-root>/<dataset-name>/<relative-to-test_folder>/<img_stem_with_key_replaced>_pred.nii.gz
    """
    save_root = save_root.rstrip("/")
    imgp = Path(img_path).resolve()
    base_dir = Path(test_folder).resolve()
    rel_parent = imgp.parent.relative_to(base_dir) if str(imgp.parent).startswith(str(base_dir)) else Path(".")
    stem = strip_known_suffixes(imgp.name)
    stem = stem.replace("_img_", "_pred_")  # keep key visible
    out_dir = Path(save_root) / dataset_name / rel_parent
    out_name = stem + ".nii.gz"
    return str(out_dir / out_name)


# ----------------------------
# Main inference loop
# ----------------------------
def run_inference_on_dataset_path(
    predictor,
    dataset_path: str,
    save_root: str,
    prompt_mode: str,
) -> None:
    test_folders = collect_test_folders(dataset_path)

    for test_folder in test_folders:
        tf = Path(test_folder).resolve()
        ds_name = tf.parent.name if tf.name == "testing" else tf.name

        print(f"\n[Dataset] {ds_name}")
        print(f"[Scan  ] {test_folder}")

        pairs = scan_pairs_same_folder(test_folder)
        if not pairs:
            print(f"[Warn] No *_img_/*_gt_* pairs under: {test_folder}")
            continue

        for (img_path, gt_path) in pairs:
            print(f"\n[Case ] {img_path}")
            img_zyx, img_sitk = read_3d_image(img_path)
            gt_zyx, _ = read_3d_image(gt_path)
            ensure_same_shape(img_zyx, gt_zyx, img_path, gt_path)

            labels = sorted([int(x) for x in np.unique(gt_zyx) if int(x) != 0])
            if not labels:
                print("[Warn] No foreground labels in GT; skip.")
                continue

            shape = img_zyx.shape  # (Z,Y,X)
            primary_axis = choose_primary_axis(shape)
            cut_idx = largest_mask_slice_index(gt_zyx, primary_axis, label=None)

            # axis-0 view
            vol0 = np.moveaxis(img_zyx, primary_axis, 0)
            gt0 = np.moveaxis(gt_zyx, primary_axis, 0)

            pred0 = run_two_videos_predict(
                predictor=predictor,
                vol0=vol0,
                gt0=gt0,
                labels=labels,
                prompt_mode=prompt_mode,
                cut_idx=cut_idx,
            )
            pred_zyx = np.moveaxis(pred0, 0, primary_axis)

            out_path = build_out_path(save_root, ds_name, test_folder, img_path)
            write_pred_mask_nifti(pred_zyx, img_sitk, out_path)
            print(f"[Save ] {out_path}")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--save-root", type=str, required=True, help="Root folder to save predicted masks.")
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
    print(f"Save root     : {args.save_root}")
    print(f"Prompt mode   : {args.prompt_mode}")
    print(f"Device        : {args.device}")
    print("======================================\n")

    t0 = time.time()
    predictor = build_sam2_video_predictor(args.sam2_config, args.ckpt_path, device=args.device)

    run_inference_on_dataset_path(
        predictor=predictor,
        dataset_path=args.dataset_path,
        save_root=args.save_root,
        prompt_mode=args.prompt_mode,
    )

    print(f"\nDone. Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
