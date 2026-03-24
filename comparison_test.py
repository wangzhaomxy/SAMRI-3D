#!/usr/bin/env python3
"""
comparison_test.py

Evaluate comparison 3D segmentation models on SAMRI2-format NIfTI test data.

Design
------
1. Data loading  : imported directly from test_vis3d_multi.py (shared functions)
2. Per-model shim: converts loaded (img_zyx, gt_zyx) → each model's required input format
3. Inference     : calls the original inference code from the comparison/ subfolder
4. Evaluation    : same dice_3d / hd_msd_3d_sitk as test_vis3d_multi.py
5. Save format   : same {ds_name: [row_dicts]} structure as test_vis3d_multi.py

Implemented runners
-------------------
  sam_med3d     SAM-Med3D    — medim.create_model; validate_paired_img_gt (torchio
                               resampling + CropOrPad 128³); GT-guided point clicks.
                               Install: pip install medim
  medsam2       MedSAM2      — build_sam2_video_predictor_npz (MedSAM2's sam2 fork,
                               swapped via _swap_sam2); (S,3,512,512) tensor input;
                               box prompt at max-fg slice; forward + reverse propagation.
  medical_sam2  Medical-SAM2 — build_sam2_video_predictor (sam2_train package, no
                               namespace conflict); (S,3,512,512) tensor input;
                               bbox prompt; val_init_state + propagate_in_video.
  samed2        SAMed-2      — MedicalSegmenter (SAMed-2's sam2 fork, swapped via
                               _swap_sam2); 2D per-slice 1024×1024 input; box prompt
                               from GT per slice; aggregated to 3D volume.

Stub (format incompatible)
--------------------------
  ma_sam        MA-SAM       — requires Synapse-format pickle data (2Dimage_* / 2Dmask_*);
                               NIfTI input not supported by its test.py. Add a
                               NIfTI→Synapse-pickle conversion wrapper to enable.

sam2 namespace management
--------------------------
MedSAM2 and SAMed-2 each ship their own sam2 fork with symbols absent from SAMRI2's sam2.
Both use _swap_sam2() during model construction to temporarily replace sys.modules['sam2*']
with the correct fork, then restore SAMRI2's sam2 on exit.

Weights directory layout
------------------------
Each subfolder must contain exactly one checkpoint file (.pt/.pth/.pkl/.bin/.safetensors).
The filename does not matter — the script picks up whichever file is present.

<weights_dir>/
├── sam_med3d/      <one .pth>   (https://huggingface.co/blueyo0/SAM-Med3D)
├── MedSAM2/        <one .pt>    (https://huggingface.co/wanglab/MedSAM2)
├── medica_sam2/    <one .pt>    (https://huggingface.co/jiayuanz3/MedSAM2_pretrain)
│                                (override with --medical-sam2-ckpt /explicit/path)
├── samed2/         <one .pth>   (SAMed-2 GitHub page)
│                                (override with --samed2-ckpt /explicit/path)
└── masam/
    ├── sam_vit_h_4b8939.pth    (Meta SAM ViT-H base)
    └── ma_sam_btcv.pth         (MA-SAM Google Drive — see comparison/MA-SAM-main/README.md)

Usage
-----
# Single model
python comparison_test.py \\
    --model medsam2 \\
    --dataset-path /path/to/Test_3d/ /path/to/ZeroShot/ \\
    --weights-dir /path/to/weights/ \\
    --save-path results/medsam2.pkl

# All models sequentially (saves per-model .pkl and a combined .pkl)
python comparison_test.py \\
    --model all \\
    --dataset-path /path/to/Test_3d/ /path/to/ZeroShot/ \\
    --weights-dir /path/to/weights/ \\
    --save-path results/comparison.pkl \\
    --device cuda:0

# Debug: limit to 2 cases per dataset
python comparison_test.py \
--model all \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ /scratch/user/s4670484/Datasets/Zeroshot3D/ \
--weights-dir /scratch/user/s4670484/comparison_ckpt/ \
--save-path /scratch/user/s4670484/Eval_results/comparison/samed2_debug.pkl \
--max-cases 2

example:
python comparison_test.py \
--model medsam2 \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ /scratch/user/s4670484/Datasets/Zeroshot3D/ \
--weights-dir /scratch/user/s4670484/comparison_ckpt/ \
--save-path /scratch/user/s4670484/Eval_results/comparison/medsam2.pkl
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import pickle
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image as PILImage
try:
    from tqdm import tqdm as _tqdm
    def _progress(it, **kw):
        # Write to the real stderr so the bar is visible in the terminal
        # but does NOT appear in the Tee-captured log file.
        return _tqdm(it, file=sys.__stderr__, dynamic_ncols=True, **kw)
except ImportError:
    import contextlib

    class _NoBar:
        """No-op stand-in when tqdm is not installed."""
        def __init__(self, it, **_): self._it = it
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def __iter__(self): return iter(self._it)
        def set_postfix(self, **_): pass

    def _progress(it, **kw):
        return _NoBar(it, **kw)

# PIL Pillow 10+ compatibility: Image.BILINEAR / NEAREST removed → Resampling enum
try:
    _BILINEAR = PILImage.Resampling.BILINEAR
    _NEAREST  = PILImage.Resampling.NEAREST
except AttributeError:          # Pillow < 9.1
    _BILINEAR = PILImage.BILINEAR  # type: ignore[attr-defined]
    _NEAREST  = PILImage.NEAREST   # type: ignore[attr-defined]

# ─── Import shared functions from test_vis3d_multi.py ────────────────────────
_SAMRI2_ROOT = Path(__file__).resolve().parent
if str(_SAMRI2_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAMRI2_ROOT))

from test_vis3d_multi import (
    read_3d_image,
    get_labels_from_mask,
    choose_primary_axis,
    normalize_to_uint8,
    dice_3d,
    hd_msd_3d_sitk,
    scan_pairs_same_folder,
    collect_test_folders,
)


# ─── sam2 namespace swap helper ───────────────────────────────────────────────
# Importing test_vis3d_multi above loads SAMRI2's sam2 into sys.modules['sam2'].
# MedSAM2 and SAMed-2 each ship their own sam2 fork with different symbols.
# _swap_sam2() temporarily replaces sys.modules['sam2'] during model construction
# so their build functions resolve correctly.  After construction the original
# sam2 is restored; the built model objects hold direct class references that
# remain valid regardless of what sys.modules['sam2'] points to.

def _is_sam2_or_tv(k: str) -> bool:
    return k == 'sam2' or k.startswith('sam2.') or k == 'torchvision' or k.startswith('torchvision.')


@contextlib.contextmanager
def _swap_sam2(new_sam2_parent: Path):
    """
    Context manager: replace sys.modules['sam2*'] with the sam2 found under
    new_sam2_parent, then restore on exit.

    Also saves/restores torchvision* so that a failed import inside the context
    (e.g. missing CUDA extension on the first call) does not leave a partially
    initialised torchvision in sys.modules, which would cause a circular-import
    error on every subsequent call.
    """
    saved = {k: v for k, v in sys.modules.items() if _is_sam2_or_tv(k)}
    for k in list(saved):
        del sys.modules[k]
    new_p = str(new_sam2_parent)
    was_in = new_p in sys.path
    if not was_in:
        sys.path.insert(0, new_p)
    try:
        yield
    finally:
        for k in [k for k in sys.modules if _is_sam2_or_tv(k)]:
            del sys.modules[k]
        if not was_in and new_p in sys.path:
            sys.path.remove(new_p)
        sys.modules.update(saved)


# ─── Shared shim utilities ────────────────────────────────────────────────────

def mask2d_to_bbox_xyxy(mask2d: np.ndarray) -> Optional[np.ndarray]:
    """[x0, y0, x1, y1] float32 bounding box, or None if mask is empty."""
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _imagenet_normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Normalize (S, 3, H, W) float32 tensor [0,1] with ImageNet stats."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    return (t - mean) / std


def _vol_to_rgb_tensor(vol_shw: np.ndarray, frame_size: int,
                        device: str) -> torch.Tensor:
    """
    Convert [S, H, W] float32 volume to an ImageNet-normalized
    (S, 3, frame_size, frame_size) CUDA tensor for SAM2-style video predictor input.
    """
    S = vol_shw.shape[0]
    out = np.zeros((S, 3, frame_size, frame_size), dtype=np.float32)
    for i in range(S):
        s2d = normalize_to_uint8(vol_shw[i])
        pil = PILImage.fromarray(s2d).convert("RGB").resize(
            (frame_size, frame_size), _BILINEAR)
        arr = np.array(pil, dtype=np.float32) / 255.0   # (H, W, 3)
        out[i] = arr.transpose(2, 0, 1)                  # (3, H, W)
    t = torch.from_numpy(out)
    t = _imagenet_normalize_tensor(t)
    return t.to(device)


def _resize_mask_to_hw(mask_pred: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize a boolean mask (any resolution) back to (H, W)."""
    return np.array(
        PILImage.fromarray(mask_pred.astype(np.uint8) * 255).resize(
            (W, H), _NEAREST)
    ) > 127


def _find_checkpoint(folder: Path, model_name: str) -> Path:
    """Return the single checkpoint file inside *folder*.

    Raises FileNotFoundError with a clear message if the folder is missing or empty.
    """
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{model_name} checkpoint folder not found: {folder}"
        )
    ckpts = [p for p in folder.iterdir()
             if p.suffix in (".pt", ".pth", ".pkl", ".bin", ".safetensors")
             and p.is_file()]
    if not ckpts:
        raise FileNotFoundError(
            f"{model_name}: no checkpoint file found in {folder}\n"
            f"  (looked for .pt / .pth / .pkl / .bin / .safetensors)"
        )
    if len(ckpts) > 1:
        # Deterministic fallback: alphabetically first
        ckpts.sort()
    return ckpts[0]


# ─── SAM-Med3D runner ─────────────────────────────────────────────────────────

def run_sam_med3d(
    img_path: str, gt_path: str, label_id: int,
    img_zyx: np.ndarray, gt_zyx: np.ndarray,
    spacing_zyx: Tuple, weights_dir: str, device: str,
    comparison_dir: str, tmp_dir: str, **_,
) -> np.ndarray:
    """
    SAM-Med3D inference via the original validate_paired_img_gt helper.

    Shim: write a per-label binary GT NIfTI → call validate_paired_img_gt →
          read output back as [Z,Y,X] binary array.

    Prompt: point clicks (GT-guided) — SAM-Med3D's native mode.
    The model handles all resampling/cropping internally (torchio + tio.CropOrPad).
    """
    try:
        import medim
    except ImportError:
        raise RuntimeError(
            "medim not installed.  pip install medim\n"
            "or see https://github.com/uni-medical/medim"
        )

    comp = Path(comparison_dir)
    sam_med3d_root = comp / "SAM-Med3D-main"
    if str(sam_med3d_root) not in sys.path:
        sys.path.insert(0, str(sam_med3d_root))
    from utils.infer_utils import validate_paired_img_gt  # noqa: PLC0415

    ckpt_path = _find_checkpoint(Path(weights_dir) / "sam_med3d", "SAM-Med3D")

    # Cache model across labels/cases
    cache_key = str(ckpt_path)
    if not hasattr(run_sam_med3d, "_cache") or run_sam_med3d._cache[0] != cache_key:
        run_sam_med3d._cache = (
            cache_key,
            medim.create_model("SAM-Med3D", pretrained=True,
                                checkpoint_path=str(ckpt_path)),
        )
    model = run_sam_med3d._cache[1]

    # Write binary GT NIfTI for this label (preserving original image metadata)
    gt_binary = (gt_zyx == label_id).astype(np.uint8)
    gt_sitk = sitk.ReadImage(gt_path)
    gt_bin_img = sitk.GetImageFromArray(gt_binary)
    gt_bin_img.CopyInformation(gt_sitk)
    tmp_gt   = str(Path(tmp_dir) / f"gt_label{label_id}.nii.gz")
    tmp_pred = str(Path(tmp_dir) / f"pred_label{label_id}.nii.gz")
    sitk.WriteImage(gt_bin_img, tmp_gt)

    # Call original inference function (takes file paths, handles resampling internally)
    validate_paired_img_gt(model, img_path, tmp_gt, tmp_pred, num_clicks=1)

    pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(tmp_pred)).astype(bool)
    return pred_arr  # [Z, Y, X]


# ─── MedSAM2 runner ───────────────────────────────────────────────────────────

def run_medsam2(
    img_path: str, gt_path: str, label_id: int,
    img_zyx: np.ndarray, gt_zyx: np.ndarray,
    spacing_zyx: Tuple, weights_dir: str, device: str,
    comparison_dir: str, tmp_dir: str, **_,
) -> np.ndarray:
    """
    MedSAM2 inference using build_sam2_video_predictor_npz.

    Shim: img_zyx → primary axis → [S,H,W] → resize 512×512 per slice
          → RGB → ImageNet normalize → (S, 3, 512, 512) CUDA tensor.

    Inference: predictor.init_state(tensor, H_orig, W_orig);
               box prompt at max-fg slice (scaled to 512);
               forward propagate + reset + reverse propagate (original pattern).

    Note: MedSAM2 ships its own sam2 fork.  _swap_sam2() is used during model
    construction only so that build_sam2_video_predictor_npz resolves correctly.
    """
    comp = Path(comparison_dir)
    medsam2_root = comp / "MedSAM2-main"

    ckpt_path = _find_checkpoint(Path(weights_dir) / "MedSAM2", "MedSAM2")

    cfg_candidates = [
        medsam2_root / "sam2" / "configs" / "sam2.1_hiera_t512.yaml",
        medsam2_root / "sam2" / "configs" / "sam2.1_hiera_t.yaml",
        medsam2_root / "configs" / "sam2.1_hiera_t512.yaml",
    ]
    cfg_path = next((c for c in cfg_candidates if c.exists()), None)
    if cfg_path is None:
        # Fallback: search recursively for any 512 config
        hits = list(medsam2_root.rglob("sam2.1_hiera_t512.yaml"))
        if not hits:
            hits = list(medsam2_root.rglob("sam2.1_hiera_t*.yaml"))
        cfg_path = hits[0] if hits else None
    if cfg_path is None:
        raise FileNotFoundError(
            "MedSAM2 config YAML not found. Searched:\n"
            + "\n".join(f"  {c}" for c in cfg_candidates)
            + f"\n  (also tried rglob under {medsam2_root})"
        )

    # Build predictor using MedSAM2's own sam2 fork (temporarily swap sys.modules)
    cache_key = str(ckpt_path)
    if not hasattr(run_medsam2, "_cache") or run_medsam2._cache[0] != cache_key:
        if str(medsam2_root) not in sys.path:
            sys.path.insert(0, str(medsam2_root))
        with _swap_sam2(medsam2_root):
            from hydra.core.global_hydra import GlobalHydra  # noqa: PLC0415
            GlobalHydra.instance().clear()
            from sam2.build_sam import build_sam2_video_predictor_npz  # noqa: PLC0415
            # Hydra search path is pkg://sam2 (the sam2 package dir).
            # The config name must be relative to that dir, e.g. "configs/sam2.1_hiera_t512".
            # Using cfg_path.stem ("sam2.1_hiera_t512") fails because the YAML
            # lives in sam2/configs/, not at the package root.
            sam2_pkg_dir = medsam2_root / "sam2"
            try:
                cfg_name = str(cfg_path.relative_to(sam2_pkg_dir).with_suffix(""))
            except ValueError:
                cfg_name = cfg_path.stem  # fallback for unusual layouts
            predictor = build_sam2_video_predictor_npz(
                cfg_name, str(ckpt_path), device=device)
        run_medsam2._cache = (cache_key, predictor)
    predictor = run_medsam2._cache[1]

    FRAME_SIZE = 512

    primary_axis = choose_primary_axis(img_zyx.shape)
    vol_shw = np.moveaxis(img_zyx, primary_axis, 0)  # [S, H, W]
    gt_shw  = np.moveaxis(gt_zyx,  primary_axis, 0)
    S, H, W = vol_shw.shape

    img_tensor = _vol_to_rgb_tensor(vol_shw, FRAME_SIZE, device)  # (S,3,512,512)

    fg = (gt_shw == label_id).sum(axis=(1, 2))
    if fg.max() == 0:
        return np.zeros(img_zyx.shape, dtype=bool)
    key_s = int(np.argmax(fg))
    box = mask2d_to_bbox_xyxy((gt_shw[key_s] == label_id).astype(np.uint8))
    if box is None:
        return np.zeros(img_zyx.shape, dtype=bool)
    box_512 = box * np.array([FRAME_SIZE / W, FRAME_SIZE / H,
                               FRAME_SIZE / W, FRAME_SIZE / H], dtype=np.float32)

    pred_shw = np.zeros((S, H, W), dtype=bool)
    autocast_device = "cuda" if "cuda" in device else "cpu"
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
        # Forward pass
        state = predictor.init_state(img_tensor, H, W)
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=key_s, obj_id=1, box=box_512)
        for fidx, _, logits in predictor.propagate_in_video(state):
            # logits[0] shape: (1, H512, W512)
            m = (logits[0] > 0.0).cpu().numpy()[0]
            pred_shw[fidx] = _resize_mask_to_hw(m, H, W)
        predictor.reset_state(state)

        # Reverse pass (original MedSAM2 pattern: re-init + re-prompt + reverse=True)
        state = predictor.init_state(img_tensor, H, W)
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=key_s, obj_id=1, box=box_512)
        for fidx, _, logits in predictor.propagate_in_video(state, reverse=True):
            m = (logits[0] > 0.0).cpu().numpy()[0]
            pred_shw[fidx] |= _resize_mask_to_hw(m, H, W)
        predictor.reset_state(state)

    pred_zyx = np.moveaxis(pred_shw, 0, primary_axis)
    return pred_zyx


# ─── Medical-SAM2 runner ──────────────────────────────────────────────────────

def run_medical_sam2(
    img_path: str, gt_path: str, label_id: int,
    img_zyx: np.ndarray, gt_zyx: np.ndarray,
    spacing_zyx: Tuple, weights_dir: str, device: str,
    comparison_dir: str, tmp_dir: str,
    medical_sam2_ckpt: str = None, **_,
) -> np.ndarray:
    """
    Medical-SAM2 inference using sam2_train video predictor.

    Shim: img_zyx → primary axis → [S,H,W] → resize 512×512 per slice
          → RGB → ImageNet normalize → (S, 3, 512, 512) CUDA tensor.

    Inference: net.val_init_state(imgs_tensor); bbox prompt at max-fg frame;
               net.propagate_in_video().

    Note: Medical-SAM2 ships sam2_train (not sam2), so no namespace conflict.
    """
    comp = Path(comparison_dir)
    med_sam2_root = comp / "Medical-SAM2-main"
    if str(med_sam2_root) not in sys.path:
        sys.path.insert(0, str(med_sam2_root))

    ckpt_path = (Path(medical_sam2_ckpt) if medical_sam2_ckpt else
                 _find_checkpoint(Path(weights_dir) / "medica_sam2", "Medical-SAM2"))

    # Medical-SAM2 ships its own sam2_train package (no conflict with sam2)
    from sam2_train.build_sam import build_sam2_video_predictor  # noqa: PLC0415

    cfg_path = med_sam2_root / "sam2_train" / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml"
    if not cfg_path.exists():
        cfgs = sorted((med_sam2_root / "sam2_train").rglob("*.yaml"))
        if not cfgs:
            raise FileNotFoundError(
                f"Medical-SAM2 config YAML not found under {med_sam2_root}")
        cfg_path = cfgs[0]

    cache_key = str(ckpt_path)
    if not hasattr(run_medical_sam2, "_cache") or run_medical_sam2._cache[0] != cache_key:
        from hydra.core.global_hydra import GlobalHydra  # noqa: PLC0415
        GlobalHydra.instance().clear()
        net = build_sam2_video_predictor(str(cfg_path), str(ckpt_path), device=device)
        net.eval()
        run_medical_sam2._cache = (cache_key, net)
    net = run_medical_sam2._cache[1]

    FRAME_SIZE = 512

    primary_axis = choose_primary_axis(img_zyx.shape)
    vol_shw = np.moveaxis(img_zyx, primary_axis, 0)  # [S, H, W]
    gt_shw  = np.moveaxis(gt_zyx,  primary_axis, 0)
    S, H, W = vol_shw.shape

    img_tensor = _vol_to_rgb_tensor(vol_shw, FRAME_SIZE, device)  # (S,3,512,512)

    fg = (gt_shw == label_id).sum(axis=(1, 2))
    if fg.max() == 0:
        return np.zeros(img_zyx.shape, dtype=bool)
    key_s = int(np.argmax(fg))
    box = mask2d_to_bbox_xyxy((gt_shw[key_s] == label_id).astype(np.uint8))
    if box is None:
        return np.zeros(img_zyx.shape, dtype=bool)
    box_512 = torch.tensor(
        box * np.array([FRAME_SIZE / W, FRAME_SIZE / H,
                        FRAME_SIZE / W, FRAME_SIZE / H], dtype=np.float32),
        device=device,
    ).unsqueeze(0)  # (1, 4)

    pred_shw = np.zeros((S, H, W), dtype=bool)
    with torch.no_grad():
        state = net.val_init_state(imgs_tensor=img_tensor)
        net.train_add_new_bbox(
            inference_state=state,
            frame_idx=key_s,
            obj_id=label_id,
            bbox=box_512,
            clear_old_points=False,
        )
        for fidx, out_obj_ids, out_mask_logits in net.propagate_in_video(
                state, start_frame_idx=0):
            idx = next(
                (j for j, o in enumerate(out_obj_ids) if int(o) == label_id), None)
            if idx is None:
                continue
            # out_mask_logits[idx] shape: (1, H512, W512)
            m = (out_mask_logits[idx] > 0.0).cpu().numpy()[0]
            pred_shw[fidx] = _resize_mask_to_hw(m, H, W)
        net.reset_state(state)

    pred_zyx = np.moveaxis(pred_shw, 0, primary_axis)
    return pred_zyx


# ─── SAMed-2 runner (2D per-slice) ───────────────────────────────────────────

def run_samed2(
    img_path: str, gt_path: str, label_id: int,
    img_zyx: np.ndarray, gt_zyx: np.ndarray,
    spacing_zyx: Tuple, weights_dir: str, device: str,
    comparison_dir: str, tmp_dir: str,
    samed2_ckpt: str = None, **_,
) -> np.ndarray:
    """
    SAMed-2 inference using MedicalSegmenter.predict() (2D, box-prompted).

    Shim: for each slice along the primary axis, normalize to uint8 PIL image;
          derive box from GT label on that slice (scaled to 1024×1024);
          call MedicalSegmenter.predict(pil_image, box) → mask.
    Aggregate per-slice masks into [Z,Y,X] volume.

    Note: SAMed-2 has its own sam2 fork.  _swap_sam2() is used during model
    construction so that sam2.function and sam2.build_sam resolve correctly.
    """
    comp = Path(comparison_dir)
    samed2_root = comp / "SAMed-2"
    if str(samed2_root) not in sys.path:
        sys.path.insert(0, str(samed2_root))

    ckpt_path = (Path(samed2_ckpt) if samed2_ckpt else
                 _find_checkpoint(Path(weights_dir) / "samed2", "SAMed-2"))

    cache_key = str(ckpt_path)
    if not hasattr(run_samed2, "_cache") or run_samed2._cache[0] != cache_key:
        with _swap_sam2(samed2_root):
            from predict import MedicalSegmenter  # noqa: PLC0415
            segmenter = MedicalSegmenter(
                model_type='samed2',
                checkpoint_path=str(ckpt_path),
                device=device,
            )
        run_samed2._cache = (cache_key, segmenter)
    segmenter = run_samed2._cache[1]

    IMG_SIZE = 1024

    primary_axis = choose_primary_axis(img_zyx.shape)
    vol_shw = np.moveaxis(img_zyx, primary_axis, 0)  # [S, H, W]
    gt_shw  = np.moveaxis(gt_zyx,  primary_axis, 0)
    S, H, W = vol_shw.shape

    fg = (gt_shw == label_id).sum(axis=(1, 2))
    if fg.max() == 0:
        return np.zeros(img_zyx.shape, dtype=bool)

    pred_shw = np.zeros((S, H, W), dtype=bool)
    for s in range(S):
        s2d = normalize_to_uint8(vol_shw[s])
        pil = PILImage.fromarray(s2d).convert("RGB")

        # Box from GT label on this slice, scaled to SAMed-2's 1024×1024 input
        box_hw = mask2d_to_bbox_xyxy((gt_shw[s] == label_id).astype(np.uint8))
        if box_hw is not None:
            box_1024 = (box_hw * np.array(
                [IMG_SIZE / W, IMG_SIZE / H, IMG_SIZE / W, IMG_SIZE / H]
            )).tolist()
        else:
            # Empty slice — use SAMed-2's default center-80% box
            margin = IMG_SIZE * 0.1
            box_1024 = [margin, margin, IMG_SIZE - margin, IMG_SIZE - margin]

        result    = segmenter.predict(pil, box=box_1024)
        mask_1024 = result['mask']  # (1024, 1024) bool

        # Resize prediction back to original (H, W)
        pred_shw[s] = _resize_mask_to_hw(mask_1024, H, W)

    pred_zyx = np.moveaxis(pred_shw, 0, primary_axis)
    return pred_zyx


# ─── MA-SAM stub ──────────────────────────────────────────────────────────────

def run_ma_sam(
    img_path: str, gt_path: str, label_id: int,
    img_zyx: np.ndarray, gt_zyx: np.ndarray,
    spacing_zyx: Tuple, weights_dir: str = None, **_,
) -> np.ndarray:
    """
    MA-SAM: SAM ViT-H + FacT adaptation.

    Pretrained weights are available (BTCV / prostate on Google Drive) but
    the inference script (MA-SAM-main/MA-SAM/test.py) expects Synapse-format
    pickle data (2Dimage_* / 2Dmask_*), not NIfTI.
    A NIfTI→Synapse-pickle conversion wrapper is required.

    Required checkpoints in <weights_dir>/masam/:
        sam_vit_h_4b8939.pth   — base SAM ViT-H
        ma_sam_btcv.pth        — MA-SAM adapted weights (see comparison/MA-SAM-main/README.md)
    """
    raise NotImplementedError(
        "MA-SAM requires Synapse-format pickle data (2Dimage_* / 2Dmask_*).\n"
        "NIfTI input is not directly supported by its test.py.\n"
        "Add a NIfTI→Synapse-pickle conversion step to enable this runner.\n"
        "Pretrained weights are available — see comparison/MA-SAM-main/README.md."
    )


# ─── Runner registry ─────────────────────────────────────────────────────────

RUNNERS = {
    "sam_med3d":    run_sam_med3d,
    "medsam2":      run_medsam2,
    "medical_sam2": run_medical_sam2,
    "samed2":       run_samed2,
    "ma_sam":       run_ma_sam,
}


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    dataset_paths: List[str],
    weights_dir: str,
    save_path: str,
    comparison_dir: str,
    device: str,
    max_cases: Optional[int] = None,
    medical_sam2_ckpt: str = None,
    samed2_ckpt: str = None,
) -> Dict:
    """
    Evaluate one model across all datasets.
    Save format matches test_vis3d_multi.py:
      {ds_name: [{"img_name", "mask_name", "label",
                  "voxel_count", "voxel_percentage",
                  "dice", "hd", "msd",
                  "cut_axis", "cut_idx"}, ...]}
    """
    runner = RUNNERS[model_name]
    final_record: Dict[str, List[Dict]] = {}

    test_folders = collect_test_folders(dataset_paths)
    print(f"  Found {len(test_folders)} test folder(s).")

    for test_folder in test_folders:
        tf = Path(test_folder).resolve()
        ds_name = tf.parent.name if tf.name == "testing" else tf.name

        pairs = scan_pairs_same_folder(test_folder)
        if not pairs:
            final_record[ds_name] = []
            continue

        if max_cases is not None:
            pairs = pairs[:max_cases]

        ds_rows: List[Dict] = []
        skipped = 0
        kept = 0
        errors = 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_skipped = False
            with _progress(pairs, desc=f"  {ds_name}", unit="case") as pbar:
                for img_path, gt_path in pbar:
                    case_name = Path(img_path).name
                    try:
                        img_zyx, img_sitk = read_3d_image(img_path)
                        gt_zyx,  _        = read_3d_image(gt_path)
                        if img_zyx.shape != gt_zyx.shape:
                            raise ValueError(
                                f"Shape mismatch: img {img_zyx.shape} vs gt {gt_zyx.shape}")
                    except Exception as e:
                        print(f"  [ERROR] loading {case_name}: {e}")
                        skipped += 1
                        continue

                    spacing_xyz = img_sitk.GetSpacing()
                    spacing_zyx = (float(spacing_xyz[2]),
                                   float(spacing_xyz[1]),
                                   float(spacing_xyz[0]))

                    labels = get_labels_from_mask(gt_zyx)
                    if not labels:
                        skipped += 1
                        continue

                    total_voxels = int(gt_zyx.size)
                    primary_axis = choose_primary_axis(img_zyx.shape)
                    gt0 = np.moveaxis(gt_zyx, primary_axis, 0)

                    for lb in labels:
                        gt_binary = (gt_zyx == lb).astype(np.uint8)
                        try:
                            pred_binary = runner(
                                img_path=img_path,
                                gt_path=gt_path,
                                label_id=lb,
                                img_zyx=img_zyx,
                                gt_zyx=gt_zyx,
                                spacing_zyx=spacing_zyx,
                                weights_dir=weights_dir,
                                device=device,
                                comparison_dir=comparison_dir,
                                tmp_dir=tmp_dir,
                                medical_sam2_ckpt=medical_sam2_ckpt,
                                samed2_ckpt=samed2_ckpt,
                            )

                            dsc = dice_3d(pred_binary, gt_binary)
                            hd, msd = hd_msd_3d_sitk(pred_binary, gt_binary,
                                                       spacing_zyx=spacing_zyx)
                            vc = int(gt_binary.sum())
                            vp = vc / float(total_voxels)
                            fg_lb = (gt0 == lb).sum(axis=(1, 2))
                            cut_idx = int(np.argmax(fg_lb)) if fg_lb.max() > 0 else -1

                            ds_rows.append({
                                "img_name":         img_path,
                                "mask_name":        gt_path,
                                "label":            int(lb),
                                "voxel_count":      vc,
                                "voxel_percentage": vp,
                                "dice":             float(dsc),
                                "hd":               float(hd),
                                "msd":              float(msd),
                                "cut_axis":         int(primary_axis),
                                "cut_idx":          cut_idx,
                            })
                            kept += 1

                        except NotImplementedError as e:
                            print(f"  [SKIP] {model_name}: {e}")
                            model_skipped = True
                            break
                        except Exception as e:
                            print(f"  [ERROR] {case_name} label {lb}: {e}")
                            errors += 1

                    pbar.set_postfix(kept=kept, skip=skipped, err=errors)

                    if model_skipped:
                        break

        final_record[ds_name] = ds_rows
        print(f"[Done] {ds_name}: kept={kept}  skipped={skipped}  errors={errors}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(final_record, f)
    print(f"\nSaved → {save_path}")

    return final_record


def print_summary(final_record: Dict, model_name: str) -> None:
    all_rows = [r for rows in final_record.values() for r in rows]
    if not all_rows:
        print(f"\n  No results for {model_name}.")
        return

    by_label: Dict = defaultdict(lambda: {"dice": [], "hd": [], "msd": []})
    for r in all_rows:
        lb = r["label"]
        by_label[lb]["dice"].append(r["dice"])
        if r["hd"]  != float("inf"): by_label[lb]["hd"].append(r["hd"])
        if r["msd"] != float("inf"): by_label[lb]["msd"].append(r["msd"])

    print(f"\n{'═'*62}")
    print(f"  Results — {model_name}"
          f"  (cases: {len(set(r['img_name'] for r in all_rows))})")
    print(f"{'═'*62}")
    print(f"  {'Label':<8}  {'Dice':>8}  {'HD (mm)':>10}  {'MSD (mm)':>10}")
    print(f"  {'─'*48}")
    all_d, all_h, all_m = [], [], []
    for lb in sorted(by_label):
        d = float(np.mean(by_label[lb]["dice"])) if by_label[lb]["dice"] else float("nan")
        h = float(np.mean(by_label[lb]["hd"]))   if by_label[lb]["hd"]   else float("inf")
        m = float(np.mean(by_label[lb]["msd"]))  if by_label[lb]["msd"]  else float("inf")
        all_d.append(d); all_h.append(h); all_m.append(m)
        print(f"  {lb:<8}  {d:>8.4f}  {h:>10.2f}  {m:>10.2f}")
    print(f"  {'─'*48}")
    valid_d = [x for x in all_d if not np.isnan(x)]
    valid_h = [x for x in all_h if x != float("inf")]
    valid_m = [x for x in all_m if x != float("inf")]
    print(
        f"  {'Mean':<8}  "
        f"{np.mean(valid_d) if valid_d else float('nan'):>8.4f}  "
        f"{np.mean(valid_h) if valid_h else float('inf'):>10.2f}  "
        f"{np.mean(valid_m) if valid_m else float('inf'):>10.2f}"
    )
    print(f"{'═'*62}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate comparison 3D segmentation models on SAMRI2 NIfTI test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", choices=list(RUNNERS) + ["all"], required=True,
        help="Model to evaluate. 'all' runs every registered model sequentially.",
    )
    p.add_argument(
        "--dataset-path", nargs="+", required=True,
        help="Test dataset root path(s) (same as test_vis3d_multi.py --dataset-path).",
    )
    p.add_argument("--weights-dir", required=True,
                   help="Root directory with per-model weight sub-folders.")
    p.add_argument("--save-path", required=True,
                   help="Output .pkl path (same format as test_vis3d_multi.py output).")
    p.add_argument(
        "--comparison-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison"),
        help="Path to comparison/ folder containing model subfolders.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-cases", type=int, default=None,
                   help="Limit cases per dataset (for debugging).")
    p.add_argument("--medical-sam2-ckpt", default=None,
                   help="Explicit checkpoint path for Medical-SAM2 "
                        "(overrides <weights-dir>/medica_sam2/checkpoint.pt).")
    p.add_argument("--samed2-ckpt", default=None,
                   help="Explicit checkpoint path for SAMed-2 "
                        "(overrides <weights-dir>/samed2/samed2.pth).")
    return p.parse_args()


class _Tee:
    """Write to both the original stream and a log file simultaneously."""
    def __init__(self, original, log_file):
        self._orig = original
        self._log  = log_file

    def write(self, data):
        self._orig.write(data)
        self._orig.flush()
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._orig.flush()
        self._log.flush()

    def fileno(self):          # needed by some libraries that check for a real fd
        return self._orig.fileno()


def main():
    args = parse_args()

    # ── Tee stdout + stderr → <save_path_dir>/log.txt ──────────────────────
    log_dir  = Path(args.save_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"
    _log_fh  = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    print(f"Logging to: {log_path}")
    # ────────────────────────────────────────────────────────────────────────

    device = args.device if torch.cuda.is_available() else "cpu"
    models_to_run = list(RUNNERS) if args.model == "all" else [args.model]

    t0 = time.time()
    all_results = {}

    for model_name in models_to_run:
        print(f"\n{'═'*62}")
        print(f"  Model: {model_name}")
        print(f"{'═'*62}")
        t1 = time.time()

        # Derive per-model save path when running 'all'
        if args.model == "all":
            sp = Path(args.save_path)
            save_path = str(sp.parent / (sp.stem + f"_{model_name}" + sp.suffix))
        else:
            save_path = args.save_path

        record = evaluate_model(
            model_name=model_name,
            dataset_paths=args.dataset_path,
            weights_dir=args.weights_dir,
            save_path=save_path,
            comparison_dir=args.comparison_dir,
            device=device,
            max_cases=args.max_cases,
            medical_sam2_ckpt=args.medical_sam2_ckpt,
            samed2_ckpt=args.samed2_ckpt,
        )
        all_results[model_name] = record
        print_summary(record, model_name)
        print(f"  Elapsed: {time.time() - t1:.1f}s")

    if args.model == "all":
        with open(args.save_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\nCombined results saved → {args.save_path}")

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")

    # Restore original streams and close log file
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_fh.close()
    print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()
