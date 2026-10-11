#!/usr/bin/env python3
"""
comparison_test.py

Evaluate comparison 3D segmentation models on SAMRI-3D-format NIfTI test data.

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
                               (2026-10 fix: box now passed in original coordinates --
                               it was previously scaled twice; rerun required.)
  medical_sam2  Medical-SAM2 — build_sam2_video_predictor (sam2_train package, no
                               namespace conflict); (S,3,512,512) tensor input;
                               bbox prompt; val_init_state + propagate_in_video.
  samed2        SAMed-2      — MedicalSegmenter (SAMed-2's sam2 fork, swapped via
                               _swap_sam2); 2D per-slice 1024×1024 input; box prompt
                               from GT per slice; aggregated to 3D volume.
                               NOTE: SAMed-2 gets a GT box on EVERY slice containing the
                               target and empty slices are skipped (no propagation), a
                               stronger prompt than the single cut-slice box used for the
                               other methods.
  slm_sam2      SLM-SAM2     — SLM-SAM2's sam2 fork (swapped via _swap_sam2), trained by
                               us with the official recipe on the SAMRI-3D training split
                               (comparison/slm_sam2_tools/). JPEG frames along the primary
                               axis; box (default) or GT-mask (--slm-prompt mask) prompt on
                               the cut slice; forward + reverse propagation; recent_n=1.

Stub (format incompatible) (running but too slow to run on the full test set, the 
                            initial run is around 20 minutes, then 30 min for big slice case,
                            using 22GB VRAM, so not suitable for large-scale evaluation 
                            without further optimization):
--------------------------
  ma_sam        MA-SAM       — requires Synapse-format pickle data (2Dimage_* / 2Dmask_*);
                               NIfTI input not supported by its test.py. Add a
                               NIfTI→Synapse-pickle conversion wrapper to enable.

sam2 namespace management
--------------------------
MedSAM2 and SAMed-2 each ship their own sam2 fork with symbols absent from SAMRI-3D's sam2.
Both use _swap_sam2() during model construction to temporarily replace sys.modules['sam2*']
with the correct fork, then restore SAMRI-3D's sam2 on exit.

Weights directory layout
------------------------
Each subfolder must contain exactly one checkpoint file (.pt/.pth/.pkl/.bin/.safetensors).
The filename does not matter — the script picks up whichever file is present.

<weights_dir>/
├── sam_med3d/      <one .pth>   (https://huggingface.co/blueyo0/SAM-Med3D)
├── MedSAM2/        <one .pt>    (https://huggingface.co/wanglab/MedSAM2)
├── medical_sam2/    <one .pt>    (https://huggingface.co/jiayuanz3/MedSAM2_pretrain)
│                                (override with --medical-sam2-ckpt /explicit/path)
├── samed2/         <one .pth>   (SAMed-2 GitHub page)
│                                (override with --samed2-ckpt /explicit/path)
├── slm_sam2/       <checkpoint.pt> trained by us (no released weights)
│                                (override with --slm-sam2-ckpt /explicit/path)
└── masam/
    ├── sam_vit_h_4b8939.pth    (Meta SAM ViT-H base)
    └── ma_sam_btcv.pth         (MA-SAM Google Drive — see comparison/MA-SAM-main/README.md)

Usage
-----
# Single model
python comparison_test.py \
    --model medsam2 \
    --dataset-path /path/to/Test_3d/ /path/to/ZeroShot/ \
    --weights-dir /path/to/weights/ \
    --save-path results/medsam2.pkl

# All models sequentially (saves per-model .pkl and a combined .pkl)
python comparison_test.py \
    --model all \
    --dataset-path /path/to/Test_3d/ /path/to/ZeroShot/ \
    --weights-dir /path/to/weights/ \
    --save-path results/comparison.pkl \
    --device cuda:0

# Debug: limit to 2 cases per dataset
python comparison_test.py \
--model samed2 \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ /scratch/user/s4670484/Datasets/Zeroshot3D/ \
--weights-dir /scratch/user/s4670484/comparison_ckpt/ \
--save-path /scratch/user/s4670484/Eval_results/comparison/samed2_debug.pkl \
--max-cases 1

# SLM-SAM2 (our trained checkpoint; see comparison/slm_sam2_tools/README.md)
python comparison_test.py \
--model slm_sam2 --slm-prompt box \
--slm-sam2-ckpt /scratch/user/s4670484/Model_save/slm_sam2_samri3d/checkpoints/checkpoint.pt \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ /scratch/user/s4670484/Datasets/Zeroshot3D/ \
--weights-dir /scratch/user/s4670484/comparison_ckpt/ \
--save-path /scratch/user/s4670484/Eval_results/comparison/slm_sam2_box/slm_sam2_box.pkl

example:
python comparison_test.py \
--model medical_sam2 \
--dataset-path /scratch/user/s4670484/Datasets/SAMRI_datasets_3D/ /scratch/user/s4670484/Datasets/Zeroshot3D/ \
--weights-dir /scratch/user/s4670484/comparison_ckpt/ \
--save-path /scratch/user/s4670484/Eval_results/comparison/medical_sam2.pkl
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
_SAMRI3D_ROOT = Path(__file__).resolve().parent
if str(_SAMRI3D_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAMRI3D_ROOT))

from test_vis3d_multi import (
    read_3d_image,
    get_labels_from_mask,
    choose_primary_axis,
    normalize_to_uint8,
    write_video_frames_to_dir,
    dice_3d,
    hd_msd_3d_sitk,
    scan_pairs_same_folder,
    collect_test_folders,
)


# ─── sam2 namespace swap helper ───────────────────────────────────────────────
# Importing test_vis3d_multi above loads SAMRI-3D's sam2 into sys.modules['sam2'].
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
    # FIX (2026-10): pass the box in ORIGINAL (H, W) pixel coordinates.
    # init_state(img_tensor, H, W) stores video_height/width = H, W, and
    # add_new_points_or_box (normalize_coords=True) divides by (W, H) and multiplies
    # by the model's image_size (512) itself -- exactly as MedSAM2's own
    # medsam2_infer_3D_CT.py does. The previous code pre-scaled the box to 512
    # (box_512), so it was scaled twice and landed in the wrong place unless
    # H = W = 512; this explains MedSAM2's near-zero Dice in the earlier results.

    pred_shw = np.zeros((S, H, W), dtype=bool)
    autocast_device = "cuda" if "cuda" in device else "cpu"
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
        # Forward pass
        state = predictor.init_state(img_tensor, H, W)
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=key_s, obj_id=1, box=box)
        for fidx, _, logits in predictor.propagate_in_video(state):
            # logits[0] shape: (1, H512, W512)
            m = (logits[0] > 0.0).cpu().numpy()[0]
            pred_shw[fidx] = _resize_mask_to_hw(m, H, W)
        predictor.reset_state(state)

        # Reverse pass (original MedSAM2 pattern: re-init + re-prompt + reverse=True)
        state = predictor.init_state(img_tensor, H, W)
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=key_s, obj_id=1, box=box)
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

    # sam2_train/__init__.py calls initialize_config_module("sam2_train") unconditionally
    # (no is_initialized() guard).  SAMRI-3D's sam2/__init__.py already initialized
    # GlobalHydra with "sam2", so we must clear it BEFORE the first import of sam2_train
    # (importing the package executes __init__.py).  On subsequent calls the module is
    # cached by Python, so __init__.py does NOT re-run and Hydra stays initialized with
    # "sam2_train" — clearing it would leave Hydra uninitialized and break compose().
    if "sam2_train" not in sys.modules:
        from hydra.core.global_hydra import GlobalHydra  # noqa: PLC0415
        GlobalHydra.instance().clear()

    # Medical-SAM2 ships its own sam2_train package (no conflict with sam2)
    from sam2_train.build_sam import build_sam2_video_predictor  # noqa: PLC0415

    # Configs live directly under sam2_train/ (not in configs/sam2.1/).
    # Prefer hiera_t (tiny) — that is what the released Medical-SAM2 checkpoint uses.
    sam2_train_pkg_dir = med_sam2_root / "sam2_train"
    cfg_path = None
    for _stem in ("sam2_hiera_t", "sam2_hiera_s", "sam2.1_hiera_t", "sam2.1_hiera_s"):
        _c = sam2_train_pkg_dir / f"{_stem}.yaml"
        if _c.exists():
            cfg_path = _c
            break
    if cfg_path is None:
        cfgs = sorted(sam2_train_pkg_dir.rglob("*.yaml"))
        if not cfgs:
            raise FileNotFoundError(
                f"Medical-SAM2 config YAML not found under {med_sam2_root}")
        cfg_path = cfgs[0]

    # Hydra's search path is pkg://sam2_train, so cfg name must be relative to
    # the sam2_train package dir (no extension), not an absolute path.
    try:
        cfg_name = str(cfg_path.relative_to(sam2_train_pkg_dir).with_suffix(""))
    except ValueError:
        cfg_name = cfg_path.stem

    cache_key = str(ckpt_path)
    if not hasattr(run_medical_sam2, "_cache") or run_medical_sam2._cache[0] != cache_key:
        # apply_postprocessing=False avoids fill_hole_area=8, which requires
        # the compiled sam2_train._C extension (not built in this environment).
        net = build_sam2_video_predictor(cfg_name, str(ckpt_path), device=device,
                                         apply_postprocessing=False)
        net.eval()
        run_medical_sam2._cache = (cache_key, net)
    net = run_medical_sam2._cache[1]

    # Medical-SAM2 config sets image_size=1024, backbone_stride=16 →
    # sam_image_embedding_size=64; input must be 1024×1024.
    FRAME_SIZE = getattr(net, "image_size", 1024)

    primary_axis = choose_primary_axis(img_zyx.shape)
    vol_shw = np.moveaxis(img_zyx, primary_axis, 0)  # [S, H, W]
    gt_shw  = np.moveaxis(gt_zyx,  primary_axis, 0)
    S, H, W = vol_shw.shape

    img_tensor = _vol_to_rgb_tensor(vol_shw, FRAME_SIZE, device)

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
    _autocast_device = "cuda" if "cuda" in str(device) else "cpu"
    with torch.no_grad(), torch.autocast(_autocast_device, dtype=torch.bfloat16):
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

    import torch.nn.functional as _F

    model  = segmenter.model
    dev    = segmenter.device
    tform  = segmenter.transform

    pred_shw = np.zeros((S, H, W), dtype=bool)
    for s in range(S):
        s2d = normalize_to_uint8(vol_shw[s])
        pil = PILImage.fromarray(s2d).convert("RGB")

        # Box from GT label on this slice, scaled to SAMed-2's 1024×1024 input.
        # Skip slices with no foreground — predicting on empty slices inflates FP.
        box_hw = mask2d_to_bbox_xyxy((gt_shw[s] == label_id).astype(np.uint8))
        if box_hw is None:
            continue  # leave pred_shw[s] = False
        box_1024 = (box_hw * np.array(
            [IMG_SIZE / W, IMG_SIZE / H, IMG_SIZE / W, IMG_SIZE / H]
        )).tolist()

        # SAMed-2's TemporalAdapter requires B divisible by d_size=2, so B must be 2.
        # predict.py duplicates the image but creates empty memory with B=1 (bug).
        # We replicate predict.py's inference here with the corrected B=2 memory.
        img_1  = tform(pil).unsqueeze(0).to(dev)
        img_t  = torch.cat([img_1, img_1], dim=0)            # (2, 3, 1024, 1024)
        box_t  = torch.tensor([box_1024, box_1024], device=dev)  # (2, 4)

        with torch.no_grad():
            bb_out = model.forward_image(img_t)
            _, v_feats, v_pos, feat_sizes = model._prepare_backbone_features(bb_out)

            B = 2
            C = model.hidden_dim
            fH, fW = feat_sizes[-1]

            # SAMed-2 config has directly_add_no_mem_embed=True: for the no-memory
            # (single-frame) case SAM2 skips memory_attention entirely and just adds
            # no_mem_embed to the vision features.  The memory_attention cross-attention
            # expects kv_in_dim=64, so passing zeros of dim C=256 would fail.
            if getattr(model, "directly_add_no_mem_embed", False):
                pix_feat = v_feats[-1] + model.no_mem_embed
            else:
                mem_dim = getattr(model, "mem_dim", 64)
                memory     = torch.zeros(1, B, mem_dim, device=dev)
                memory_pos = torch.zeros(1, B, mem_dim, device=dev)
                pix_feat = model.memory_attention(
                    curr=v_feats[-1:],
                    curr_pos=v_pos[-1:],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0,
                )
            backbone_feat = pix_feat.permute(1, 2, 0).view(B, C, fH, fW)

            hr_feats = (
                [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                 for x, s in zip(v_feats[:-1], feat_sizes[:-1])]
                if len(v_feats) > 1 else None
            )

            sparse_e, dense_e = model.sam_prompt_encoder(
                points=None, boxes=box_t, masks=None, batch_size=B
            )
            low_res, _, _, _ = model.sam_mask_decoder(
                image_embeddings=backbone_feat,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_e,
                dense_prompt_embeddings=dense_e,
                multimask_output=False,
                repeat_image=False,
                high_res_features=hr_feats,
            )
            masks_up = _F.interpolate(low_res.float(), size=(IMG_SIZE, IMG_SIZE),
                                      mode="bilinear", align_corners=False)
            mask_1024 = (masks_up[0, 0] > 0.0).cpu().numpy()  # (1024, 1024) bool

        # Resize prediction back to original (H, W)
        pred_shw[s] = _resize_mask_to_hw(mask_1024, H, W)

    pred_zyx = np.moveaxis(pred_shw, 0, primary_axis)
    return pred_zyx


# ─── SLM-SAM2 runner ──────────────────────────────────────────────────────────

SLM_SAM2_CONFIG = "configs/sam2.1/slm_sam2_hiera_t.yaml"


def _load_slm_sam2(slm_root: Path, ckpt_path: Path, device: str, recent_n: int):
    """Build SLM-SAM2's video predictor (its own sam2 fork) and load OUR trained weights.

    * The official inference config sets recent_n: 7, but SLM-SAM2 is trained with
      recent_n: 1 and the model reads self.recent_n (the propagate_in_video argument is
      ignored by _track_step), so we override it to match training.
    * Weights are loaded with an explicit missing-key check: SLM-SAM2's own loader uses
      strict=False and only logs, which would silently leave the short-term memory
      branch random if a plain SAM2 checkpoint were passed by mistake.
    """
    with _swap_sam2(slm_root):
        from hydra.core.global_hydra import GlobalHydra  # noqa: PLC0415
        GlobalHydra.instance().clear()
        from sam2.build_sam import build_sam2_video_predictor  # noqa: PLC0415
        predictor = build_sam2_video_predictor(
            SLM_SAM2_CONFIG, None, device=device,
            hydra_overrides_extra=[f"++model.recent_n={int(recent_n)}"],
        )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = predictor.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"SLM-SAM2 checkpoint {ckpt_path} is missing {len(missing)} keys "
            f"(e.g. {missing[:5]}). Is this a SLM-SAM2-trained checkpoint?")
    if unexpected:
        print(f"  [SLM-SAM2] ignoring {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    if not any(k.startswith("recent_memory_attention.") for k in sd):
        raise RuntimeError(f"{ckpt_path} has no recent_memory_attention weights.")
    predictor.eval()
    print(f"  [SLM-SAM2] loaded {ckpt_path} | recent_n={predictor.recent_n} | "
          f"image_size={predictor.image_size}")
    return predictor


def run_slm_sam2(
    img_path: str, gt_path: str, label_id: int,
    img_zyx: np.ndarray, gt_zyx: np.ndarray,
    spacing_zyx: Tuple, weights_dir: str, device: str,
    comparison_dir: str, tmp_dir: str,
    slm_sam2_ckpt: str = None, slm_prompt: str = "box", slm_recent_n: int = 1, **_,
) -> np.ndarray:
    """
    SLM-SAM2 (Chen et al., IEEE TMI 2025) -- SAM2.1-Tiny with an extra short-term memory
    branch, trained by us with the official recipe on the SAMRI-3D training split
    (no released weights exist).

    Shim: slices along the primary (shortest) axis -> min-max uint8 RGB JPEGs
          (same preprocessing as SAMRI-3D and as the SLM-SAM2 training export), written
          once per case and reused across labels.
    Prompt: on the cut slice (max foreground for this label), as for every other method:
          slm_prompt="box"  -> GT bounding box (headline protocol, same as SAMRI-3D)
          slm_prompt="mask" -> GT mask of that slice (SLM-SAM2's native prompt)
    Propagation: official inference.py pattern -- forward from the cut slice, then
          reverse from the cut slice in the same state.
    """
    comp = Path(comparison_dir)
    slm_root = comp / "SLM-SAM2-main"
    ckpt_path = (Path(slm_sam2_ckpt) if slm_sam2_ckpt else
                 _find_checkpoint(Path(weights_dir) / "slm_sam2", "SLM-SAM2"))

    cache_key = (str(ckpt_path), int(slm_recent_n))
    if not hasattr(run_slm_sam2, "_cache") or run_slm_sam2._cache[0] != cache_key:
        run_slm_sam2._cache = (cache_key,
                               _load_slm_sam2(slm_root, ckpt_path, device, slm_recent_n))
    predictor = run_slm_sam2._cache[1]

    primary_axis = choose_primary_axis(img_zyx.shape)
    vol_shw = np.moveaxis(img_zyx, primary_axis, 0)  # [S, H, W]
    gt_shw = np.moveaxis(gt_zyx, primary_axis, 0)
    S, H, W = vol_shw.shape

    fg = (gt_shw == label_id).sum(axis=(1, 2))
    if fg.max() == 0:
        return np.zeros(img_zyx.shape, dtype=bool)
    key_s = int(np.argmax(fg))
    key_mask = (gt_shw[key_s] == label_id)

    # JPEG frames: written once per case, reused for every label of that case
    prev = getattr(run_slm_sam2, "_frames", None)
    if prev is None or prev[0] != img_path:
        if prev is not None:
            import shutil  # noqa: PLC0415
            shutil.rmtree(prev[1], ignore_errors=True)
        frames_dir = tempfile.mkdtemp(prefix="slm_frames_", dir=tmp_dir)
        write_video_frames_to_dir(vol_shw, list(range(S)), frames_dir)
        run_slm_sam2._frames = (img_path, frames_dir)
    frames_dir = run_slm_sam2._frames[1]

    pred_shw = np.zeros((S, H, W), dtype=bool)
    use_amp = "cuda" in str(device)
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
               else contextlib.nullcontext())
    with torch.inference_mode(), amp_ctx:
        state = predictor.init_state(video_path=frames_dir)
        if slm_prompt == "box":
            box = mask2d_to_bbox_xyxy(key_mask.astype(np.uint8))  # original (H, W) coords
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=key_s, obj_id=1, box=box)
        elif slm_prompt == "mask":
            predictor.add_new_mask(
                inference_state=state, frame_idx=key_s, obj_id=1, mask=key_mask)
        else:
            raise ValueError(f"unknown slm_prompt: {slm_prompt}")

        for fidx, _, logits in predictor.propagate_in_video(
                state, start_frame_idx=key_s, recent_n=slm_recent_n):
            pred_shw[fidx] = (logits[0] > 0.0).cpu().numpy()[0]
        if key_s > 0:
            for fidx, _, logits in predictor.propagate_in_video(
                    state, start_frame_idx=key_s, reverse=True, recent_n=slm_recent_n):
                pred_shw[fidx] = (logits[0] > 0.0).cpu().numpy()[0]
        predictor.reset_state(state)

    return np.moveaxis(pred_shw, 0, primary_axis)


# ─── MA-SAM stub ──────────────────────────────────────────────────────────────

def run_ma_sam(
    img_path: str, gt_path: str, label_id: int,
    img_zyx: np.ndarray, gt_zyx: np.ndarray,
    spacing_zyx: Tuple, weights_dir: str = None, device: str = "cuda",
    comparison_dir: str = None, **_,
) -> np.ndarray:
    """
    MA-SAM: SAM ViT-H + FacT-TT + 3D adapters (2.5D per-slice inference).

    Inputs are built inline from NIfTI arrays — no pickle conversion needed.
    Each forward pass takes a 5-slice window (1, 5, 3, 512, 512); the centre
    slice prediction is used.  Overlapping window averaging is not necessary
    for a binary-per-label evaluation.

    Required checkpoints in <weights_dir>/ma_sam/:
        sam_vit_h_4b8939.pth   — base SAM ViT-H  (~2.4 GB)
        <adapted>.pth          — MA-SAM fine-tuned weights (smaller file)
    """
    from scipy.ndimage import zoom as _zoom

    ma_sam_root = Path(comparison_dir) / "MA-SAM-main" / "MA-SAM"
    ckpt_dir    = Path(weights_dir) / "ma_sam"
    ckpts = sorted(ckpt_dir.glob("*.pth")) + sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    # Base SAM ViT-H is ~2.4 GB; MA-SAM adapted weights are much smaller.
    # Use size ≥ 1 GB as the threshold to distinguish them — avoids loading
    # 2.4 GB just to peek at keys.
    _GB = 1024 ** 3
    large = [c for c in ckpts if c.stat().st_size >= _GB]
    small = [c for c in ckpts if c.stat().st_size <  _GB]
    base_ckpt  = large[0] if large else None   # base SAM ViT-H (may be absent)
    adapt_ckpt = (small or large)[0]           # MA-SAM adapted weights

    cache_key = str(adapt_ckpt)
    if not hasattr(run_ma_sam, "_cache") or run_ma_sam._cache[0] != cache_key:
        _ma_path = str(ma_sam_root)
        if _ma_path not in sys.path:
            sys.path.insert(0, _ma_path)

        # segment_anything's __init__.py imports predictor and automatic_mask_generator,
        # both of which pull in torchvision.  The torchvision in this environment has a
        # torch.onnx.DiagnosticOptions incompatibility that crashes on import.
        # Pre-stub every torchvision-touching submodule before segment_anything loads
        # so Python never executes those files.  We don't use SamPredictor or
        # SamAutomaticMaskGenerator at inference — only build_sam and Fact_tt_Sam.
        import types as _types

        def _stub(key, **attrs):
            if key not in sys.modules:
                _m = _types.ModuleType(key)
                for _k, _v in attrs.items():
                    setattr(_m, _k, _v)
                sys.modules[key] = _m

        _stub("segment_anything.utils")
        _stub("segment_anything.utils.transforms", ResizeLongestSide=None)
        _stub("segment_anything.utils.onnx")
        _stub("segment_anything.utils.amg")
        _stub("segment_anything.predictor", SamPredictor=None)
        _stub("segment_anything.automatic_mask_generator", SamAutomaticMaskGenerator=None)

        from segment_anything.build_sam import sam_model_registry  # noqa: PLC0415
        from sam_fact_tt_image_encoder import Fact_tt_Sam          # noqa: PLC0415

        # Detect num_classes from the adapted checkpoint's mask_decoder keys.
        # MaskDecoder stores num_multimask_outputs+1 MLPs (the +1 is the single-mask
        # head), so subtract 1 to recover the actual num_classes used at training.
        _sd = torch.load(str(adapt_ckpt), map_location="cpu")
        _n_mlps = sum(
            1 for k in _sd
            if "mask_decoder.output_hypernetworks_mlps" in k
            and "layers.0.weight" in k
        )
        num_classes = max(_n_mlps - 1, 1) if _n_mlps > 0 else 12  # default: BTCV/Synapse

        sam, _ = sam_model_registry["vit_h"](
            image_size=512,
            num_classes=num_classes,
            checkpoint=str(base_ckpt) if base_ckpt else None,
            pixel_mean=[0., 0., 0.],
            pixel_std=[1., 1., 1.],
        )
        net = Fact_tt_Sam(sam, r=32, s=1.0).to(device)
        net.load_parameters(str(adapt_ckpt))
        net.eval()
        run_ma_sam._cache = (cache_key, net, num_classes)

    net, num_classes = run_ma_sam._cache[1], run_ma_sam._cache[2]
    IMG_SIZE = 512
    D_WINDOW = 5  # 2.5D context window; centre index = 2

    primary_axis = choose_primary_axis(img_zyx.shape)
    vol_shw = np.moveaxis(img_zyx, primary_axis, 0).astype(np.float32)  # (S, H, W)
    gt_shw  = np.moveaxis(gt_zyx,  primary_axis, 0)
    S, H, W = vol_shw.shape

    if (gt_shw == label_id).sum() == 0:
        return np.zeros(img_zyx.shape, dtype=bool)

    # Per-slice min-max normalise to [0, 1] (MA-SAM uses pixel_mean/std=0/1)
    norm_vol = np.zeros_like(vol_shw)
    for i in range(S):
        sl = vol_shw[i]
        lo, hi = sl.min(), sl.max()
        norm_vol[i] = (sl - lo) / (hi - lo + 1e-8)

    pred_shw = np.zeros((S, H, W), dtype=bool)
    with torch.no_grad():
        for ind in range(S):
            # Build 5-slice window (H, W, 5); pad by repeating boundary slices
            idx5   = [max(0, min(S - 1, ind + k - 2)) for k in range(D_WINDOW)]
            window = norm_vol[idx5]                      # (5, H, W)
            window = window.transpose(1, 2, 0)           # (H, W, 5)

            if H != IMG_SIZE or W != IMG_SIZE:
                window = _zoom(window, (IMG_SIZE / H, IMG_SIZE / W, 1.0), order=3)

            # Replicate test.py: (1,1,H,W,5) → repeat3 → (1,3,H,W,5) → permute → (1,5,3,H,W)
            t = torch.from_numpy(window).float().to(device)
            t = t.unsqueeze(0).unsqueeze(0)              # (1, 1, H, W, 5)
            t = t.repeat(1, 3, 1, 1, 1)                 # (1, 3, H, W, 5)
            t = t.permute(0, 4, 1, 2, 3)                # (1, 5, 3, H, W)

            outputs   = net(t, multimask_output=True, image_size=IMG_SIZE)
            masks_out = outputs["masks"]                 # (5, num_classes, IMG_SIZE, IMG_SIZE)

            # Take the centre slice (index 2) prediction
            centre = masks_out[2]                        # (num_classes, IMG_SIZE, IMG_SIZE)
            pred_class = centre.argmax(dim=0).cpu().numpy()   # (IMG_SIZE, IMG_SIZE)

            if H != IMG_SIZE or W != IMG_SIZE:
                pred_class = _zoom(pred_class.astype(np.float32),
                                   (H / IMG_SIZE, W / IMG_SIZE), order=0).astype(int)

            pred_shw[ind] = (pred_class == label_id)

    return np.moveaxis(pred_shw, 0, primary_axis)


# ─── Runner registry ─────────────────────────────────────────────────────────

RUNNERS = {
    "sam_med3d":    run_sam_med3d,
    "medsam2":      run_medsam2,
    "medical_sam2": run_medical_sam2,
    "samed2":       run_samed2,
    "slm_sam2":     run_slm_sam2,
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
    slm_sam2_ckpt: str = None,
    slm_prompt: str = "box",
    slm_recent_n: int = 1,
    resume: bool = False,
) -> Dict:
    """
    Evaluate one model across all datasets.

    After every dataset the rows collected so far are written to
    <save_path>.partial.pkl; with resume=True, datasets already in that file are
    skipped, so a job cut by the scheduler only loses the dataset in progress.
    Save format matches test_vis3d_multi.py:
      {ds_name: [{"img_name", "mask_name", "label",
                  "voxel_count", "voxel_percentage",
                  "dice", "hd", "msd",
                  "cut_axis", "cut_idx"}, ...]}
    """
    runner = RUNNERS[model_name]
    final_record: Dict[str, List[Dict]] = {}

    partial_path = str(save_path) + ".partial.pkl"
    done: Dict[str, List[Dict]] = {}
    if resume and os.path.exists(partial_path):
        with open(partial_path, "rb") as f:
            done = pickle.load(f)
        print(f"  Resuming: {len(done)} dataset(s) already in {partial_path}")

    def _save_partial():
        Path(partial_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = partial_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(final_record, f)
        os.replace(tmp, partial_path)  # atomic: a cut mid-write cannot corrupt it

    test_folders = collect_test_folders(dataset_paths)
    print(f"  Found {len(test_folders)} test folder(s).")

    for test_folder in test_folders:
        tf = Path(test_folder).resolve()
        ds_name = tf.parent.name if tf.name == "testing" else tf.name

        if ds_name in done:
            final_record[ds_name] = done[ds_name]
            print(f"[Skip] {ds_name}: {len(done[ds_name])} rows from partial file")
            continue

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
                                slm_sam2_ckpt=slm_sam2_ckpt,
                                slm_prompt=slm_prompt,
                                slm_recent_n=slm_recent_n,
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
                            import traceback as _tb
                            _msg = str(e) or "(no message)"
                            print(f"  [ERROR] {case_name} label {lb}: {type(e).__name__}: {_msg}")
                            print("".join(_tb.format_exception(type(e), e, e.__traceback__)).strip())
                            errors += 1

                    pbar.set_postfix(kept=kept, skip=skipped, err=errors)

                    if model_skipped:
                        break

        final_record[ds_name] = ds_rows
        print(f"[Done] {ds_name}: kept={kept}  skipped={skipped}  errors={errors}")
        _save_partial()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(final_record, f)
    print(f"\nSaved → {save_path}")
    if os.path.exists(partial_path):
        os.remove(partial_path)

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
        description="Evaluate comparison 3D segmentation models on SAMRI-3D NIfTI test data",
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
    p.add_argument("--slm-sam2-ckpt", default=None,
                   help="Our SLM-SAM2 checkpoint (<exp-dir>/checkpoints/checkpoint.pt from "
                        "comparison/slm_sam2_tools/train_slm_sam2.py); "
                        "overrides <weights-dir>/slm_sam2/.")
    p.add_argument("--slm-prompt", choices=["box", "mask"], default="box",
                   help="SLM-SAM2 prompt on the cut slice: box (headline protocol, same as "
                        "SAMRI-3D) or mask (SLM-SAM2's native GT-mask prompt).")
    p.add_argument("--resume", action="store_true",
                   help="Skip datasets already saved in <save-path>.partial.pkl "
                        "(written after every dataset).")
    p.add_argument("--slm-recent-n", type=int, default=1,
                   help="SLM-SAM2 short-term memory length; 1 = value used in training.")
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
    # append when resuming so the log of the interrupted run is kept
    _log_fh  = open(log_path, "a" if args.resume else "w", buffering=1)
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
            slm_sam2_ckpt=args.slm_sam2_ckpt,
            slm_prompt=args.slm_prompt,
            slm_recent_n=args.slm_recent_n,
            resume=args.resume,
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
