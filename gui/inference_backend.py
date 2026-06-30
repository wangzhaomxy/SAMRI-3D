#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gui/inference_backend.py

Backend used by the SAMRI-3D interactive GUI (gui/samri3d_gui.py). It wraps the same
3D propagation pipeline used by test_vis3d_multi.py, but takes a *user-drawn* prompt
(a box and/or click points on one chosen slice) instead of a GT-derived prompt.

Pipeline (identical to the rest of SAMRI-3D):
  1. View the volume along a chosen axis; the user picks a prompt slice and draws a box
     (and/or positive/negative points) on it.
  2. The volume is split at the prompt slice into a forward and a backward "video".
  3. SAM2's video predictor propagates the prompt outward through both sub-sequences.
  4. The two halves are merged into a single 3D binary mask in the view orientation.

Nothing here imports GUI libraries, so it can be unit-tested / scripted headless.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Repo root on sys.path so `sam2`, `gvt`, and test_vis3d_multi resolve from gui/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sam2.build_sam import build_sam2_video_predictor
from test_vis3d_multi import (
    read_3d_image,
    choose_primary_axis,
    build_two_videos_indices,
    prepare_videos_once,
    init_state_compat,
    precompute_slice_features,
    inject_cached_features,
    propagate_video_singleobj,
    merge_two_videos_single_label,
)

_GVT_AVAILABLE = False
try:
    from gvt.gvt_integration import wrap_model_with_gvt, has_gvt, load_gvt_checkpoint
    from gvt.gvt_inference import compute_gvt_from_cached_features, clear_gvt_for_predictor
    _GVT_AVAILABLE = True
except ImportError:
    pass


# ----------------------------
# Volume loading
# ----------------------------
def load_volume(path: str, view_axis: Optional[int] = None):
    """Read a NIfTI/MHA/etc. volume.

    Returns (vol0, img_zyx, sitk_img, view_axis) where `vol0` is the volume re-oriented
    so that slices run along axis 0 (the viewing/prompt axis). `view_axis` defaults to
    the shortest axis (matches SAMRI-3D's cut-axis convention).
    """
    img_zyx, sitk_img = read_3d_image(path)
    if view_axis is None:
        view_axis = choose_primary_axis(img_zyx.shape)
    vol0 = np.moveaxis(img_zyx, view_axis, 0)
    return vol0, img_zyx, sitk_img, int(view_axis)


# ----------------------------
# Model loading (mirrors test_vis3d_multi.py main())
# ----------------------------
def load_predictor(sam2_config: str, ckpt_path: str, device: str = "cuda",
                   use_gvt: bool = False):
    """Build a SAM2 video predictor; optionally wrap it with GVT (config auto-detected
    from the checkpoint, exactly like `test_vis3d_multi.py --use-gvt`).

    Returns (predictor, is_gvt).
    """
    import torch

    if not use_gvt:
        predictor = build_sam2_video_predictor(sam2_config, ckpt_path, device=device)
        return predictor, False

    if not _GVT_AVAILABLE:
        raise RuntimeError("GVT requested but the gvt/ package is not importable.")

    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    gvt_cfg = ckpt_data.get("gvt_config", None)
    sd = ckpt_data.get("model", ckpt_data)
    if gvt_cfg is not None:
        num_tokens, num_layers, num_heads = (
            gvt_cfg["num_tokens"], gvt_cfg["num_layers"], gvt_cfg["num_heads"])
    else:
        q_key = "memory_attention.gvt_module.global_queries"
        num_tokens = sd[q_key].shape[0] if q_key in sd else 16
        layer_indices = {
            int(k.split("gvt_module.cross_attn.layers.")[1].split(".")[0])
            for k in sd if "gvt_module.cross_attn.layers." in k
        }
        num_layers = max(layer_indices) + 1 if layer_indices else 2
        num_heads = 4
    use_gated = any("memory_gate" in k for k in sd)
    del ckpt_data

    predictor = build_sam2_video_predictor(sam2_config, ckpt_path=None, device=device)
    predictor = wrap_model_with_gvt(
        predictor, num_tokens=num_tokens, num_layers=num_layers,
        num_heads=num_heads, device=device, use_gated_attention=use_gated)
    load_gvt_checkpoint(predictor, ckpt_path, device=device)
    return predictor, True


# ----------------------------
# Prompted bidirectional propagation
# ----------------------------
def _add_prompt(predictor, state, frame_idx: int, obj_id: int,
                box: Optional[np.ndarray], points: Optional[np.ndarray],
                labels: Optional[np.ndarray]) -> None:
    """Add a user box and/or click points on the annotation frame."""
    kwargs = dict(inference_state=state, frame_idx=int(frame_idx), obj_id=int(obj_id))
    if box is not None:
        kwargs["box"] = np.asarray(box, dtype=np.float32)
    if points is not None and len(points) > 0:
        kwargs["points"] = np.asarray(points, dtype=np.float32)
        kwargs["labels"] = np.asarray(labels, dtype=np.int32)
    predictor.add_new_points_or_box(**kwargs)


def _propagate_half(predictor, vol0, v_slices, slice_cache, use_gvt,
                    box, points, labels, obj_id) -> Dict[int, np.ndarray]:
    """Init a sub-video, inject cached features, prompt frame 0, propagate, reset."""
    prepared = prepare_videos_once(predictor, vol0, v_slices, [v_slices[0]])
    try:
        state = init_state_compat(predictor, prepared.v1_source, prepared.v1_len,
                                  prefer_streaming=prepared.use_streaming)
        if slice_cache:
            inject_cached_features(state, v_slices, slice_cache)
        if use_gvt:
            compute_gvt_from_cached_features(predictor, state)
        _add_prompt(predictor, state, 0, obj_id, box, points, labels)
        pred = propagate_video_singleobj(predictor, state, obj_id=obj_id)
        predictor.reset_state(state)
    finally:
        prepared.cleanup()
    return pred


def run_prompt_inference(
    predictor, vol0: np.ndarray, prompt_idx: int,
    box: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    use_gvt: bool = False, obj_id: int = 1,
) -> np.ndarray:
    """Segment a 3D volume from a single prompt placed on slice `prompt_idx`.

    Args:
        predictor:  built (optionally GVT-wrapped) SAM2 video predictor.
        vol0:       [S, H, W] volume in the view orientation (slices along axis 0).
        prompt_idx: index of the slice the user drew on.
        box:        [x0, y0, x1, y1] in that slice's pixel coords, or None.
        points:     (P, 2) array of (x, y) click coords, or None.
        labels:     (P,) array of 1 (positive) / 0 (negative) per point.
        use_gvt:    whether the predictor is GVT-wrapped.

    Returns:
        [S, H, W] uint8 binary mask in the view orientation.
    """
    import torch

    if box is None and (points is None or len(points) == 0):
        raise ValueError("Provide a box and/or at least one point prompt.")

    N, H, W = vol0.shape
    prompt_idx = int(np.clip(prompt_idx, 0, N - 1))

    with torch.inference_mode():
        slice_cache = precompute_slice_features(predictor, vol0)
        v1, v2 = build_two_videos_indices(N, prompt_idx)

        p1 = _propagate_half(predictor, vol0, v1, slice_cache, use_gvt,
                             box, points, labels, obj_id)
        p2 = _propagate_half(predictor, vol0, v2, slice_cache, use_gvt,
                             box, points, labels, obj_id)

        pred = merge_two_videos_single_label(N, H, W, v1, v2, p1, p2)

    if use_gvt and _GVT_AVAILABLE:
        clear_gvt_for_predictor(predictor)
    return pred


# ----------------------------
# Saving
# ----------------------------
def save_mask(pred0: np.ndarray, view_axis: int, ref_sitk, out_path: str) -> str:
    """Write the [S,H,W] view-oriented prediction back to a NIfTI in the original
    geometry (spacing/origin/direction copied from the source image)."""
    import SimpleITK as sitk

    pred_zyx = np.moveaxis(pred0, 0, view_axis).astype(np.uint8)
    img = sitk.GetImageFromArray(pred_zyx)
    img.CopyInformation(ref_sitk)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))
    return out_path
