#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qualitative_study.py

Generates the SAMRI2 DICTA2026 qualitative overlay figures: for a fixed list of
20 evaluation cases (15 showcase + 5 failure, defined in qualitative_samples.json),
runs inference with a hardcoded candidate model set, renders GT/prediction contour
overlays on the per-case cut slice, and assembles two figure grids.

Candidate models (see MODEL_SPECS):
  - sam2_small    SAM2-Small, zero-shot (pretrained_ckpt/, no fine-tuning)
  - sam_med3d     SAM-Med3D                 \
  - medical_sam2  Medical-SAM2               } via comparison_test.py RUNNERS,
  - samed2        SAMed-2                   /  checkpoints under --weights-dir
  - baseline      SAMRI2-2dloss (fine-tuned, no GVT/3D loss)
  - gvt_v1        SAMRI2-GVT-v1 (GVT encoder only)
  - gvt_v2        SAMRI2-GVT-v2 (+ GVTReconDecoder)
  - gvt_v3        SAMRI2-GVT-v3 (+ TSDFDecoder, ours/best)
All SAMRI2 variants (baseline, gvt_v1-3) are fine-tuned from the SAM2-Small backbone.
Select/reorder columns with --models (default: all of the above, comparisons first,
ending with gvt_v3).

Reproduces the exact per-label cut-slice protocol from test_vis3d_multi.py
(precompute_slice_features + predict_per_label_semantic) for the SAM2-native models
(sam2_small, baseline, gvt_v1-3), and reuses comparison_test.py's runner functions
unchanged for sam_med3d / medical_sam2 / samed2, so rendered predictions match the
Dice scores reported in qualitative_samples.json.

Before any data loading or GPU work, every requested model's checkpoint is resolved
and checked for existence (preflight_check_models). By default the whole run aborts
if any model fails this check; pass --allow-partial to proceed with whichever models
are actually available instead.

=== Usage ===

python qualitative_study.py \
    --samples-json /path/to/qualitative_samples.json \
    --output-dir outputs/qualitative \
    --models sam2_small sam_med3d medical_sam2 samed2 baseline gvt_v1 gvt_v2 gvt_v3
    
Example:
python qualitative_study.py \
--samples-json /scratch/user/s4670484/qualitative_samples.json \
--output-dir /scratch/user/s4670484/SAMRI2_qualitative/qualitative \
--models sam2_small baseline gvt_v1 gvt_v2 gvt_v3 sam_med3d medical_sam2 samed2

The SAMRI2 checkpoint defaults point at /scratch/.../Model_save/samri2/ (samri2_2dloss.pth,
samri2_GVT_v1.pth, samri2_GVT_v2.pth, samri2_GVT_v3.pth) and the SAM2-Small init ckpt at
pretrained_ckpt/ — verify they exist before trusting figures, and override with
--sam2-small-ckpt / --baseline-ckpt / --gvt-v1-ckpt / etc. and --weights-dir
(for sam_med3d / medical_sam2 / samed2) if they don't match your run.

Every case's input image, resolved-label GT mask, and each model's full-resolution
3D prediction are also saved as .nii.gz (plus a case_meta.json) under
<output-dir>/pred_volumes/<dataset>__<img-filename>__<label>/, preserving the original NIfTI
spacing/origin/direction. This lets you redo overlay rendering (different contour
colors, crop margins, etc.) straight from disk via --from-volumes, with no GPU /
re-inference needed. Disable with --no-save-pred-volumes.

=== Visualisation methods (one folder each) ===

Inference runs once; every requested visualisation is then rendered from pred_volumes/
into its own <output-dir>/visN_*/ folder, so you can compare and pick. Select with
--vis-methods (default: all). All methods use matplotlib+skimage+scipy (no pyvista/VTK):
  vis1_cut_slice        single cut-slice overlay (+ combined showcase/failure grids)
  vis2_triplanar        axial/coronal/sagittal through the GT centroid (spacing-correct)
  vis3_montage          through-plane multi-slice strip along the propagation axis
  vis4_mip              max-intensity projections from 3 axes
  vis5_surface          3D surface mesh: GT (translucent) vs prediction
  vis6_surface_distance prediction surface colored by distance-to-GT (mm)
  vis7_error_volume     3D FP (orange) vs FN (purple) meshes
vis5-7 are 3D renders (slower, lower-fidelity than a dedicated renderer); empty/tiny
masks are skipped gracefully per case.

=== Multi-environment workflow ===

The comparison models each need their own conda env (sam_med3d / medical_sam2 /
samed2), while the SAM2-native models (sam2_small, baseline, gvt_v1-3) share one.
Run model subsets separately into the SAME --output-dir: each run appends its
predictions to pred_volumes/ and MERGES case_meta.json. Figures are always cumulative
— after each inference run the figure is rebuilt from ALL models accumulated in
pred_volumes/ so far (not just that run's subset), so the final env run already
produces the complete figure. --skip-figures just suppresses that per-run rendering
for the intermediate runs; a standalone --from-volumes pass can re-assemble anytime.

  # env: samri (SAM2-native models)
  python qualitative_study.py --samples-json ... --output-dir OUT --skip-figures \
      --models sam2_small baseline gvt_v1 gvt_v2 gvt_v3
  # env: sam_med3d
  python qualitative_study.py --samples-json ... --output-dir OUT --skip-figures --models sam_med3d
  # env: medical_sam2
  python qualitative_study.py --samples-json ... --output-dir OUT --skip-figures --models medical_sam2
  # env: samed2
  python qualitative_study.py --samples-json ... --output-dir OUT --skip-figures --models samed2
  # any env: assemble combined figures from accumulated volumes
  python qualitative_study.py --output-dir OUT --from-volumes OUT/pred_volumes
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import find_contours, marching_cubes

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2_video_predictor
from test_vis3d_multi import (
    read_3d_image,
    get_labels_from_mask,
    normalize_to_uint8,
    precompute_slice_features,
    predict_per_label_semantic,
    dice_3d,
)

_GVT_AVAILABLE = False
try:
    from gvt.gvt_integration import wrap_model_with_gvt, load_gvt_checkpoint
    _GVT_AVAILABLE = True
except ImportError:
    pass


# ----------------------------
# Defaults (from train_multi_gpus.sh save-path conventions — override via CLI if wrong)
# ----------------------------
DEFAULT_BASE_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"  # "s" config == small backbone
DEFAULT_SAM2_SMALL_CKPT = "/scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_small.pt"
DEFAULT_BASELINE_CKPT = "/scratch/user/s4670484/Model_save/samri2/samri2_2dloss.pth"
DEFAULT_GVT_V1_CKPT = "/scratch/user/s4670484/Model_save/samri2/samri2_GVT_v1.pth"
DEFAULT_GVT_V2_CKPT = "/scratch/user/s4670484/Model_save/samri2/samri2_GVT_v2.pth"
DEFAULT_GVT_V3_CKPT = "/scratch/user/s4670484/Model_save/samri2/samri2_GVT_v3.pth"
DEFAULT_COMPARISON_WEIGHTS_DIR = "/scratch/user/s4670484/comparison_ckpt/"
DEFAULT_COMPARISON_DIR = str(_THIS_DIR / "comparison")

# Models reused unchanged from comparison_test.py's RUNNERS registry. Each runner
# auto-discovers its single checkpoint under <weights-dir>/<subfolder>/ (see
# comparison_test._find_checkpoint), so no per-model ckpt default is needed here.
COMPARISON_MODEL_KEYS = ["sam_med3d", "medical_sam2", "samed2"]
# <weights-dir> subfolder name per comparison model (matches comparison_test.py's
# run_* functions exactly, including the "medica_sam2" typo).
COMPARISON_WEIGHTS_SUBFOLDER = {"sam_med3d": "sam_med3d", "medical_sam2": "medica_sam2", "samed2": "samed2"}
# CLI arg name that overrides the auto-discovered checkpoint, per comparison model.
COMPARISON_CKPT_OVERRIDE_ARG = {"medical_sam2": "medical_sam2_ckpt", "samed2": "samed2_ckpt"}

# kind="sam2": built via build_predictor()+run_sam2_model() (this file).
# kind="comparison": dispatched straight to comparison_test.RUNNERS[key] (unchanged).
MODEL_SPECS = {
    "sam2_small":   {"kind": "sam2", "title": "SAM2-Small (zero-shot)", "use_gvt": False,
                      "ckpt_arg": "sam2_small_ckpt"},
    "sam_med3d":    {"kind": "comparison", "title": "SAM-Med3D"},
    "medical_sam2": {"kind": "comparison", "title": "Medical-SAM2"},
    "samed2":       {"kind": "comparison", "title": "SAMed-2"},
    "baseline":     {"kind": "sam2", "title": "SAMRI2-2dloss", "use_gvt": False,
                      "ckpt_arg": "baseline_ckpt"},
    "gvt_v1":       {"kind": "sam2", "title": "SAMRI2-GVT-v1", "use_gvt": True,
                      "ckpt_arg": "gvt_v1_ckpt"},
    "gvt_v2":       {"kind": "sam2", "title": "SAMRI2-GVT-v2", "use_gvt": True,
                      "ckpt_arg": "gvt_v2_ckpt"},
    "gvt_v3":       {"kind": "sam2", "title": "SAMRI2-GVT-v3 (ours)", "use_gvt": True,
                      "ckpt_arg": "gvt_v3_ckpt"},
}
DEFAULT_MODEL_ORDER = [
    "sam2_small", "sam_med3d", "medical_sam2", "samed2", "baseline", "gvt_v1", "gvt_v2", "gvt_v3",
]


def order_model_keys(keys) -> List[str]:
    """Order an arbitrary set of model keys by the canonical figure-column order
    (DEFAULT_MODEL_ORDER), appending any unknown keys at the end. Used when merging
    models accumulated across separate per-environment runs."""
    keyset = set(keys)
    ordered = [k for k in DEFAULT_MODEL_ORDER if k in keyset]
    ordered += [k for k in keys if k not in DEFAULT_MODEL_ORDER and k not in ordered]
    return ordered


# ----------------------------
# Model building (mirrors test_vis3d_multi.py main()'s GVT auto-detect block)
# ----------------------------
def build_predictor(sam2_config: str, ckpt_path: str, device: str, use_gvt: bool):
    if not use_gvt:
        return build_sam2_video_predictor(sam2_config, ckpt_path, device=device)

    if not _GVT_AVAILABLE:
        raise RuntimeError("--use-gvt path requires the gvt/ package; not importable here.")

    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    gvt_cfg = ckpt_data.get("gvt_config", None)
    sd = ckpt_data.get("model", ckpt_data)
    if gvt_cfg is not None:
        num_tokens, num_layers, num_heads = gvt_cfg["num_tokens"], gvt_cfg["num_layers"], gvt_cfg["num_heads"]
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
        num_heads=num_heads, device=device, use_gated_attention=use_gated,
    )
    load_gvt_checkpoint(predictor, ckpt_path, device=device)
    return predictor


# ----------------------------
# Pre-flight: verify every requested model's checkpoint resolves before any
# data loading / GPU work happens.
# ----------------------------
def resolve_model_checkpoint(model_key: str, args) -> Path:
    spec = MODEL_SPECS[model_key]
    if spec["kind"] == "sam2":
        return Path(getattr(args, spec["ckpt_arg"]))

    override_arg = COMPARISON_CKPT_OVERRIDE_ARG.get(model_key)
    if override_arg and getattr(args, override_arg):
        return Path(getattr(args, override_arg))

    from comparison_test import _find_checkpoint  # local import: avoid requiring it for sam2-only runs
    subfolder = COMPARISON_WEIGHTS_SUBFOLDER[model_key]
    return _find_checkpoint(Path(args.weights_dir) / subfolder, spec["title"])


def preflight_check_models(model_keys: List[str], args, log) -> Tuple[List[str], List[Tuple[str, str]]]:
    log("=" * 70)
    log("Pre-flight: checking model availability")
    log("=" * 70)

    if any(MODEL_SPECS[k]["kind"] == "sam2" for k in model_keys):
        cfg_path = _THIS_DIR / "sam2" / args.sam2_config
        if cfg_path.is_file():
            log(f"  [OK]      sam2-config    -> {cfg_path}")
        else:
            log(f"  [MISSING] sam2-config    -> {cfg_path} (--sam2-config {args.sam2_config})")

    ok_keys: List[str] = []
    problems: List[Tuple[str, str]] = []
    for key in model_keys:
        try:
            ckpt_path = resolve_model_checkpoint(key, args)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
            log(f"  [OK]      {key:<14} -> {ckpt_path}")
            ok_keys.append(key)
        except Exception as e:
            log(f"  [MISSING] {key:<14} -> {e}")
            problems.append((key, str(e)))

    log("-" * 70)
    if problems:
        log(f"{len(problems)} of {len(model_keys)} model(s) failed pre-flight: "
            f"{[k for k, _ in problems]}")
    else:
        log(f"All {len(model_keys)} requested model(s) passed pre-flight checks.")
    log("=" * 70)
    return ok_keys, problems


# ----------------------------
# Label resolution
# ----------------------------
def resolve_label_id(
    gt_zyx: np.ndarray, label_field, dataset: str,
    cut_axis: Optional[int] = None, cut_idx: Optional[int] = None,
    voxel_pct: Optional[float] = None, log=None,
) -> int:
    """Resolve a case's target integer label ID.

    Fast paths: an integer-like label_field is used directly; a single-label mask
    resolves to its one non-zero ID. For a multi-label mask the named label is
    resolved by inverting how qualitative_samples.json was produced from
    test_vis3d_multi.py: the target label is the one whose per-label voxel fraction
    (count / volume.size) matches `voxel_pct` and whose argmax-foreground slice along
    `cut_axis` equals `cut_idx`. The final Dice-vs-expected table is the safety net —
    a wrong resolution shows up there as a MISMATCH.
    """
    try:
        return int(label_field)
    except (TypeError, ValueError):
        pass

    labels = get_labels_from_mask(gt_zyx)
    if not labels:
        raise RuntimeError(f"Mask for dataset '{dataset}' has no non-zero labels.")
    if len(labels) == 1:
        return labels[0]

    if cut_idx is None or voxel_pct is None:
        voxel_counts = {lb: int((gt_zyx == lb).sum()) for lb in labels}
        raise RuntimeError(
            f"Cannot resolve label '{label_field}' for dataset '{dataset}': mask has "
            f"{len(labels)} non-zero labels (voxel counts {voxel_counts}) and no "
            f"cut_idx/voxel_pct available to disambiguate."
        )

    gt0 = np.moveaxis(gt_zyx, cut_axis, 0)
    total = float(gt_zyx.size)
    frac = {lb: (gt_zyx == lb).sum() / total for lb in labels}
    argmax_slice = {lb: int(np.argmax((gt0 == lb).sum(axis=(1, 2)))) for lb in labels}

    # Prefer labels whose peak-foreground slice matches the recorded cut_idx; among
    # them (or all labels if none match) pick the closest voxel fraction.
    cut_matches = [lb for lb in labels if argmax_slice[lb] == cut_idx]
    pool = cut_matches if cut_matches else labels
    chosen = min(pool, key=lambda lb: abs(frac[lb] - voxel_pct))

    if log is not None:
        log(f"  [label] {dataset}/{label_field!r} -> id {chosen}  "
            f"(voxel_frac={frac[chosen]:.4f} vs json {voxel_pct:.4f}; "
            f"argmax_slice={argmax_slice[chosen]} vs cut_idx {cut_idx}; "
            f"{'cut-matched' if chosen in cut_matches else 'NO cut match'})")
    if not cut_matches and log is not None:
        log(f"  [WARN] {dataset}/{label_field!r}: no label's argmax slice == cut_idx "
            f"{cut_idx}; fell back to closest voxel_pct (id {chosen}). Check the Dice table.")
    return chosen


# ----------------------------
# Per-case SAM2 inference (mirrors test_vis3d_multi.predict_per_label_semantic)
# ----------------------------
def run_sam2_model(
    predictor, img_zyx: np.ndarray, gt_zyx: np.ndarray,
    label_id: int, cut_axis: int, cut_idx: int, prompt_mode: str,
) -> np.ndarray:
    vol0 = np.moveaxis(img_zyx, cut_axis, 0)
    gt0 = np.moveaxis(gt_zyx, cut_axis, 0)

    slice_cache = precompute_slice_features(predictor, vol0)
    pred0_sem, added = predict_per_label_semantic(
        predictor=predictor, vol0=vol0, gt0=gt0,
        slice_cache=slice_cache, prompt_mode=prompt_mode, labels=[label_id],
    )
    if not added:
        return np.zeros_like(gt_zyx, dtype=bool)

    fg0 = (gt0 == label_id).sum(axis=(1, 2))
    actual_cut = int(np.argmax(fg0)) if fg0.max() > 0 else None
    if actual_cut != cut_idx:
        print(
            f"[WARN] cut_idx mismatch for label {label_id}: json={cut_idx} "
            f"recomputed={actual_cut} (max-foreground slice differs from JSON)",
            file=sys.stderr,
        )

    pred_zyx = np.moveaxis(pred0_sem, 0, cut_axis)
    return pred_zyx == label_id


def run_comparison_model(model_key: str, case: Dict, label_id: int, spacing_zyx: Tuple, args) -> np.ndarray:
    """Dispatch straight to comparison_test.py's RUNNERS — sam_med3d / medical_sam2 /
    samed2 all share this (img_path, gt_path, label_id, img_zyx, gt_zyx, spacing_zyx,
    weights_dir, device, comparison_dir, tmp_dir, **_) signature; each runner caches
    its own model across calls and auto-discovers its checkpoint under
    <weights-dir>/<subfolder>/ (see comparison_test._find_checkpoint).
    """
    from comparison_test import RUNNERS  # local import: only needed for comparison models

    runner = RUNNERS[model_key]
    pred = runner(
        img_path=case["img_path"], gt_path=case["mask_path"], label_id=label_id,
        img_zyx=case["img_zyx"], gt_zyx=case["gt_zyx"], spacing_zyx=spacing_zyx,
        weights_dir=args.weights_dir, device=args.device,
        comparison_dir=args.comparison_dir, tmp_dir=args.tmp_dir,
        medical_sam2_ckpt=args.medical_sam2_ckpt, samed2_ckpt=args.samed2_ckpt,
    )
    return pred.astype(bool)


# ----------------------------
# Overlay rendering
# ----------------------------
# Colorblind-friendly overlay colors (avoid pure green/red).
# Yellow + blue is the most colorblind-safe high-contrast pairing.
GT_COLOR = "#FFFF00"      # yellow — ground truth
GT_RGB = (255 / 255, 255 / 255, 0 / 255)
PRED_COLOR = "#4FC3F7"    # light sky blue — prediction
PRED_RGB = (79 / 255, 195 / 255, 247 / 255)


def _expand_to_min(lo: int, hi: int, min_size: int, limit: int) -> Tuple[int, int]:
    """Grow [lo, hi) to at least min_size (centered), clamped to [0, limit]."""
    need = min_size - (hi - lo)
    if need > 0:
        lo = max(0, lo - (need + 1) // 2)
        hi = min(limit, lo + min_size)
        lo = max(0, hi - min_size)
    return lo, hi


def crop_bbox_from_mask(mask2d: np.ndarray, margin_frac: float = 0.2,
                        min_size: int = 8) -> Tuple[int, int, int, int]:
    H, W = mask2d.shape
    ys, xs = np.where(mask2d > 0)
    if ys.size == 0:
        return 0, H, 0, W
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    my = int((y1 - y0 + 1) * margin_frac)
    mx = int((x1 - x0 + 1) * margin_frac)
    y0, y1 = max(0, y0 - my), min(H, y1 + my + 1)
    x0, x1 = max(0, x0 - mx), min(W, x1 + mx + 1)
    # Tiny/thin structures can yield a 1-px-wide box; enforce a minimum so the
    # panel keeps context and contour extraction (needs >=2x2) stays valid.
    y0, y1 = _expand_to_min(y0, y1, min_size, H)
    x0, x1 = _expand_to_min(x0, x1, min_size, W)
    return y0, y1, x0, x1


def _safe_contours(arr2d: np.ndarray):
    """find_contours that returns [] for sub-2x2 inputs instead of raising."""
    if arr2d.shape[0] < 2 or arr2d.shape[1] < 2:
        return []
    return find_contours(arr2d.astype(float), 0.5)


def draw_panel(
    ax, img2d_uint8: np.ndarray, crop: Tuple[int, int, int, int],
    gt2d: Optional[np.ndarray] = None, pred2d: Optional[np.ndarray] = None,
    gt_fill_alpha: float = 0.0, gt_contour: bool = True,
    pred_fill_alpha: float = 0.25, title: Optional[str] = None,
    aspect: Optional[float] = None,
) -> None:
    y0, y1, x0, x1 = crop
    ax.imshow(img2d_uint8[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
    if aspect is not None:
        ax.set_aspect(aspect)

    # GT area shade — used for the combined "Input + GT" panel
    if gt2d is not None and gt_fill_alpha > 0:
        gt_crop = gt2d[y0:y1, x0:x1]
        fill = np.zeros((*gt_crop.shape, 4))
        fill[gt_crop > 0] = (*GT_RGB, gt_fill_alpha)
        ax.imshow(fill, interpolation="nearest")

    if pred2d is not None and pred_fill_alpha > 0:
        pred_crop = pred2d[y0:y1, x0:x1]
        fill = np.zeros((*pred_crop.shape, 4))
        fill[pred_crop > 0] = (*PRED_RGB, pred_fill_alpha)
        ax.imshow(fill, interpolation="nearest")

    if gt2d is not None and gt_contour:
        for c in _safe_contours(gt2d[y0:y1, x0:x1]):
            ax.plot(c[:, 1], c[:, 0], color=GT_COLOR, linewidth=2)

    if pred2d is not None:
        for c in _safe_contours(pred2d[y0:y1, x0:x1]):
            ax.plot(c[:, 1], c[:, 0], color=PRED_COLOR, linewidth=2)

    ax.set_xlim(0, x1 - x0)
    ax.set_ylim(y1 - y0, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("lightgray")
        spine.set_linewidth(0.5)
    if title is not None:
        ax.set_title(title, fontsize=8)


def sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s.strip())


# ----------------------------
# .nii.gz volume persistence (so figures can be redone without re-inference)
# ----------------------------
def _img_stem(img_path: str) -> str:
    """Filename of img_path with the medical-image extension stripped."""
    name = Path(img_path).name
    for ext in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def case_uid(case: Dict) -> str:
    """Stable unique id per JSON case. The JSON may contain several cases sharing the
    same (dataset, label) (different patients/instances), so the image filename — which
    encodes the patient/instance — is included to keep their volumes/metadata distinct.
    """
    return "__".join((
        sanitize_filename(case["dataset"]),
        sanitize_filename(_img_stem(case["img_path"])),
        sanitize_filename(str(case["label"])),
    ))


def case_dir_for(pred_volumes_dir: Path, case: Dict) -> Path:
    return pred_volumes_dir / case_uid(case)


def save_volume_like(arr: np.ndarray, ref_img: "sitk.Image", out_path: Path, dtype=np.uint8) -> None:
    img = sitk.GetImageFromArray(arr.astype(dtype))
    img.CopyInformation(ref_img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))


def load_case_from_volumes(case_dir: Path) -> Tuple[Dict, str, List[str]]:
    meta = json.loads((case_dir / "case_meta.json").read_text())
    case = meta["case"]
    cut_axis, cut_idx = case["cut_axis"], case["cut_idx"]

    img_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / "input.nii.gz")))
    gt_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / "gt.nii.gz"))) > 0

    record = {
        "case": case,
        "label_id": meta["label_id"],
        "computed_dice": meta["computed_dice"],
        "img_slice": normalize_to_uint8(np.take(img_zyx, cut_idx, axis=cut_axis)),
        "gt_slice": np.take(gt_zyx, cut_idx, axis=cut_axis),
        "pred_slices": {},
    }
    for model_key in meta["model_keys"]:
        pred_path = case_dir / f"{model_key}.nii.gz"
        if not pred_path.exists():
            continue
        pred_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))) > 0
        record["pred_slices"][model_key] = np.take(pred_zyx, cut_idx, axis=cut_axis)

    return record, meta["group"], meta["model_keys"]


# ----------------------------
# Per-case processing
# ----------------------------
def process_case(case: Dict, model_keys: List[str], log=None) -> Dict:
    img_zyx, img_sitk = read_3d_image(case["img_path"])
    gt_zyx, _ = read_3d_image(case["mask_path"])
    label_id = resolve_label_id(
        gt_zyx, case["label"], case["dataset"],
        cut_axis=case["cut_axis"], cut_idx=case["cut_idx"],
        voxel_pct=case.get("voxel_pct"), log=log,
    )

    spacing_xyz = img_sitk.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

    record = {
        "case": case,
        "label_id": label_id,
        "img_zyx": img_zyx,
        "gt_zyx": gt_zyx,
        "img_sitk": img_sitk,  # template for CopyInformation when saving pred volumes
        "spacing_zyx": spacing_zyx,
        "computed_dice": {},
        "pred_slices": {},
    }

    cut_axis, cut_idx = case["cut_axis"], case["cut_idx"]
    gt_bin_3d = (gt_zyx == label_id)
    record["img_slice"] = normalize_to_uint8(np.take(img_zyx, cut_idx, axis=cut_axis))
    record["gt_slice"] = np.take(gt_bin_3d, cut_idx, axis=cut_axis)
    record["gt_bin_3d_for_dice"] = gt_bin_3d  # consumed by model loop, dropped after
    return record


def finalize_case_memory(record: Dict) -> None:
    """Drop large 3D arrays once all models have been run for this case."""
    record.pop("img_zyx", None)
    record.pop("gt_zyx", None)
    record.pop("gt_bin_3d_for_dice", None)
    record.pop("img_sitk", None)


# ----------------------------
# Figure assembly
# ----------------------------
def _draw_case_columns(axes_row, rec: Dict, model_keys: List[str], show_titles: bool) -> None:
    """Draw one case's panels across a row of axes.

    Column 0 = "Input + GT": full original-resolution slice with the GT area shaded
    (no crop, no prediction). Remaining columns = per-model panels cropped to the GT
    bounding box, with green GT contour + red prediction contour.
    """
    columns = ["input_gt"] + model_keys
    full = rec["img_slice"].shape  # (H, W)
    full_crop = (0, full[0], 0, full[1])
    bbox_crop = crop_bbox_from_mask(rec["gt_slice"], margin_frac=0.2)

    for j, col in enumerate(columns):
        ax = axes_row[j]
        if col == "input_gt":
            draw_panel(ax, rec["img_slice"], full_crop, gt2d=rec["gt_slice"],
                       gt_fill_alpha=0.35, gt_contour=False,
                       title="Input + GT" if show_titles else None)
        else:
            dsc = rec["computed_dice"].get(col, float("nan"))
            title = (f"{MODEL_SPECS[col]['title']}\nDice={dsc:.2f}"
                     if show_titles else f"Dice={dsc:.2f}")
            draw_panel(ax, rec["img_slice"], bbox_crop, gt2d=rec["gt_slice"],
                       pred2d=rec["pred_slices"].get(col), title=title)


def render_grid(records: List[Dict], model_keys: List[str], out_path_stem: Path, dpi: int) -> None:
    ncols = 1 + len(model_keys)  # "Input + GT" + one column per model
    nrows = len(records)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    if nrows == 1:
        axes = axes[None, :]

    for i, rec in enumerate(records):
        case = rec["case"]
        _draw_case_columns(axes[i], rec, model_keys, show_titles=(i == 0))
        axes[i, 0].text(
            -0.25, 0.5, f"{case['dataset']} — {case['label']}",
            transform=axes[i, 0].transAxes, ha="right", va="center",
            rotation=90, fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(str(out_path_stem) + ".pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_path_stem) + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_single_case(rec: Dict, model_keys: List[str], out_path: Path, dpi: int) -> None:
    ncols = 1 + len(model_keys)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.2, 2.6))
    _draw_case_columns(axes, rec, model_keys, show_titles=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Multi-method 3D visualisation
# (one folder per method under <output-dir>; all driven from pred_volumes/, so
#  inference runs once and every method is rendered from the saved .nii.gz)
# ----------------------------
PLANE_NAMES = {0: "Axial", 1: "Coronal", 2: "Sagittal"}


def load_case_volumes(case_dir: Path, model_keys: List[str]) -> Optional[Dict]:
    """Load a case's full 3D volumes + spacing from its pred_volumes/<case>/ dir."""
    meta_path = case_dir / "case_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    case = meta["case"]
    img_sitk = sitk.ReadImage(str(case_dir / "input.nii.gz"))
    img = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / "gt.nii.gz"))) > 0
    sp = img_sitk.GetSpacing()  # (x, y, z)
    spacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
    preds = {}
    for k in model_keys:
        p = case_dir / f"{k}.nii.gz"
        if p.exists():
            preds[k] = sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0
    return {
        "meta": meta, "case": case, "img": img, "gt": gt, "preds": preds,
        "spacing_zyx": spacing_zyx,
        "cut_axis": int(case["cut_axis"]), "cut_idx": int(case["cut_idx"]),
        "computed_dice": meta.get("computed_dice", {}),
    }


def _plane_aspect(spacing_zyx: Tuple[float, float, float], axis: int) -> float:
    """Physically correct imshow aspect for the plane perpendicular to `axis`."""
    other = [a for a in (0, 1, 2) if a != axis]
    col = spacing_zyx[other[1]]
    return spacing_zyx[other[0]] / col if col > 0 else 1.0


def _gt_centroid(gt3d: np.ndarray) -> Tuple[int, int, int]:
    pts = np.argwhere(gt3d)
    if pts.size == 0:
        return tuple(s // 2 for s in gt3d.shape)  # type: ignore
    return tuple(int(round(v)) for v in pts.mean(axis=0))  # type: ignore


def _row_label(ax, text: str) -> None:
    ax.text(-0.25, 0.5, text, transform=ax.transAxes, ha="right", va="center",
            rotation=90, fontsize=8)


def _model_title(V: Dict, key: str, show: bool, with_dice: bool = True) -> Optional[str]:
    if not show:
        return None
    if with_dice:
        return f"{MODEL_SPECS[key]['title']}\nDice={V['computed_dice'].get(key, float('nan')):.2f}"
    return MODEL_SPECS[key]["title"]


def _slice_row(axes_row, V: Dict, model_keys: List[str], axis: int, idx: int,
               show_titles: bool, crop_models: bool = True, label_suffix: str = "") -> None:
    """Draw one row: Input+GT (full) then each model (cropped) for slice `idx` along `axis`."""
    img2d = normalize_to_uint8(np.take(V["img"], idx, axis=axis))
    gt2d = np.take(V["gt"], idx, axis=axis)
    asp = _plane_aspect(V["spacing_zyx"], axis)
    full = (0, img2d.shape[0], 0, img2d.shape[1])
    bbox = crop_bbox_from_mask(gt2d) if (crop_models and gt2d.any()) else full
    cols = ["input_gt"] + model_keys
    for j, c in enumerate(cols):
        ax = axes_row[j]
        if c == "input_gt":
            draw_panel(ax, img2d, full, gt2d=gt2d, gt_fill_alpha=0.35, gt_contour=False,
                       title=("Input + GT" + label_suffix) if show_titles else None, aspect=asp)
        else:
            pr = np.take(V["preds"][c], idx, axis=axis) if c in V["preds"] else None
            draw_panel(ax, img2d, bbox, gt2d=gt2d, pred2d=pr,
                       title=_model_title(V, c, show_titles), aspect=asp)


def vis_cut_slice(out_path: Path, V: Dict, model_keys: List[str], dpi: int) -> None:
    ncols = 1 + len(model_keys)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.2, 2.6))
    _slice_row(axes, V, model_keys, V["cut_axis"], V["cut_idx"], show_titles=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def vis_triplanar(out_path: Path, V: Dict, model_keys: List[str], dpi: int) -> None:
    cz, cy, cx = _gt_centroid(V["gt"])
    centers = {0: cz, 1: cy, 2: cx}
    ncols = 1 + len(model_keys)
    fig, axes = plt.subplots(3, ncols, figsize=(ncols * 2.2, 3 * 2.4))
    for r, axis in enumerate((0, 1, 2)):
        _slice_row(axes[r], V, model_keys, axis, centers[axis], show_titles=(r == 0))
        _row_label(axes[r, 0], PLANE_NAMES[axis])
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def vis_montage(out_path: Path, V: Dict, model_keys: List[str], dpi: int, k: int = 6) -> None:
    axis = V["cut_axis"]
    other = tuple(a for a in (0, 1, 2) if a != axis)
    present = np.where(V["gt"].any(axis=other))[0]
    if present.size == 0:
        present = np.array([V["cut_idx"]])
    sel = sorted(set(np.linspace(present.min(), present.max(),
                                 num=min(k, present.size)).round().astype(int).tolist()))
    ncols = 1 + len(model_keys)
    fig, axes = plt.subplots(len(sel), ncols, figsize=(ncols * 2.0, len(sel) * 2.0))
    if len(sel) == 1:
        axes = axes[None, :]
    for r, idx in enumerate(sel):
        _slice_row(axes[r], V, model_keys, axis, idx, show_titles=(r == 0))
        _row_label(axes[r, 0], f"{PLANE_NAMES[axis][:3]} {idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def vis_mip(out_path: Path, V: Dict, model_keys: List[str], dpi: int) -> None:
    ncols = 1 + len(model_keys)
    fig, axes = plt.subplots(3, ncols, figsize=(ncols * 2.2, 3 * 2.4))
    cols = ["input_gt"] + model_keys
    for r, axis in enumerate((0, 1, 2)):
        img2d = normalize_to_uint8(V["img"].max(axis=axis))
        gt2d = V["gt"].any(axis=axis)
        asp = _plane_aspect(V["spacing_zyx"], axis)
        full = (0, img2d.shape[0], 0, img2d.shape[1])
        for j, c in enumerate(cols):
            ax = axes[r, j]
            show = (r == 0)
            if c == "input_gt":
                draw_panel(ax, img2d, full, gt2d=gt2d, gt_fill_alpha=0.35, gt_contour=False,
                           title="Input + GT (MIP)" if show else None, aspect=asp)
            else:
                pr = V["preds"][c].any(axis=axis) if c in V["preds"] else None
                draw_panel(ax, img2d, full, gt2d=gt2d, pred2d=pr,
                           title=_model_title(V, c, show), aspect=asp)
        _row_label(axes[r, 0], f"{PLANE_NAMES[axis]} MIP")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _mesh_verts_faces(mask3d: np.ndarray, spacing_zyx: Tuple[float, float, float]):
    """Marching-cubes surface of a binary mask (padded so the surface closes).
    Returns (verts, faces) in physical coords, or None if the mask is too small."""
    if int(mask3d.sum()) < 27:
        return None
    try:
        verts, faces, _, _ = marching_cubes(
            np.pad(mask3d.astype(np.float32), 1), level=0.5, spacing=spacing_zyx)
    except (RuntimeError, ValueError):
        return None
    return verts, faces


def _add_mesh(ax, mask3d, spacing_zyx, color, alpha) -> Optional[np.ndarray]:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mf = _mesh_verts_faces(mask3d, spacing_zyx)
    if mf is None:
        return None
    verts, faces = mf
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    return verts


def _finalize_3d(ax, all_verts: List[np.ndarray]) -> None:
    ax.set_axis_off()
    if not all_verts:
        return
    v = np.concatenate(all_verts, axis=0)
    mins, maxs = v.min(0), v.max(0)
    ax.set_box_aspect(tuple((maxs - mins) + 1e-6))
    for setlim, lo, hi in ((ax.set_xlim, mins[0], maxs[0]),
                           (ax.set_ylim, mins[1], maxs[1]),
                           (ax.set_zlim, mins[2], maxs[2])):
        setlim(lo, hi)
    ax.view_init(elev=20, azim=-60)


def vis_surface(out_path: Path, V: Dict, model_keys: List[str], dpi: int) -> None:
    cols = ["GT"] + model_keys
    fig = plt.figure(figsize=(len(cols) * 2.6, 3.0))
    sp = V["spacing_zyx"]
    for j, c in enumerate(cols):
        ax = fig.add_subplot(1, len(cols), j + 1, projection="3d")
        verts_all = []
        if c == "GT":
            v = _add_mesh(ax, V["gt"], sp, GT_RGB, alpha=0.9)
            if v is not None:
                verts_all.append(v)
            ax.set_title("GT", fontsize=8)
        else:
            vg = _add_mesh(ax, V["gt"], sp, GT_RGB, alpha=0.18)
            if vg is not None:
                verts_all.append(vg)
            if c in V["preds"]:
                vp = _add_mesh(ax, V["preds"][c], sp, PRED_RGB, alpha=0.85)
                if vp is not None:
                    verts_all.append(vp)
            ax.set_title(_model_title(V, c, True), fontsize=8)
        _finalize_3d(ax, verts_all)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def vis_surface_distance(out_path: Path, V: Dict, model_keys: List[str], dpi: int,
                         vmax_mm: float = 10.0) -> None:
    from scipy.ndimage import distance_transform_edt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    sp = np.asarray(V["spacing_zyx"], dtype=np.float32)
    gt = V["gt"]
    # Unsigned physical distance to the GT surface (mm).
    dist = np.minimum(distance_transform_edt(~gt, sampling=sp),
                      distance_transform_edt(gt, sampling=sp))
    norm = Normalize(vmin=0.0, vmax=vmax_mm)
    cmap = colormaps["inferno"]

    fig = plt.figure(figsize=(len(model_keys) * 2.6 + 0.6, 3.0))
    for j, c in enumerate(model_keys):
        ax = fig.add_subplot(1, len(model_keys), j + 1, projection="3d")
        mf = _mesh_verts_faces(V["preds"].get(c, np.zeros_like(gt)), V["spacing_zyx"])
        if mf is None:
            ax.set_axis_off()
            ax.set_title(_model_title(V, c, True), fontsize=8)
            continue
        verts, faces = mf
        vox = np.clip(np.round(verts / sp).astype(int) - 1, 0, np.array(gt.shape) - 1)
        dvals = dist[vox[:, 0], vox[:, 1], vox[:, 2]]
        face_d = dvals[faces].mean(axis=1)
        mesh = Poly3DCollection(verts[faces])
        mesh.set_facecolor(cmap(norm(face_d)))
        mesh.set_edgecolor("none")
        ax.add_collection3d(mesh)
        _finalize_3d(ax, [verts])
        ax.set_title(_model_title(V, c, True), fontsize=8)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, fraction=0.015, pad=0.01,
                 label="dist to GT surface (mm)")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# Error-volume colors: orange = false positive (over-seg), purple = false negative (under-seg).
FP_RGB = (251 / 255, 140 / 255, 0 / 255)
FN_RGB = (142 / 255, 36 / 255, 170 / 255)


def vis_error_volume(out_path: Path, V: Dict, model_keys: List[str], dpi: int) -> None:
    sp = V["spacing_zyx"]
    gt = V["gt"]
    fig = plt.figure(figsize=(len(model_keys) * 2.6, 3.0))
    for j, c in enumerate(model_keys):
        ax = fig.add_subplot(1, len(model_keys), j + 1, projection="3d")
        verts_all = []
        if c in V["preds"]:
            pr = V["preds"][c]
            for mask, color, alpha in ((pr & gt, (0.6, 0.6, 0.6), 0.12),
                                       (pr & ~gt, FP_RGB, 0.85),
                                       (~pr & gt, FN_RGB, 0.85)):
                v = _add_mesh(ax, mask, sp, color, alpha)
                if v is not None:
                    verts_all.append(v)
        _finalize_3d(ax, verts_all)
        ax.set_title(_model_title(V, c, True), fontsize=8)
    fig.suptitle("orange = false positive   ·   purple = false negative", fontsize=8, y=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# Ordered registry: folder name -> renderer. "visN_" prefix preserves a stable order.
VIS_METHODS: "Dict[str, Callable]" = {
    "vis1_cut_slice": vis_cut_slice,
    "vis2_triplanar": vis_triplanar,
    "vis3_montage": vis_montage,
    "vis4_mip": vis_mip,
    "vis5_surface": vis_surface,
    "vis6_surface_distance": vis_surface_distance,
    "vis7_error_volume": vis_error_volume,
}
VIS_NUM_TO_FOLDER = {f"vis{i + 1}": name for i, name in enumerate(VIS_METHODS)}


def resolve_vis_methods(tokens: List[str]) -> List[str]:
    """Map CLI tokens ('all' | 'vis1'.. | full folder name) to ordered folder names."""
    if not tokens or "all" in tokens:
        return list(VIS_METHODS)
    chosen = []
    for t in tokens:
        folder = VIS_NUM_TO_FOLDER.get(t, t)
        if folder in VIS_METHODS and folder not in chosen:
            chosen.append(folder)
    return [m for m in VIS_METHODS if m in chosen]  # keep canonical order


def generate_visualisations(pred_volumes_dir: Path, out_dir: Path,
                            records_by_group: Dict[str, List[Dict]], model_keys: List[str],
                            method_folders: List[str], dpi: int, log) -> None:
    """Render every requested method into out_dir/<folder>/, loading each case's
    volumes from pred_volumes/ once and feeding all enabled per-case renderers."""
    for folder in method_folders:
        (out_dir / folder).mkdir(parents=True, exist_ok=True)

    # vis1 also gets the combined showcase/failure grids (the multi-case overview).
    if "vis1_cut_slice" in method_folders:
        sub = out_dir / "vis1_cut_slice"
        try:
            render_grid(records_by_group["showcase"], model_keys, sub / "figure_showcase", dpi)
            render_grid(records_by_group["failure"], model_keys, sub / "figure_failure", dpi)
        except Exception as e:
            log(f"[WARN] vis1 grid rendering failed: {e}")

    case_dirs = sorted(p for p in Path(pred_volumes_dir).iterdir() if p.is_dir())
    log(f"\nRendering {len(method_folders)} visualisation method(s) for {len(case_dirs)} "
        f"case(s): {method_folders}")
    for cd in case_dirs:
        V = load_case_volumes(cd, model_keys)
        if V is None:
            continue
        for folder in method_folders:
            try:
                VIS_METHODS[folder](out_dir / folder / f"{cd.name}.png", V, model_keys, dpi)
            except Exception as e:
                log(f"[WARN] {folder} failed for {cd.name}: {e}")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SAMRI2 qualitative overlay figure generator")
    p.add_argument("--samples-json", default=None,
                    help="Path to qualitative_samples.json. Required unless --from-volumes is set.")
    p.add_argument("--output-dir", default="outputs/qualitative")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt-mode", choices=["box", "mask"], default="box")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--dice-tolerance", type=float, default=0.03)
    p.add_argument("--vis-methods", nargs="+", default=["all"],
                    choices=["all"] + list(VIS_NUM_TO_FOLDER) + list(VIS_METHODS),
                    help="Which visualisation method(s) to render, each into its own "
                         "<output-dir>/visN_* folder. 'all' (default) renders every method; "
                         "or pass e.g. vis1 vis2 vis5. vis5-7 are 3D renders (slower).")

    p.add_argument("--models", nargs="+", default=DEFAULT_MODEL_ORDER, choices=list(MODEL_SPECS),
                    help="Which model columns to run, in order (default: all candidates, "
                         "comparisons first, ending with gvt_v3).")

    p.add_argument("--sam2-config", default=DEFAULT_BASE_SAM2_CONFIG,
                    help="Config shared by all SAM2-native models (all fine-tuned from "
                         "the small backbone, per train_multi_gpus.sh SAM2_CONFIG).")
    p.add_argument("--sam2-small-ckpt", default=DEFAULT_SAM2_SMALL_CKPT)
    p.add_argument("--baseline-ckpt", default=DEFAULT_BASELINE_CKPT)
    p.add_argument("--gvt-v1-ckpt", default=DEFAULT_GVT_V1_CKPT)
    p.add_argument("--gvt-v2-ckpt", default=DEFAULT_GVT_V2_CKPT)
    p.add_argument("--gvt-v3-ckpt", default=DEFAULT_GVT_V3_CKPT)

    p.add_argument("--weights-dir", default=DEFAULT_COMPARISON_WEIGHTS_DIR,
                    help="Root dir with per-model weight sub-folders for sam_med3d / "
                         "medica_sam2 / samed2 (see comparison_test.py docstring).")
    p.add_argument("--medical-sam2-ckpt", default=None, help="Override <weights-dir>/medica_sam2/.")
    p.add_argument("--samed2-ckpt", default=None, help="Override <weights-dir>/samed2/.")
    p.add_argument("--comparison-dir", default=DEFAULT_COMPARISON_DIR)
    p.add_argument("--tmp-dir", default=str(Path(tempfile.gettempdir()) / "samri2_qualitative"))

    p.add_argument("--no-save-pred-volumes", action="store_true",
                    help="Don't save input/GT/prediction volumes as .nii.gz (saved by default).")
    p.add_argument("--pred-volumes-dir", default=None,
                    help="Where to save .nii.gz volumes (default: <output-dir>/pred_volumes).")
    p.add_argument("--from-volumes", default=None,
                    help="Skip inference entirely; rebuild figures from a previous run's "
                         "pred_volumes/ directory (e.g. to change contour colors/crop margin).")

    p.add_argument("--allow-partial", action="store_true",
                    help="If some --models fail the pre-flight checkpoint check, proceed with "
                         "whichever models are available instead of aborting (default: abort).")
    p.add_argument("--skip-figures", action="store_true",
                    help="Run inference and save .nii.gz volumes + merge case_meta.json, but "
                         "don't render figures. Use for per-environment subset runs, then do a "
                         "final --from-volumes pass to assemble the combined figure.")
    return p.parse_args()


def pred_volumes_dir_for(args) -> Path:
    """Resolve where .nii.gz volumes live: --pred-volumes-dir, else <output-dir>/pred_volumes."""
    return Path(args.pred_volumes_dir) if args.pred_volumes_dir else Path(args.output_dir) / "pred_volumes"


def run_inference_pipeline(args, log) -> Tuple[Dict[str, List[Dict]], List[str]]:
    if not args.samples_json:
        raise SystemExit("--samples-json is required unless --from-volumes is set.")

    save_volumes = not args.no_save_pred_volumes
    pred_volumes_dir = pred_volumes_dir_for(args)

    model_keys = list(args.models)
    if any(MODEL_SPECS[k]["kind"] == "comparison" for k in model_keys):
        try:
            import comparison_test  # noqa: F401  (import-only sanity check)
        except Exception as e:
            comparison_keys = [k for k in model_keys if MODEL_SPECS[k]["kind"] == "comparison"]
            print(f"[WARN] comparison_test could not be imported ({e}); omitting "
                  f"{comparison_keys} from --models.", file=sys.stderr)
            model_keys = [k for k in model_keys if MODEL_SPECS[k]["kind"] != "comparison"]

    # Pre-flight: resolve every model's checkpoint before touching the dataset or GPU.
    ok_keys, problems = preflight_check_models(model_keys, args, log)
    if problems:
        if not args.allow_partial:
            raise SystemExit(
                f"Pre-flight check failed for {[k for k, _ in problems]}. Fix the paths above "
                f"(or pass the matching --*-ckpt / --weights-dir override), or pass "
                f"--allow-partial to proceed with just the available models."
            )
        model_keys = ok_keys
        if not model_keys:
            raise SystemExit("--allow-partial set but no requested models passed pre-flight checks.")

    with open(args.samples_json) as f:
        samples = json.load(f)

    log("=" * 70)
    log("SAMRI2 qualitative figure generation")
    log(f"  Models (in order)             : {model_keys}")
    log(f"  SAM2 config (all SAM2-native) : {args.sam2_config}")
    for key in model_keys:
        spec = MODEL_SPECS[key]
        if spec["kind"] == "sam2":
            log(f"    {key:<14} ckpt : {getattr(args, spec['ckpt_arg'])}")
    if any(MODEL_SPECS[k]["kind"] == "comparison" for k in model_keys):
        log(f"  Comparison weights dir        : {args.weights_dir}")
    if save_volumes:
        log(f"  Saving .nii.gz volumes to     : {pred_volumes_dir}")
    log("=" * 70)

    # Pre-load raw case data once; reused across all models. Save input + GT
    # volumes immediately (img_sitk is freshest here).
    figure_groups = {"showcase": samples["best"], "failure": samples["failure"]}
    records_by_group: Dict[str, List[Dict]] = {}
    for group_name, cases in figure_groups.items():
        recs = []
        for case in cases:
            try:
                rec = process_case(case, model_keys, log=log)
                if save_volumes:
                    cdir = case_dir_for(pred_volumes_dir, case)
                    save_volume_like(rec["img_zyx"], rec["img_sitk"], cdir / "input.nii.gz", dtype=np.float32)
                    save_volume_like(rec["gt_bin_3d_for_dice"], rec["img_sitk"], cdir / "gt.nii.gz")
                recs.append(rec)
            except Exception as e:
                log(f"[ERROR] {group_name}/{case['dataset']} ({case['label']}): "
                    f"failed to load/resolve — {e}")
                traceback.print_exc()
        records_by_group[group_name] = recs

    rec_groups = {case_uid(rec["case"]): g
                  for g, recs in records_by_group.items() for rec in recs}
    all_records = records_by_group["showcase"] + records_by_group["failure"]

    # Model-major loop: at most one SAM2-native predictor in memory at a time;
    # comparison models manage their own (cached) model internally.
    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        predictor = None
        if spec["kind"] == "sam2":
            ckpt_path = getattr(args, spec["ckpt_arg"])
            log(f"\n[{model_key}] building predictor from {ckpt_path} (use_gvt={spec['use_gvt']})")
            try:
                predictor = build_predictor(args.sam2_config, ckpt_path, args.device, spec["use_gvt"])
            except Exception as e:
                log(f"[ERROR] could not build predictor for '{model_key}': {e}")
                traceback.print_exc()
                continue
        else:
            log(f"\n[{model_key}] running via comparison_test.RUNNERS['{model_key}'] "
                f"(weights_dir={args.weights_dir})")

        for rec in all_records:
            case = rec["case"]
            try:
                if spec["kind"] == "sam2":
                    pred_3d = run_sam2_model(
                        predictor, rec["img_zyx"], rec["gt_zyx"], rec["label_id"],
                        case["cut_axis"], case["cut_idx"], args.prompt_mode,
                    )
                else:
                    pred_3d = run_comparison_model(
                        model_key, {**case, "img_zyx": rec["img_zyx"], "gt_zyx": rec["gt_zyx"]},
                        rec["label_id"], rec["spacing_zyx"], args,
                    )
                rec["computed_dice"][model_key] = dice_3d(
                    pred_3d.astype(np.uint8), rec["gt_bin_3d_for_dice"].astype(np.uint8)
                )
                rec["pred_slices"][model_key] = np.take(pred_3d, case["cut_idx"], axis=case["cut_axis"])
                if save_volumes:
                    cdir = case_dir_for(pred_volumes_dir, case)
                    save_volume_like(pred_3d, rec["img_sitk"], cdir / f"{model_key}.nii.gz")
            except Exception as e:
                log(f"[ERROR] {model_key} failed on {case['dataset']} ({case['label']}): {e}")
                traceback.print_exc()
                rec["computed_dice"][model_key] = float("nan")
                rec["pred_slices"][model_key] = np.zeros_like(rec["gt_slice"])

        if predictor is not None:
            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # case_meta.json (written before dropping img_sitk/3D arrays) so a later
    # --from-volumes run can rebuild figures without touching the dataset.
    # MERGE with any existing meta from a previous run so that running model
    # subsets in separate conda envs accumulates into one pred_volumes/ dir
    # (e.g. SAM2-native models in `samri`, then sam_med3d/medical_sam2/samed2
    # each in their own env) instead of overwriting each other.
    if save_volumes:
        for rec in all_records:
            case = rec["case"]
            cdir = case_dir_for(pred_volumes_dir, case)
            meta_path = cdir / "case_meta.json"
            merged_keys: List[str] = []
            merged_dice: Dict = {}
            if meta_path.exists():
                try:
                    prev = json.loads(meta_path.read_text())
                    merged_keys += list(prev.get("model_keys", []))
                    merged_dice.update(prev.get("computed_dice", {}))
                except Exception as e:
                    log(f"[WARN] could not merge existing {meta_path}: {e}")
            merged_keys += [k for k in model_keys if k not in merged_keys]
            merged_dice.update(rec["computed_dice"])  # this run's results win on overlap
            meta = {
                "case": case,
                "label_id": rec["label_id"],
                "group": rec_groups[case_uid(case)],
                "model_keys": order_model_keys(merged_keys),
                "computed_dice": merged_dice,
            }
            meta_path.write_text(json.dumps(meta, indent=2))

    for rec in all_records:
        finalize_case_memory(rec)

    return records_by_group, model_keys


def run_from_volumes_pipeline(pred_volumes_dir: Path, log) -> Tuple[Dict[str, List[Dict]], List[str]]:
    pred_volumes_dir = Path(pred_volumes_dir)
    log(f"Rebuilding figures from saved volumes in: {pred_volumes_dir}")

    records_by_group: Dict[str, List[Dict]] = {"showcase": [], "failure": []}
    all_keys: List[str] = []
    found_any = False
    for case_dir in sorted(p for p in pred_volumes_dir.iterdir() if p.is_dir()):
        meta_path = case_dir / "case_meta.json"
        if not meta_path.exists():
            continue
        try:
            rec, group, mk = load_case_from_volumes(case_dir)
        except Exception as e:
            log(f"[ERROR] failed to load {case_dir.name} from volumes: {e}")
            traceback.print_exc()
            continue
        found_any = True
        all_keys += [k for k in mk if k not in all_keys]
        records_by_group.setdefault(group, []).append(rec)

    if not found_any:
        raise SystemExit(f"No valid case_meta.json found under {pred_volumes_dir}")
    # Only keep models that actually have a saved volume for at least one case.
    model_keys = order_model_keys(all_keys)
    log(f"  Combined models found across runs: {model_keys}")
    return records_by_group, model_keys


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    if args.from_volumes:
        records_by_group, model_keys = run_from_volumes_pipeline(Path(args.from_volumes), log)
    else:
        records_by_group, model_keys = run_inference_pipeline(args, log)
        # Figures must reflect ALL models accumulated in pred_volumes/ (across separate
        # per-environment runs), not just this run's in-memory subset. So when volumes
        # were saved and we're about to render, reload the full merged set from disk.
        if not args.skip_figures and not args.no_save_pred_volumes:
            log("\nReloading all accumulated models from pred_volumes/ for cumulative figures...")
            records_by_group, model_keys = run_from_volumes_pipeline(pred_volumes_dir_for(args), log)

    pvd = Path(args.from_volumes) if args.from_volumes else pred_volumes_dir_for(args)
    volumes_available = bool(args.from_volumes) or not args.no_save_pred_volumes

    if args.skip_figures and not args.from_volumes:
        log("\n--skip-figures set: skipping figure rendering (volumes + case_meta.json saved). "
            "Run a final --from-volumes pass to assemble the combined figures.")
    elif not volumes_available:
        log("\n[WARN] --no-save-pred-volumes leaves no 3D volumes on disk; the multi-method "
            "visualisations need them. Skipping figures (use without --no-save-pred-volumes).")
    else:
        methods = resolve_vis_methods(args.vis_methods)
        generate_visualisations(pvd, out_dir, records_by_group, model_keys, methods,
                                dpi=args.dpi, log=log)

    # Verification table
    log("\n" + "=" * 70)
    log("Verification: computed vs expected Dice")
    log(f"{'case':<45}{'model':<14}{'computed':>10}{'expected':>10}{'|delta|':>10}  flag")
    log("-" * 100)
    any_mismatch = False
    for recs in records_by_group.values():
        for rec in recs:
            case = rec["case"]
            case_id = f"{case['dataset']}/{case['label']}"
            for model_key in model_keys:
                computed = rec["computed_dice"].get(model_key, float("nan"))
                expected = case["dice"].get(model_key, float("nan"))
                delta = abs(computed - expected) if np.isfinite(computed) and np.isfinite(expected) else float("nan")
                flag = "MISMATCH" if (np.isfinite(delta) and delta > args.dice_tolerance) else ""
                if flag:
                    any_mismatch = True
                log(f"{case_id:<45}{model_key:<14}{computed:>10.3f}{expected:>10.3f}{delta:>10.3f}  {flag}")
    log("-" * 100)
    if any_mismatch:
        log(f"Some cases exceed the |delta| > {args.dice_tolerance} tolerance — "
            f"check checkpoint paths, label resolution, and preprocessing before "
            f"trusting the figures.")
    else:
        log("All computed Dice scores are within tolerance of expected_dice.")

    with open(out_dir / "run_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    if args.skip_figures and not args.from_volumes:
        print(f"\nSaved prediction volumes + run_log.txt under: {out_dir} "
              f"(figures skipped; assemble with --from-volumes)")
    else:
        print(f"\nSaved figures + run_log.txt to: {out_dir}")


if __name__ == "__main__":
    main()
