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
--models sam2_small sam_med3d medical_sam2 samed2 baseline gvt_v1 gvt_v2 gvt_v3

The checkpoint defaults below come from train_multi_gpus.sh save-path conventions
(SAM2-Small init ckpt is exact; GVT v1/v2/v3/baseline final-checkpoint filenames are
the scripts' own "*_final.pth" naming) — verify they exist before trusting figures,
and override with --sam2-small-ckpt / --baseline-ckpt / --gvt-v1-ckpt / etc. and
--weights-dir (for sam_med3d / medical_sam2 / samed2) if they don't match your run.

Every case's input image, resolved-label GT mask, and each model's full-resolution
3D prediction are also saved as .nii.gz (plus a case_meta.json) under
<output-dir>/pred_volumes/<dataset>__<label>/, preserving the original NIfTI
spacing/origin/direction. This lets you redo overlay rendering (different contour
colors, crop margins, etc.) straight from disk via --from-volumes, with no GPU /
re-inference needed. Disable with --no-save-pred-volumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import find_contours

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
DEFAULT_BASELINE_CKPT = "/scratch/user/s4670484/Model_save/samri2_2dloss_all3d_emb/sam2_emb3d_final.pth"
DEFAULT_GVT_V1_CKPT = "/scratch/user/s4670484/Model_save/samri2_gvt1_K16_perlabel/sam2_gvt_tsdf_final.pth"
DEFAULT_GVT_V2_CKPT = "/scratch/user/s4670484/Model_save/samri2_gvt2_recon_K16_rec32_new/sam2_gvt_tsdf_final.pth"
DEFAULT_GVT_V3_CKPT = "/scratch/user/s4670484/Model_save/samri2_gvt3_tsdf_K16_AUX1_5x3d/sam2_gvt_tsdf_final.pth"
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
def resolve_label_id(gt_zyx: np.ndarray, label_field, dataset: str) -> int:
    try:
        return int(label_field)
    except (TypeError, ValueError):
        pass
    labels = get_labels_from_mask(gt_zyx)
    if len(labels) == 1:
        return labels[0]
    voxel_counts = {lb: int((gt_zyx == lb).sum()) for lb in labels}
    raise RuntimeError(
        f"Cannot resolve label '{label_field}' for dataset '{dataset}': mask has "
        f"{len(labels)} non-zero labels (voxel counts {voxel_counts}) and there is no "
        f"name->ID mapping in this repo. Please confirm which integer ID corresponds "
        f"to '{label_field}'."
    )


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
def crop_bbox_from_mask(mask2d: np.ndarray, margin_frac: float = 0.2) -> Tuple[int, int, int, int]:
    H, W = mask2d.shape
    ys, xs = np.where(mask2d > 0)
    if ys.size == 0:
        return 0, H, 0, W
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    my = int((y1 - y0 + 1) * margin_frac)
    mx = int((x1 - x0 + 1) * margin_frac)
    return max(0, y0 - my), min(H, y1 + my + 1), max(0, x0 - mx), min(W, x1 + mx + 1)


def draw_panel(
    ax, img2d_uint8: np.ndarray, crop: Tuple[int, int, int, int],
    gt2d: Optional[np.ndarray] = None, pred2d: Optional[np.ndarray] = None,
    pred_fill_alpha: float = 0.25, title: Optional[str] = None,
) -> None:
    y0, y1, x0, x1 = crop
    ax.imshow(img2d_uint8[y0:y1, x0:x1], cmap="gray", interpolation="nearest")

    if pred2d is not None and pred_fill_alpha > 0:
        pred_crop = pred2d[y0:y1, x0:x1]
        fill = np.zeros((*pred_crop.shape, 4))
        fill[pred_crop > 0] = (1.0, 0.0, 0.0, pred_fill_alpha)
        ax.imshow(fill, interpolation="nearest")

    if gt2d is not None:
        for c in find_contours(gt2d[y0:y1, x0:x1].astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color="lime", linewidth=2)

    if pred2d is not None:
        for c in find_contours(pred2d[y0:y1, x0:x1].astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color="red", linewidth=2)

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
def case_dir_for(pred_volumes_dir: Path, case: Dict) -> Path:
    return pred_volumes_dir / f"{sanitize_filename(case['dataset'])}__{sanitize_filename(case['label'])}"


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
def process_case(case: Dict, model_keys: List[str]) -> Dict:
    img_zyx, img_sitk = read_3d_image(case["img_path"])
    gt_zyx, _ = read_3d_image(case["mask_path"])
    label_id = resolve_label_id(gt_zyx, case["label"], case["dataset"])

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
def render_grid(records: List[Dict], model_keys: List[str], out_path_stem: Path, dpi: int) -> None:
    columns = ["input", "gt"] + model_keys
    ncols = len(columns)
    nrows = len(records)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    if nrows == 1:
        axes = axes[None, :]

    for i, rec in enumerate(records):
        case = rec["case"]
        crop = crop_bbox_from_mask(rec["gt_slice"], margin_frac=0.2)
        for j, col in enumerate(columns):
            ax = axes[i, j]
            if col == "input":
                draw_panel(ax, rec["img_slice"], crop, title="Input" if i == 0 else None)
            elif col == "gt":
                draw_panel(ax, rec["img_slice"], crop, gt2d=rec["gt_slice"], title="GT" if i == 0 else None)
            else:
                dsc = rec["computed_dice"].get(col, float("nan"))
                title = f"{MODEL_SPECS[col]['title']}\nDice={dsc:.2f}" if i == 0 else f"Dice={dsc:.2f}"
                draw_panel(
                    ax, rec["img_slice"], crop, gt2d=rec["gt_slice"],
                    pred2d=rec["pred_slices"].get(col), title=title,
                )
            if j == 0:
                ax.text(
                    -0.25, 0.5, f"{case['dataset']} — {case['label']}",
                    transform=ax.transAxes, ha="right", va="center",
                    rotation=90, fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(str(out_path_stem) + ".pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_path_stem) + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_single_case(rec: Dict, model_keys: List[str], out_path: Path, dpi: int) -> None:
    columns = ["input", "gt"] + model_keys
    fig, axes = plt.subplots(1, len(columns), figsize=(len(columns) * 2.2, 2.6))
    crop = crop_bbox_from_mask(rec["gt_slice"], margin_frac=0.2)
    for j, col in enumerate(columns):
        ax = axes[j]
        if col == "input":
            draw_panel(ax, rec["img_slice"], crop, title="Input")
        elif col == "gt":
            draw_panel(ax, rec["img_slice"], crop, gt2d=rec["gt_slice"], title="GT")
        else:
            dsc = rec["computed_dice"].get(col, float("nan"))
            draw_panel(
                ax, rec["img_slice"], crop, gt2d=rec["gt_slice"],
                pred2d=rec["pred_slices"].get(col), title=f"{MODEL_SPECS[col]['title']}\nDice={dsc:.2f}",
            )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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
    return p.parse_args()


def run_inference_pipeline(args, log) -> Tuple[Dict[str, List[Dict]], List[str]]:
    if not args.samples_json:
        raise SystemExit("--samples-json is required unless --from-volumes is set.")

    out_dir = Path(args.output_dir)
    save_volumes = not args.no_save_pred_volumes
    pred_volumes_dir = Path(args.pred_volumes_dir) if args.pred_volumes_dir else out_dir / "pred_volumes"

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
                rec = process_case(case, model_keys)
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

    rec_groups = {rec["case"]["dataset"] + "__" + rec["case"]["label"]: g
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
    if save_volumes:
        for rec in all_records:
            case = rec["case"]
            cdir = case_dir_for(pred_volumes_dir, case)
            meta = {
                "case": case,
                "label_id": rec["label_id"],
                "group": rec_groups[case["dataset"] + "__" + case["label"]],
                "model_keys": model_keys,
                "computed_dice": rec["computed_dice"],
            }
            (cdir / "case_meta.json").write_text(json.dumps(meta, indent=2))

    for rec in all_records:
        finalize_case_memory(rec)

    return records_by_group, model_keys


def run_from_volumes_pipeline(args, log) -> Tuple[Dict[str, List[Dict]], List[str]]:
    pred_volumes_dir = Path(args.from_volumes)
    log(f"Rebuilding figures from saved volumes in: {pred_volumes_dir}")

    records_by_group: Dict[str, List[Dict]] = {"showcase": [], "failure": []}
    model_keys: Optional[List[str]] = None
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
        model_keys = mk
        records_by_group.setdefault(group, []).append(rec)

    if model_keys is None:
        raise SystemExit(f"No valid case_meta.json found under {pred_volumes_dir}")
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
        records_by_group, model_keys = run_from_volumes_pipeline(args, log)
    else:
        records_by_group, model_keys = run_inference_pipeline(args, log)

    # Per-case PNGs
    for recs in records_by_group.values():
        for rec in recs:
            case = rec["case"]
            fname = f"{sanitize_filename(case['dataset'])}__{sanitize_filename(case['label'])}.png"
            render_single_case(rec, model_keys, out_dir / fname, dpi=args.dpi)

    # Figure grids
    render_grid(records_by_group["showcase"], model_keys, out_dir / "figure_showcase", dpi=args.dpi)
    render_grid(records_by_group["failure"], model_keys, out_dir / "figure_failure", dpi=args.dpi)

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

    print(f"\nSaved figures + run_log.txt to: {out_dir}")


if __name__ == "__main__":
    main()
