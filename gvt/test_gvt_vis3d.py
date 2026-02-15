#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_gvt_vis3d.py

3D single-object evaluation with Global Volume Token (GVT).

Extends test_vis3d.py by computing GVT tokens before propagation.
Uses the state-reuse pattern (init_state once, prefill, then loop labels).

Usage:
    python gvt/test_gvt_vis3d.py \\
        --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
        --ckpt-path /path/to/gvt_checkpoint.pth \\
        --dataset-path /path/to/Test_3d/ /path/to/ZeroShot_results/ \\
        --save-path /path/to/results_gvt_single.pkl \\
        --use-gvt --gvt-num-tokens 4 --gvt-num-layers 2 \\
        --prompt-mode box
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2_video_predictor

from gvt.gvt_integration import wrap_model_with_gvt, has_gvt, load_gvt_checkpoint
from gvt.gvt_inference import compute_gvt_from_cached_features, clear_gvt_for_predictor

from test_vis3d import (
    suppress_sam2_output,
    init_state_compat,
    prefill_image_features,
    add_prompt_single_label,
    propagate_video_singleobj,
    merge_two_videos_single_label,
    PreparedVideos,
    save_test_record_3d_single_object,
)


def predict_all_labels_reuse_state_gvt(
    predictor,
    prepared: PreparedVideos,
    gt_cut_slice: np.ndarray,
    prompt_mode: str,
    labels: List[int],
) -> Dict[int, Optional[np.ndarray]]:
    """
    Single-object inference for every label with GVT and state reuse.

    After init_state + prefill, computes GVT tokens from cached features.
    GVT tokens persist across reset_state calls (they live on the wrapper).
    """
    results: Dict[int, Optional[np.ndarray]] = {}
    N = len(prepared.v1_slices) + (len(prepared.v2_slices) - 1)
    H, W = gt_cut_slice.shape

    promptable = [int(lb) for lb in labels if np.sum(gt_cut_slice == int(lb)) > 0]
    if not promptable:
        return results

    use_gvt = has_gvt(predictor)

    with suppress_sam2_output(True):
        # Init states once per case
        st1 = init_state_compat(predictor, prepared.v1_source, prepared.v1_len,
                                prefer_streaming=prepared.use_streaming)
        st2 = init_state_compat(predictor, prepared.v2_source, prepared.v2_len,
                                prefer_streaming=prepared.use_streaming)

        # Pre-fill image features (always do this for state-reuse pattern)
        if hasattr(predictor, "forward_image"):
            prefill_image_features(predictor, st1)
            prefill_image_features(predictor, st2)

        for lb in promptable:
            # --- video1 (use v1 GVT) ---
            if use_gvt:
                compute_gvt_from_cached_features(predictor, st1)

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

            # --- video2 (use v2 GVT) ---
            if use_gvt:
                compute_gvt_from_cached_features(predictor, st2)

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

            results[lb] = merge_two_videos_single_label(
                N=N, H=H, W=W,
                v1_slices=prepared.v1_slices,
                v2_slices=prepared.v2_slices,
                v1_pred=pred1,
                v2_pred=pred2,
            )

    if use_gvt:
        clear_gvt_for_predictor(predictor)

    return results


def main():
    parser = argparse.ArgumentParser(description="3D single-object evaluation with GVT")
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, nargs=2, required=True,
                        metavar=("TEST_DATA", "ZEROSHOT_RESULTS"))
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--prompt-mode", type=str, choices=["box", "mask"], default="box")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    # GVT configuration
    parser.add_argument("--use-gvt", action="store_true")
    parser.add_argument("--gvt-num-tokens", type=int, default=4)
    parser.add_argument("--gvt-num-layers", type=int, default=2)
    parser.add_argument("--gvt-num-heads", type=int, default=4)

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
    print(f"GVT           : {'enabled' if args.use_gvt else 'disabled'}")
    print("======================================\n")

    t0 = time.time()

    if args.use_gvt:
        # Build without checkpoint, wrap with GVT, then load GVT checkpoint
        predictor = build_sam2_video_predictor(args.sam2_config, ckpt_path=None, device=args.device)
        predictor = wrap_model_with_gvt(
            predictor,
            num_tokens=args.gvt_num_tokens,
            num_layers=args.gvt_num_layers,
            num_heads=args.gvt_num_heads,
            device=args.device,
        )
        load_gvt_checkpoint(predictor, args.ckpt_path, device=args.device)
        gvt_params = sum(p.numel() for p in predictor.memory_attention.gvt_module.parameters())
        print(f"GVT module: {gvt_params:,} parameters")
    else:
        predictor = build_sam2_video_predictor(args.sam2_config, args.ckpt_path, device=args.device)

    # Monkey-patch the prediction function
    import test_vis3d as base_module
    original_fn = base_module.predict_all_labels_reuse_state
    base_module.predict_all_labels_reuse_state = predict_all_labels_reuse_state_gvt

    try:
        save_test_record_3d_single_object(
            dataset_paths=args.dataset_path,
            predictor=predictor,
            prompt_mode=args.prompt_mode,
            save_path=args.save_path,
            num_workers=args.num_workers,
        )
    finally:
        base_module.predict_all_labels_reuse_state = original_fn

    print(f"\nSaved: {args.save_path}")
    print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
