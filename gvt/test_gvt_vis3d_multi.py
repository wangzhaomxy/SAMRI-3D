#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_gvt_vis3d_multi.py

3D evaluation with Global Volume Token (GVT).

Extends test_vis3d_multi.py by:
1. Loading GVT module weights alongside SAM2 checkpoint
2. Pre-encoding all frames and computing GVT tokens before propagation
3. GVT tokens are automatically injected via GVTMemoryAttentionWrapper

Usage:
    python gvt/test_gvt_vis3d_multi.py \\
        --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
        --ckpt-path /path/to/gvt_checkpoint.pth \\
        --dataset-path /path/to/Test_3d/ /path/to/ZeroShot_results/ \\
        --save-path /path/to/results_gvt.pkl \\
        --use-gvt --gvt-num-tokens 4 --gvt-num-layers 2 \\
        --prompt-mode box
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch

# Prefer local project root
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2_video_predictor

from gvt.gvt_integration import wrap_model_with_gvt, clear_gvt_cache, has_gvt, load_gvt_checkpoint
from gvt.gvt_inference import precompute_gvt_for_state, clear_gvt_for_predictor

# Import all evaluation infrastructure from the base script
from test_vis3d_multi import (
    suppress_sam2_output,
    init_state_compat,
    add_prompts_for_all_labels,
    propagate_video_multiobj,
    merge_two_videos_to_semantic_volume,
    PreparedVideos,
    prepare_videos_once,
    save_test_record_3d_multi_object,
    collect_test_folders,
    dice_3d,
    hd_msd_3d_sitk,
    choose_primary_axis,
    largest_mask_slice_index,
    build_two_videos_indices,
)


def predict_multi_object_semantic_gvt(
    predictor,
    prepared: PreparedVideos,
    gt_cut_slice: np.ndarray,
    prompt_mode: str,
    labels: List[int],
) -> Tuple[Optional[np.ndarray], List[int]]:
    """
    Multi-object inference with GVT.

    Same as predict_multi_object_semantic but pre-computes GVT tokens
    after init_state and before propagation.
    """
    use_gvt = has_gvt(predictor)

    # Video 1
    with suppress_sam2_output(True):
        st1 = init_state_compat(predictor, prepared.v1_source, prepared.v1_len, prefer_streaming=prepared.use_streaming)
        if use_gvt:
            precompute_gvt_for_state(predictor, st1)
        added = add_prompts_for_all_labels(predictor, st1, prompt_mode, gt_cut_slice, labels, ann_frame_idx=0)
        if not added:
            if use_gvt:
                clear_gvt_for_predictor(predictor)
            return None, []
        pred1 = propagate_video_multiobj(predictor, st1)

    # Video 2
    with suppress_sam2_output(True):
        st2 = init_state_compat(predictor, prepared.v2_source, prepared.v2_len, prefer_streaming=prepared.use_streaming)
        if use_gvt:
            precompute_gvt_for_state(predictor, st2)
        _ = add_prompts_for_all_labels(predictor, st2, prompt_mode, gt_cut_slice, added, ann_frame_idx=0)
        pred2 = propagate_video_multiobj(predictor, st2)

    if use_gvt:
        clear_gvt_for_predictor(predictor)

    N = len(prepared.v1_slices) + (len(prepared.v2_slices) - 1)
    H, W = gt_cut_slice.shape
    pred0 = merge_two_videos_to_semantic_volume(
        N=N, H=H, W=W,
        v1_slices=prepared.v1_slices,
        v2_slices=prepared.v2_slices,
        v1_pred=pred1,
        v2_pred=pred2,
        labels=added,
    )
    return pred0, added


def save_test_record_3d_gvt(
    dataset_paths: List[str],
    predictor,
    prompt_mode: str,
    save_path: str,
    num_workers: int = 4,
):
    """
    GVT-aware evaluation. Patches the prediction function and delegates to
    the base save_test_record_3d_multi_object.

    We monkey-patch the module-level predict_multi_object_semantic in
    test_vis3d_multi to use our GVT version, then call the base function.
    """
    import test_vis3d_multi as base_module

    # Save original and patch
    original_fn = base_module.predict_multi_object_semantic
    base_module.predict_multi_object_semantic = predict_multi_object_semantic_gvt

    try:
        result = save_test_record_3d_multi_object(
            dataset_paths=dataset_paths,
            predictor=predictor,
            prompt_mode=prompt_mode,
            save_path=save_path,
            num_workers=num_workers,
        )
    finally:
        # Restore original
        base_module.predict_multi_object_semantic = original_fn

    return result


def main():
    parser = argparse.ArgumentParser(description="3D evaluation with GVT")
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
    parser.add_argument("--gvt-num-tokens", type=int, default=16)
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
    if args.use_gvt:
        print(f"  Tokens: {args.gvt_num_tokens}, Layers: {args.gvt_num_layers}")
    print("======================================\n")

    t0 = time.time()

    # Build predictor
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

    save_test_record_3d_gvt(
        dataset_paths=args.dataset_path,
        predictor=predictor,
        prompt_mode=args.prompt_mode,
        save_path=args.save_path,
        num_workers=args.num_workers,
    )

    print(f"\nSaved: {args.save_path}")
    print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
