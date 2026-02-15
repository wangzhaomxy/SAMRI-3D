"""
GVT Inference Helpers

Pre-encodes all video frames and computes GVT tokens before propagation.
Works transparently with the GVTMemoryAttentionWrapper.
"""

import torch
import numpy as np

from gvt.gvt_integration import GVTMemoryAttentionWrapper, clear_gvt_cache


def precompute_gvt_for_state(predictor, inference_state, use_amp: bool = True):
    """
    Pre-encode all frames in an inference state and compute GVT tokens.

    This must be called after init_state() but before propagate_in_video().
    It extracts backbone_fpn[-1] (256, 64, 64) from each frame and passes
    all of them through the GVT module to produce global volume tokens.

    Args:
        predictor: SAM2VideoPredictor with GVTMemoryAttentionWrapper
        inference_state: State dict from init_state()
        use_amp: Whether to use automatic mixed precision
    """
    wrapper = predictor.memory_attention
    if not isinstance(wrapper, GVTMemoryAttentionWrapper):
        return

    device = inference_state["device"]
    num_frames = inference_state["num_frames"]

    # Pre-encode all frames and collect top-level features
    all_features = []
    for frame_idx in range(num_frames):
        image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                backbone_out = predictor.forward_image(image)

        # backbone_fpn[-1] is the top-level feature: (1, 256, 64, 64)
        top_feat = backbone_out["backbone_fpn"][-1]
        all_features.append(top_feat)

        # Also cache for later use during propagation
        inference_state["cached_features"][frame_idx] = (image, backbone_out)

    # Stack features: (N, 256, 64, 64)
    all_features = torch.cat(all_features, dim=0)

    # Compute GVT tokens
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            gvt_tokens, gvt_pos = wrapper.gvt_module(all_features)

    wrapper.gvt_tokens_cache = (gvt_tokens, gvt_pos)


def compute_gvt_from_cached_features(predictor, inference_state):
    """
    Compute GVT tokens from already-cached backbone features.

    Use this when prefill_image_features() has already been called,
    so all frames' backbone_out are in inference_state["cached_features"].
    """
    wrapper = predictor.memory_attention
    if not isinstance(wrapper, GVTMemoryAttentionWrapper):
        return

    cache = inference_state.get("cached_features", {})
    if not cache:
        precompute_gvt_for_state(predictor, inference_state)
        return

    device = inference_state["device"]
    num_frames = inference_state["num_frames"]

    all_features = []
    for i in range(num_frames):
        if i not in cache:
            image = inference_state["images"][i].to(device).float().unsqueeze(0)
            with torch.no_grad():
                backbone_out = predictor.forward_image(image)
            cache[i] = (image, backbone_out)
        _, backbone_out = cache[i]
        top_feat = backbone_out["backbone_fpn"][-1].to(device)
        all_features.append(top_feat)

    all_features = torch.cat(all_features, dim=0)

    with torch.no_grad():
        gvt_tokens, gvt_pos = wrapper.gvt_module(all_features)

    wrapper.gvt_tokens_cache = (gvt_tokens, gvt_pos)


def clear_gvt_for_predictor(predictor):
    """Clear GVT cache on a predictor."""
    clear_gvt_cache(predictor)
