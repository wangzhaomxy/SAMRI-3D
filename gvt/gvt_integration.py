"""
GVT Integration with SAM2

Provides a wrapper around SAM2's memory_attention module that injects
Global Volume Tokens (GVT) into the memory bank without modifying any
files in the sam2/ directory.

Usage:
    model = build_sam2(...)
    model = wrap_model_with_gvt(model, num_tokens=4, num_layers=2)

    # Before processing a volume:
    compute_gvt_tokens(model, image_embeds, device, use_amp=True)

    # ... process slices (GVT tokens are automatically injected) ...

    # After processing:
    clear_gvt_cache(model)
"""

import torch
import torch.nn as nn
import numpy as np

from gvt.gvt_module import GVTModule


class GVTMemoryAttentionWrapper(nn.Module):
    """
    Wraps SAM2's MemoryAttention to inject GVT tokens into the memory bank.

    GVT tokens are appended to the memory and memory_pos tensors before
    calling the original MemoryAttention forward. They are excluded from
    spatial RoPE (like object pointer tokens) since they have no spatial structure.
    """

    def __init__(self, original_memory_attention: nn.Module, gvt_module: GVTModule):
        super().__init__()
        self.original = original_memory_attention
        self.gvt_module = gvt_module
        self.gvt_tokens_cache = None  # Set by compute_gvt_tokens()

        # Expose original attributes that SAM2Base accesses
        self.d_model = original_memory_attention.d_model

    def forward(
        self,
        curr,
        memory,
        curr_pos=None,
        memory_pos=None,
        num_obj_ptr_tokens: int = 0,
    ):
        if self.gvt_tokens_cache is not None:
            gvt_tokens, gvt_pos = self.gvt_tokens_cache
            device = memory.device
            gvt_tokens = gvt_tokens.to(device=device, dtype=memory.dtype)
            gvt_pos = gvt_pos.to(device=device, dtype=memory_pos.dtype)

            # Expand batch dim if needed (memory is (seq, B, dim))
            B = memory.shape[1]
            if gvt_tokens.shape[1] != B:
                gvt_tokens = gvt_tokens.expand(-1, B, -1)
                gvt_pos = gvt_pos.expand(-1, B, -1)

            # Append GVT tokens to memory bank
            memory = torch.cat([memory, gvt_tokens], dim=0)
            memory_pos = torch.cat([memory_pos, gvt_pos], dim=0)

            # Exclude GVT tokens from spatial RoPE
            num_obj_ptr_tokens += gvt_tokens.shape[0]

        return self.original(
            curr=curr,
            memory=memory,
            curr_pos=curr_pos,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )


def wrap_model_with_gvt(
    model,
    num_tokens: int = 16,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    device=None,
):
    """
    Wrap a SAM2 model's memory_attention with GVT injection.

    Args:
        model: SAM2Base or SAM2VideoPredictor model instance.
        num_tokens: Number of global volume tokens (K).
        num_layers: Number of Perceiver cross-attention layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        device: Device to place GVT module on.

    Returns:
        The same model with memory_attention replaced by GVTMemoryAttentionWrapper.
    """
    if device is None:
        device = next(model.parameters()).device

    input_dim = model.hidden_dim  # 256
    output_dim = model.mem_dim  # 64

    gvt_module = GVTModule(
        num_tokens=num_tokens,
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    wrapper = GVTMemoryAttentionWrapper(
        original_memory_attention=model.memory_attention,
        gvt_module=gvt_module,
    ).to(device)

    model.memory_attention = wrapper
    return model


def compute_gvt_tokens(model, image_embeds, device, use_amp: bool = False):
    """
    Compute and cache GVT tokens from all slice embeddings.

    Call this once per volume before the slice-by-slice propagation loop.

    Args:
        model: SAM2 model (unwrapped from DDP). Must have GVTMemoryAttentionWrapper.
        image_embeds: numpy array (N, 256, 64, 64) or torch tensor.
        device: Torch device.
        use_amp: Whether to use automatic mixed precision.
    """
    wrapper = model.memory_attention
    if not isinstance(wrapper, GVTMemoryAttentionWrapper):
        return

    if isinstance(image_embeds, np.ndarray):
        all_embeds = torch.from_numpy(image_embeds).to(device)
    else:
        all_embeds = image_embeds.to(device)

    with torch.amp.autocast("cuda", enabled=use_amp):
        gvt_tokens, gvt_pos = wrapper.gvt_module(all_embeds)

    wrapper.gvt_tokens_cache = (gvt_tokens, gvt_pos)


def clear_gvt_cache(model):
    """Clear cached GVT tokens after processing a volume."""
    wrapper = model.memory_attention
    if isinstance(wrapper, GVTMemoryAttentionWrapper):
        wrapper.gvt_tokens_cache = None


def has_gvt(model) -> bool:
    """Check if a model has GVT wrapper installed."""
    return isinstance(
        getattr(model, "memory_attention", None), GVTMemoryAttentionWrapper
    )


def load_gvt_checkpoint(model, ckpt_path, device="cpu"):
    """
    Load a checkpoint that contains GVT wrapper weights.

    The model must already be wrapped with GVT (via wrap_model_with_gvt)
    before calling this, so that the state_dict keys match.

    Args:
        model: SAM2 model already wrapped with GVTMemoryAttentionWrapper.
        ckpt_path: Path to checkpoint file (expects {"model": state_dict, ...}).
        device: Device for torch.load map_location.

    Returns:
        The model with loaded weights.
    """
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_gvt_checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[load_gvt_checkpoint] Unexpected keys: {unexpected}")
    return model
