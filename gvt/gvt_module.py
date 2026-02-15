"""
GVT (Global Volume Token) Module

Perceiver-style cross-attention that compresses N spatially-pooled slice
embeddings into K global tokens (default K=4). These tokens are injected
into SAM2's memory bank to provide volume-level context to every slice.

Architecture:
    Input:  (N, 256, 64, 64) slice embeddings from SAM2 image encoder
    Pool:   (N, 256) via spatial average pooling
    Project: (N, 64) via linear projection to mem_dim
    Cross-attention: K learnable queries attend to N slice tokens
    Output: (K, 1, 64) global volume tokens + (K, 1, 64) positional encoding
"""

import math

import torch
import torch.nn as nn


class GVTModule(nn.Module):
    def __init__(
        self,
        num_tokens: int = 16,
        input_dim: int = 256,
        output_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_slices: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_tokens: Number of global volume tokens (K).
            input_dim: Channel dimension of input slice embeddings (256 for SAM2).
            output_dim: Output dimension matching SAM2's mem_dim (64).
            num_heads: Number of attention heads in cross-attention.
            num_layers: Number of Perceiver cross-attention layers.
            max_slices: Maximum number of slices for positional encoding.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim

        # Learnable global query tokens
        self.global_queries = nn.Parameter(torch.randn(num_tokens, output_dim) * 0.02)

        # Project spatial-pooled slice embeddings from input_dim to output_dim
        self.input_proj = nn.Linear(input_dim, output_dim)

        # Sinusoidal slice position embedding (not learnable, like standard PE)
        self.register_buffer(
            "slice_pos_enc",
            self._build_sinusoidal_pe(max_slices, output_dim),
        )

        # Perceiver cross-attention layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Learnable positional encoding for GVT tokens in memory bank
        self.gvt_pos_enc = nn.Parameter(torch.randn(num_tokens, output_dim) * 0.02)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
        """Build sinusoidal positional encoding (max_len, dim)."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, dim)

    def forward(self, slice_embeddings: torch.Tensor) -> tuple:
        """
        Compress N slice embeddings into K global volume tokens.

        Args:
            slice_embeddings: (N, 256, 64, 64) image encoder outputs for all slices.

        Returns:
            gvt_tokens: (K, 1, 64) global volume tokens for memory bank.
            gvt_pos: (K, 1, 64) positional encoding for GVT tokens.
        """
        N = slice_embeddings.shape[0]

        # Spatial average pool: (N, 256, 64, 64) -> (N, 256)
        pooled = slice_embeddings.flatten(2).mean(dim=2)

        # Project to mem_dim: (N, 256) -> (N, 64)
        projected = self.input_proj(pooled)

        # Add slice positional encoding: (N, 64)
        projected = projected + self.slice_pos_enc[:N].to(projected.device)

        # Perceiver cross-attention: queries (1, K, 64) attend to KV (1, N, 64)
        queries = self.global_queries.unsqueeze(0)  # (1, K, 64)
        kv = projected.unsqueeze(0)  # (1, N, 64)
        gvt_out = self.cross_attn(queries, kv)  # (1, K, 64)

        # Reshape for memory bank format: (K, 1, 64)
        gvt_tokens = gvt_out.squeeze(0).unsqueeze(1)  # (K, 1, 64)
        gvt_pos = self.gvt_pos_enc.unsqueeze(1)  # (K, 1, 64)

        return gvt_tokens, gvt_pos
