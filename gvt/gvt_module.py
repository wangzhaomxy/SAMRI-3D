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

        # LayerNorm for KV (projected slice tokens) and Q (global queries)
        self.input_norm = nn.LayerNorm(output_dim)
        self.query_norm = nn.LayerNorm(output_dim)

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

        # Add slice positional encoding and normalize: (N, 64)
        projected = projected + self.slice_pos_enc[:N].to(projected.device)
        projected = self.input_norm(projected)

        # Perceiver cross-attention: queries (1, K, 64) attend to KV (1, N, 64)
        queries = self.query_norm(self.global_queries).unsqueeze(0)  # (1, K, 64)
        kv = projected.unsqueeze(0)  # (1, N, 64)
        gvt_out = self.cross_attn(queries, kv)  # (1, K, 64)

        # Reshape for memory bank format: (K, 1, 64)
        gvt_tokens = gvt_out.squeeze(0).unsqueeze(1)  # (K, 1, 64)
        gvt_pos = self.gvt_pos_enc.unsqueeze(1)  # (K, 1, 64)

        return gvt_tokens, gvt_pos


class GVTReconDecoder(nn.Module):
    """
    Training-only auxiliary decoder. Expands K GVT tokens back to N slice masks
    at low resolution. Mirrors the GVT encoder: encoder compresses N→K, decoder
    expands K→N. Discarded at inference — zero overhead.

    Architecture:
        Input:  gvt_tokens (K, 1, C_mem) from GVT encoder
                mem_features (1, M, C_mem) label-conditioned memory from cut slice (optional)
        Label conditioning (when mem_features provided):
                Cross-attention: Q=GVT tokens, K/V=mem_features → conditioned GVT tokens
        Query:  sinusoidal slice PE (N, C_mem) — same PE as GVTModule
        Decoder: 2-layer TransformerDecoder (N slice queries cross-attend to conditioned tokens)
        Head:   Linear → (N, recon_res * recon_res) → reshape to (1, N, 1, R, R)
    """

    def __init__(self, C_mem: int = 64, max_slices: int = 256, recon_res: int = 16):
        super().__init__()
        self.recon_res = recon_res

        # Reuse the SAME sinusoidal PE as GVTModule for consistent slice positions
        self.register_buffer(
            "slice_pos_enc",
            GVTModule._build_sinusoidal_pe(max_slices, C_mem),
        )

        # Label conditioning: GVT tokens (Q) attend to cut-slice memory features (K/V)
        self.label_cross_attn = nn.MultiheadAttention(
            embed_dim=C_mem, num_heads=4, batch_first=True
        )
        self.label_cross_attn_norm = nn.LayerNorm(C_mem)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=C_mem, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.mask_head = nn.Sequential(
            nn.Linear(C_mem, C_mem * 2),
            nn.ReLU(),
            nn.Linear(C_mem * 2, recon_res * recon_res),
        )

    def forward(
        self,
        gvt_tokens: torch.Tensor,
        num_slices: int,
        mem_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            gvt_tokens:   (K, 1, C_mem) — output of GVT encoder.
            num_slices:   int — N slices in this volume.
            mem_features: (1, M, C_mem) — label-conditioned memory features from the
                          cut slice (Phase 1 output). When provided, GVT tokens are
                          conditioned on this label before slice reconstruction.
        Returns:
            recon_logits: (1, N, 1, recon_res, recon_res)
        """
        # Normalise to (1, K, C_mem) for batch_first attention
        if gvt_tokens.shape[1] == 1:
            tokens = gvt_tokens.transpose(0, 1)  # (K, 1, C_mem) → (1, K, C_mem)
        else:
            tokens = gvt_tokens  # already (1, K, C_mem)

        # Label conditioning: cross-attend GVT tokens to cut-slice memory features
        if mem_features is not None:
            kv = mem_features.to(device=tokens.device, dtype=tokens.dtype)
            attn_out, _ = self.label_cross_attn(query=tokens, key=kv, value=kv)
            tokens = self.label_cross_attn_norm(tokens + attn_out)  # (1, K, C_mem)

        # Slice queries from sinusoidal PE — match dtype/device of tokens
        queries = self.slice_pos_enc[:num_slices].unsqueeze(0)   # (1, N, C_mem)
        queries = queries.to(device=tokens.device, dtype=tokens.dtype)

        decoded = self.decoder(queries, tokens)                   # (1, N, C_mem)
        masks = self.mask_head(decoded.float())                   # (1, N, recon_res^2)
        return masks.view(1, num_slices, 1, self.recon_res, self.recon_res)


class TSDFDecoder(nn.Module):
    """
    Per-slice TSDF decoder: GVT tokens → 256×256 TSDF prediction.
    Training-only auxiliary decoder (discarded at inference).

    Architecture:
        1. Add learned slice positional embedding to GVT tokens
        2. Cross-attention: Q=GVT tokens, K/V=prompted slice memory features (label condition)
        3. Linear projection: flatten K tokens → single feature vector
        4. Reshape to spatial seed (seed_channels × 4 × 4)
        5. ConvTranspose2d upsampling: 4→8→16→32→64→128→256 (6 stages)
        6. Final 1×1 conv → single-channel TSDF (linear activation, NO sigmoid)

    Parameter count: ~100-150K.
    """

    def __init__(self,
                 num_tokens: int = 16,
                 token_dim: int = 64,
                 max_slices: int = 256,
                 seed_channels: int = 64,
                 seed_size: int = 4,
                 mem_feature_dim: int = None):  # if None, assume same as token_dim
        super().__init__()

        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.seed_channels = seed_channels
        self.seed_size = seed_size

        # Project memory features to token_dim if they differ
        actual_mem_dim = mem_feature_dim if mem_feature_dim is not None else token_dim
        self.mem_proj = nn.Linear(actual_mem_dim, token_dim) if actual_mem_dim != token_dim else nn.Identity()

        # Learned slice positional embedding
        self.slice_pos_embed = nn.Embedding(max_slices, token_dim)

        # Cross-attention: GVT tokens (Q) attend to prompted memory features (K/V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=4,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(token_dim)

        # Linear projection: K tokens → spatial seed
        self.token_proj = nn.Linear(num_tokens * token_dim, seed_channels * seed_size * seed_size)

        # Convolutional upsampling: 4×4 → 256×256 (6 stages, each ×2)
        channels = [seed_channels, 64, 64, 32, 32, 16, 8]
        self.upsample = nn.ModuleList()
        for i in range(6):
            self.upsample.append(nn.Sequential(
                nn.ConvTranspose2d(channels[i], channels[i+1],
                                   kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(8, channels[i+1]), channels[i+1]),
                nn.GELU()
            ))

        # Final 1×1 conv → single channel TSDF (linear activation, no sigmoid)
        self.head = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward(self,
                gvt_tokens: torch.Tensor,       # (B, K, C_mem)
                mem_features: torch.Tensor,      # (B, M, C_mem_actual)
                slice_indices: torch.Tensor      # (S,) integer tensor
               ) -> torch.Tensor:
        """
        Predict TSDF for given slices.

        Returns:
            tsdf_pred: (B, S, 256, 256)
        """
        B, K, C = gvt_tokens.shape
        S = slice_indices.shape[0]

        # Project memory features if needed
        mem_proj = self.mem_proj(mem_features)  # (B, M, token_dim)

        predictions = []
        for s_idx in range(S):
            # 1. Add slice positional embedding
            slice_pe = self.slice_pos_embed(slice_indices[s_idx:s_idx+1])  # (1, C)
            tokens_s = gvt_tokens + slice_pe.unsqueeze(0)  # (B, K, C)

            # 2. Cross-attention with label-conditioned memory features
            attn_out, _ = self.cross_attn(
                query=tokens_s,
                key=mem_proj,
                value=mem_proj
            )
            tokens_s = self.cross_attn_norm(tokens_s + attn_out)  # (B, K, C)

            # 3. Flatten tokens → linear → spatial seed
            flat = tokens_s.reshape(B, K * C)
            seed = self.token_proj(flat)  # (B, seed_channels * 4 * 4)
            seed = seed.reshape(B, self.seed_channels, self.seed_size, self.seed_size)

            # 4. Convolutional upsampling
            x = seed
            for layer in self.upsample:
                x = layer(x)

            # 5. Final 1×1 conv → single-channel TSDF
            tsdf = self.head(x)  # (B, 1, 256, 256)
            predictions.append(tsdf.squeeze(1))  # (B, 256, 256)

        return torch.stack(predictions, dim=1)  # (B, S, 256, 256)
