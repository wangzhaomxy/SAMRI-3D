#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_gvt_recon.py

Diagnostic script for GVT v2 checkpoints.

For each test volume:
  1. Load a trained GVT v2 checkpoint (model + GVTReconDecoder weights).
  2. Run the GVT encoder to get K global tokens.
  3. Run the GVTReconDecoder to reconstruct per-slice low-res masks.
  4. Compute per-volume reconstruction Dice.
  5. Optionally save visualisations (GT vs reconstruction side-by-side).

This is for analysis only — not part of the main evaluation pipeline.
GVT v2 inference (test_gvt_vis3d_multi.py) is identical to v1 and never loads
the reconstruction decoder.

Usage:
    python gvt/eval_gvt_recon.py \\
        --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
        --ckpt-path /path/to/gvt_v2_checkpoint.pth \\
        --embedding-dir /path/to/3D_embeddings/ \\
        --gvt-num-tokens 16 \\
        --gvt-recon-res 16 \\
        --save-dir /path/to/recon_vis/ \\
        --device cuda
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2
from gvt.gvt_module import GVTReconDecoder
from gvt.gvt_integration import (
    wrap_model_with_gvt,
    load_gvt_checkpoint,
    GVTMemoryAttentionWrapper,
)
from train_unified_emb import Embedding3DDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_decoder(args, device):
    """Build SAM2 + GVT wrapper and load the v2 checkpoint."""
    model = build_sam2(
        config_file=args.sam2_config,
        ckpt_path=None,
        device=device,
        mode="eval",
        apply_postprocessing=False,
    )
    model = wrap_model_with_gvt(
        model,
        num_tokens=args.gvt_num_tokens,
        num_layers=args.gvt_num_layers,
        num_heads=args.gvt_num_heads,
        device=device,
    )
    load_gvt_checkpoint(model, args.ckpt_path, device=device)
    model.eval()

    # Load reconstruction decoder
    sd = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    if "gvt_recon_decoder" not in sd:
        raise KeyError(
            f"Checkpoint '{args.ckpt_path}' has no 'gvt_recon_decoder' key. "
            "Was it trained with --gvt-recon?"
        )
    mem_dim = model.mem_dim  # 64
    recon_decoder = GVTReconDecoder(
        C_mem=mem_dim,
        max_slices=256,
        recon_res=args.gvt_recon_res,
    ).to(device)
    missing, unexpected = recon_decoder.load_state_dict(sd["gvt_recon_decoder"], strict=False)
    if missing:
        print(f"[eval_gvt_recon] Missing keys in recon decoder (old checkpoint — "
              f"label_cross_attn will be randomly initialized): {missing}")
    if unexpected:
        print(f"[eval_gvt_recon] Unexpected keys in recon decoder: {unexpected}")
    recon_decoder.eval()

    return model, recon_decoder


def compute_recon_dice(recon_logits: torch.Tensor, gt_vol: torch.Tensor) -> float:
    """Compute binary Dice between sigmoid(recon_logits) and gt_vol."""
    prob = torch.sigmoid(recon_logits)
    pred = (prob > 0.5).float()
    gt = gt_vol.float()
    intersection = (pred * gt).sum().item()
    denom = pred.sum().item() + gt.sum().item()
    if denom < 1e-6:
        return 1.0  # both empty — perfect agreement
    return (2.0 * intersection) / denom


def save_recon_vis(
    recon_logits: torch.Tensor,
    gt_vol: np.ndarray,
    save_path: str,
    recon_res: int,
    volume_name: str,
):
    """
    Save a grid visualisation: each column is a slice, top row = GT (downsampled),
    bottom row = sigmoid(recon_logits).
    Shows every 4th slice to keep the figure manageable.
    """
    # recon_logits: (1, N, 1, R, R) on device → (N, R, R)
    prob = torch.sigmoid(recon_logits)[0, :, 0, :, :].cpu().numpy()  # (N, R, R)
    N = prob.shape[0]
    step = max(1, N // 16)
    slice_ids = list(range(0, N, step))[:16]

    # Downsample GT to recon_res
    gt_tensor = torch.from_numpy(gt_vol.astype(np.float32))  # (N, 1, H, W)
    gt_low = F.interpolate(
        gt_tensor, size=(recon_res, recon_res), mode="bilinear", align_corners=False
    ).squeeze(1).numpy()  # (N, R, R)

    n_cols = len(slice_ids)
    fig, axes = plt.subplots(2, n_cols, figsize=(2 * n_cols, 4))
    fig.suptitle(f"{volume_name}  —  GVT v2 reconstruction", fontsize=9)

    for col_idx, s in enumerate(slice_ids):
        # GT (top row)
        ax_gt = axes[0, col_idx] if n_cols > 1 else axes[0]
        ax_gt.imshow(gt_low[s], vmin=0, vmax=1, cmap="gray")
        ax_gt.set_title(f"s{s}", fontsize=6)
        ax_gt.axis("off")
        if col_idx == 0:
            ax_gt.set_ylabel("GT", fontsize=7)

        # Prediction (bottom row)
        ax_pr = axes[1, col_idx] if n_cols > 1 else axes[1]
        ax_pr.imshow(prob[s], vmin=0, vmax=1, cmap="hot")
        ax_pr.axis("off")
        if col_idx == 0:
            ax_pr.set_ylabel("Pred", fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GVT v2 reconstruction diagnostic: visualise and score per-volume reconstruction quality."
    )
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--embedding-dir", type=str, required=True,
                        help="Root of pre-computed 3D embeddings (same as training).")

    # GVT config (must match checkpoint)
    parser.add_argument("--gvt-num-tokens", type=int, default=16)
    parser.add_argument("--gvt-num-layers", type=int, default=2)
    parser.add_argument("--gvt-num-heads", type=int, default=4)
    parser.add_argument("--gvt-recon-res", type=int, default=16)

    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save PNG visualisations (skip if not set).")
    parser.add_argument("--max-volumes", type=int, default=None,
                        help="Limit number of volumes evaluated (for quick checks).")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt_path}")
    model, recon_decoder = load_model_and_decoder(args, device)

    wrapper = model.memory_attention
    assert isinstance(wrapper, GVTMemoryAttentionWrapper), \
        "Model does not have GVT wrapper — was it built with wrap_model_with_gvt?"

    print(f"Loading 3D embedding dataset from: {args.embedding_dir}")
    dataset = Embedding3DDataset(
        embedding_dir=args.embedding_dir,
        samples_per_dataset=10000,  # load everything
        seed=0,
    )

    all_dice = []
    volumes_processed = 0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating reconstruction"):
            if args.max_volumes is not None and volumes_processed >= args.max_volumes:
                break

            sample = dataset[idx]
            image_embeds = sample["image_embeds"]     # (N, 256, 64, 64) np or tensor
            masks = sample["masks"]                   # (N, H, W) numpy int32 array
            labels = sample["labels"]
            ori_h, ori_w = sample["ori_h"], sample["ori_w"]

            # Convert embeddings to tensor
            # Cast to float32: embeddings are stored as float16, but GVTModule weights
            # are float32. Without AMP autocast (used in training but not here), calling
            # gvt_module with float16 input causes a dtype mismatch error.
            if isinstance(image_embeds, np.ndarray):
                all_embeds = torch.from_numpy(image_embeds).to(device).float()
            else:
                all_embeds = image_embeds.to(device).float()

            N_total = all_embeds.shape[0]

            # Run GVT encoder
            gvt_tokens, _ = wrapper.gvt_module(all_embeds)   # (K, 1, 64)

            # Run reconstruction decoder (unconditioned — no per-label mem_features).
            # For checkpoints trained with label conditioning (--gvt-recon + Phase 1),
            # scores here will be lower than per-label conditioned evaluation since the
            # decoder cross-attention is bypassed. This is a union-GT sanity check only.
            recon_logits = recon_decoder(gvt_tokens, num_slices=N_total)
            # (1, N, 1, recon_res, recon_res)

            # Build combined binary GT — masks is (N, H, W) int32 numpy array
            gt_vol_list = []
            for si in range(N_total):
                m = masks[si]  # (H, W) numpy int32
                if isinstance(m, np.ndarray):
                    m_t = torch.from_numpy(m.astype(np.int32))
                else:
                    m_t = m.int()
                # Explicit loop: safe when labels is empty (produces all-zero tensor)
                any_fg = torch.zeros(m_t.shape, dtype=torch.float32)
                for lbl in labels:
                    any_fg = any_fg + (m_t == int(lbl)).float()
                any_fg = any_fg.clamp(max=1.0).unsqueeze(0)  # (1, H, W)
                gt_vol_list.append(any_fg)
            gt_vol = torch.stack(gt_vol_list, dim=0).to(device)  # (N, 1, H, W)

            # Dice
            # Downsample GT to recon_res for fair comparison
            gt_low = F.interpolate(
                gt_vol.float(),
                size=(args.gvt_recon_res, args.gvt_recon_res),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(0)  # (1, N, 1, R, R)

            dice = compute_recon_dice(recon_logits, gt_low)
            all_dice.append(dice)

            # Visualisation
            if args.save_dir is not None:
                vol_name = f"vol_{idx:04d}"
                vis_path = os.path.join(args.save_dir, f"{vol_name}_recon.png")
                gt_np = gt_vol.cpu().numpy()  # (N, 1, H, W)
                save_recon_vis(
                    recon_logits.cpu(),
                    gt_np,
                    save_path=vis_path,
                    recon_res=args.gvt_recon_res,
                    volume_name=f"{vol_name}  Dice={dice:.3f}",
                )

            volumes_processed += 1

    if all_dice:
        mean_dice = np.mean(all_dice)
        std_dice = np.std(all_dice)
        print(f"\n{'='*50}")
        print(f"Reconstruction Dice over {volumes_processed} volumes:")
        print(f"  Mean: {mean_dice:.4f}")
        print(f"  Std:  {std_dice:.4f}")
        print(f"  Min:  {min(all_dice):.4f}")
        print(f"  Max:  {max(all_dice):.4f}")
        if args.save_dir is not None:
            # Save per-volume CSV
            csv_path = os.path.join(args.save_dir, "recon_dice.csv")
            os.makedirs(args.save_dir, exist_ok=True)
            with open(csv_path, "w") as f:
                f.write("volume_idx,dice\n")
                for i, d in enumerate(all_dice):
                    f.write(f"{i},{d:.6f}\n")
            print(f"Per-volume Dice saved to: {csv_path}")


if __name__ == "__main__":
    main()
