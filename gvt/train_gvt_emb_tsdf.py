#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_gvt_emb_tsdf.py

GVT v3: Training from pre-computed 3D embeddings with GVT and combined
2D + 3D volumetric loss + TSDF shape prior auxiliary loss.

Extends train_gvt_emb_3dloss.py by adding a TSDFDecoder auxiliary loss.

Three-phase training loop per volume:
  Phase 1 (no_grad): collect prompted memory features after add_new_points_or_box
  Phase 2:           TSDF auxiliary loss (TSDFDecoder, capped at max_slices)
  Phase 3:           Standard seg loss (existing v2 per-label propagation, unchanged)

Total loss = L_seg + lambda_tsdf * L_tsdf   (additive, NOT averaged)

Usage:
    python gvt/train_gvt_emb_tsdf.py \\
        --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
        --pretrained-ckpt /path/to/pretrained.pth \\
        --embedding-dir /path/to/3D_embeddings/ \\
        --save-path /path/to/gvt_tsdf_checkpoints/ \\
        --use-gvt --gvt-num-tokens 16 --gvt-num-layers 2 \\
        --lambda-2d 1.0 --lambda-3d 0.5 \\
        --lambda-tsdf 0.1 --tsdf-tau 32.0 \\
        --batch-size 2 --num-epochs 50 --lr 5e-6 --use-amp
"""

import os
import sys
import csv
import argparse
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from tqdm import tqdm

# Prefer local ./sam2 and project root
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2
from utils.losses import DiceFocalLoss, gvt_recon_loss
from utils.tsdf_utils import prepare_tsdf_targets, tsdf_loss
from gvt.gvt_module import GVTReconDecoder, TSDFDecoder
from gvt.gvt_integration import (
    wrap_model_with_gvt,
    compute_gvt_tokens,
    clear_gvt_cache,
    has_gvt,
    load_gvt_checkpoint,
)

# Reuse from the base scripts
from train_unified_emb import (
    Embedding3DDataset,
    MixedEmbeddingDataset,
    mixed_emb_collate_fn,
    freeze_unified_components,
    count_parameters,
    ddp_setup,
)
from train_unified_emb_3dloss import (
    train_2d_sample_from_embedding,
    process_video_with_memory_3dloss,
)

# Re-import helpers from GVT v2 training script
from gvt.train_gvt_emb_3dloss import (
    get_lambda_aux,
    _build_gt_volume_binary,
)


# =============================================================================
# GVT-aware 3D Sample Training with 3D Loss + TSDF
# =============================================================================

def subsample_slices(slices: List[int], max_len: int) -> List[int]:
    """Uniformly subsample a list of slice indices to at most max_len entries."""
    if len(slices) <= max_len:
        return slices
    step = max(1, (len(slices) - 1) // (max_len - 1))
    subsampled = [slices[0]] + slices[1::step][:max_len - 1]
    return subsampled


def train_3d_sample_from_embedding_gvt_tsdf(
    model,
    predictor,
    sample: Dict,
    loss_fn,
    args,
    device,
    recon_decoder=None,
    tsdf_decoder: Optional[nn.Module] = None,
    epoch: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train on a 3D volume with GVT, combined 2D + 3D volumetric loss, and TSDF auxiliary loss.

    Three-phase structure:
      Phase 1 (no_grad): collect per-label prompted memory features
      Phase 2:           TSDF auxiliary loss (TSDFDecoder, capped at max_slices)
      Phase 3:           Standard seg loss (recon + per-label propagation)

    Returns:
        (total_loss, avg_vol_loss, recon_loss_scalar, tsdf_loss_scalar)
    """
    image_embeds = sample["image_embeds"]
    high_res_feat_0s = sample["high_res_feat_0s"]
    high_res_feat_1s = sample["high_res_feat_1s"]
    masks = sample["masks"]
    labels = sample["labels"]
    ori_h, ori_w = sample["ori_h"], sample["ori_w"]

    base_model = model.module if hasattr(model, 'module') else model

    N_total = image_embeds.shape[0] if hasattr(image_embeds, 'shape') else len(image_embeds)

    # Compute GVT tokens from all slice embeddings (ALL N slices, uncapped)
    if has_gvt(base_model):
        compute_gvt_tokens(base_model, image_embeds, device, use_amp=args.use_amp)

    max_slices = getattr(args, 'max_slices_per_video', 32)

    # Subsample labels if needed
    max_labels_per_sample = getattr(args, 'max_labels_per_sample', 1)
    active_labels = labels
    if len(active_labels) > max_labels_per_sample:
        rng = np.random.RandomState()
        active_labels = rng.choice(active_labels, size=max_labels_per_sample, replace=False).tolist()

    masks_np = masks if isinstance(masks, np.ndarray) else (
        masks.numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
    )

    # -------------------------------------------------------------------------
    # GVT v2: Reconstruction loss (same as train_gvt_emb_3dloss.py)
    # -------------------------------------------------------------------------
    recon_loss_scalar = torch.tensor(0.0, device=device)
    recon_loss_val = None
    if recon_decoder is not None and has_gvt(base_model):
        wrapper = base_model.memory_attention
        gvt_tokens_tensor, _ = wrapper.gvt_tokens_cache   # (K, 1, 64)
        N_total = image_embeds.shape[0] if hasattr(image_embeds, "shape") else len(image_embeds)
        recon_slice_indices = list(range(N_total))
        if N_total > max_slices:
            recon_slice_indices = subsample_slices(recon_slice_indices, max_slices)
        N_recon = len(recon_slice_indices)
        recon_masks = masks[np.array(recon_slice_indices)] if N_recon < N_total else masks
        gt_vol = _build_gt_volume_binary(recon_masks, active_labels, N_recon, ori_h, ori_w, device)
        with torch.amp.autocast("cuda", enabled=args.use_amp):
            recon_logits = recon_decoder(gvt_tokens_tensor, num_slices=N_recon)
            recon_loss_val = gvt_recon_loss(recon_logits, gt_vol, recon_res=args.gvt_recon_res)
        recon_loss_scalar = recon_loss_val.detach()

    # -------------------------------------------------------------------------
    # Phase 1 + Phase 2: TSDF auxiliary loss
    # -------------------------------------------------------------------------
    tsdf_loss_val = None
    tsdf_loss_scalar = torch.tensor(0.0, device=device)

    if tsdf_decoder is not None and has_gvt(base_model) and len(active_labels) > 0:
        wrapper = base_model.memory_attention
        gvt_tokens_raw, _ = wrapper.gvt_tokens_cache   # (K, 1, C_mem)
        # Reshape: (K, 1, C_mem) → (1, K, C_mem) for TSDFDecoder
        gvt_tokens_for_tsdf = gvt_tokens_raw.transpose(0, 1)  # (1, K, C_mem)

        N_total = image_embeds.shape[0] if hasattr(image_embeds, "shape") else len(image_embeds)

        # Determine which slices to use for TSDF (capped at max_slices)
        all_slice_indices = list(range(N_total))
        tsdf_slice_indices = subsample_slices(all_slice_indices, max_slices)

        # Prepare TSDF ground truth targets (on CPU first, move to device later)
        tsdf_tau = getattr(args, 'tsdf_tau', 32.0)
        if isinstance(masks, np.ndarray):
            tsdf_targets = prepare_tsdf_targets(
                gt_masks_3d=masks,
                label_ids=active_labels,
                target_size=256,
                tau=tsdf_tau,
            )
        else:
            # Convert to numpy if needed
            masks_np = masks.numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
            tsdf_targets = prepare_tsdf_targets(
                gt_masks_3d=masks_np,
                label_ids=active_labels,
                target_size=256,
                tau=tsdf_tau,
            )

        # Phase 1: collect prompted memory features per label (no_grad)
        # Use per-label cut slice as init frame — guaranteed to contain the label.
        phase1_mem_features: Dict[int, Optional[torch.Tensor]] = {}
        with torch.no_grad():
            for label in active_labels:
                fg_counts_p1 = (masks_np == label).sum(axis=(1, 2))
                if fg_counts_p1.max() == 0:
                    phase1_mem_features[label] = None
                    continue
                init_slice = int(np.argmax(fg_counts_p1))

                label_arr = masks_np[init_slice]
                binary = (label_arr == label).astype(np.float32)

                # Build bounding box from GT
                rows = np.any(binary, axis=1)
                cols = np.any(binary, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                scale_h = 256.0 / ori_h
                scale_w = 256.0 / ori_w
                box = np.array([
                    cmin * scale_w, rmin * scale_h,
                    cmax * scale_w, rmax * scale_h
                ], dtype=np.float32)
                box_t = torch.from_numpy(box).to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, 4)

                try:
                    embed = image_embeds[init_slice:init_slice+1]
                    if isinstance(embed, np.ndarray):
                        embed_t = torch.from_numpy(embed).to(device)
                    else:
                        embed_t = embed.to(device)

                    hr0 = high_res_feat_0s[init_slice:init_slice+1]
                    hr1 = high_res_feat_1s[init_slice:init_slice+1]
                    if isinstance(hr0, np.ndarray):
                        hr0 = torch.from_numpy(hr0).to(device)
                        hr1 = torch.from_numpy(hr1).to(device)
                    else:
                        hr0 = hr0.to(device)
                        hr1 = hr1.to(device)

                    with torch.amp.autocast("cuda", enabled=args.use_amp):
                        # Encode prompt
                        sparse_embeddings, dense_embeddings = predictor.sam_prompt_encoder(
                            points=None,
                            boxes=box_t,
                            masks=None,
                        )

                        # Decode mask; capture object_score_logits for _encode_new_memory
                        low_res_masks, _, _, object_score_logits = predictor.sam_mask_decoder(
                            image_embeddings=embed_t,
                            image_pe=predictor.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            repeat_image=False,
                            high_res_features=[hr0, hr1],
                        )

                        # _encode_new_memory feeds masks through MaskDownSampler
                        # (total_stride=16, strided convolutions). It expects input at
                        # image_size × image_size (e.g. 1024×1024) so it downsamples to
                        # 64×64 matching pix_feat. Passing low_res_masks (256×256) would
                        # downsample to 16×16, causing a shape mismatch with pix_feat.
                        # Replicate _forward_sam_heads: upsample to image_size first.
                        image_size = predictor.image_size
                        high_res_masks = F.interpolate(
                            low_res_masks.float(),
                            size=(image_size, image_size),
                            mode="bilinear",
                            align_corners=False,
                        )

                        # Encode prediction into memory features
                        # current_vision_feats: list of (HW, B, C) tensors per scale
                        feat_hw = embed_t.shape[2], embed_t.shape[3]
                        current_vision_feats = [embed_t.flatten(2).permute(2, 0, 1)]
                        maskmem_features, _ = predictor._encode_new_memory(
                            current_vision_feats=current_vision_feats,
                            feat_sizes=[feat_hw],
                            pred_masks_high_res=high_res_masks,
                            object_score_logits=object_score_logits,
                            is_mask_from_pts=False,
                        )

                    # maskmem_features: (1, C_mem, H_mem, W_mem) → (1, H*W, C_mem)
                    mem_flat = maskmem_features.detach().flatten(2).permute(0, 2, 1)
                    phase1_mem_features[label] = mem_flat

                except Exception as e:
                    import traceback
                    print(f"[Phase1] label={label} init_slice={init_slice} failed: {e}\n"
                          f"{traceback.format_exc()}")
                    phase1_mem_features[label] = None

        # Phase 2: compute TSDF losses per label
        tsdf_label_losses = []
        tsdf_alpha = getattr(args, 'tsdf_alpha', 4.0)

        # Pre-compute per-phase constants (same for all labels)
        slice_idx_tensor = torch.tensor(tsdf_slice_indices, dtype=torch.long, device=device)
        # float32 cast for cross-attention stability (GVT tokens are small, no OOM risk)
        gvt_tokens_fp32 = gvt_tokens_for_tsdf.float()
        masks_np = masks if isinstance(masks, np.ndarray) else (
            masks.numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
        )

        for label in active_labels:
            mem_feat = phase1_mem_features.get(label, None)
            if mem_feat is None:
                continue

            # Get TSDF targets for this label at the subsampled slices
            if label not in tsdf_targets:
                continue
            tsdf_gt_full = tsdf_targets[label]  # (D, 256, 256)
            tsdf_gt_sub = tsdf_gt_full[tsdf_slice_indices].to(device)  # (N_tsdf, 256, 256)
            tsdf_gt_sub = tsdf_gt_sub.unsqueeze(0)  # (1, N_tsdf, 256, 256)

            # Compute label area for normalization
            label_area = float((masks_np == label).sum())

            mem_feat_fp32 = mem_feat.float()

            with torch.amp.autocast("cuda", enabled=args.use_amp):
                tsdf_pred = tsdf_decoder(
                    gvt_tokens=gvt_tokens_fp32,
                    mem_features=mem_feat_fp32,
                    slice_indices=slice_idx_tensor,
                )  # (1, N_tsdf, 256, 256)

                # Cast GT to same dtype as pred
                tsdf_gt_sub = tsdf_gt_sub.to(tsdf_pred.dtype)

                loss_label = tsdf_loss(
                    tsdf_pred=tsdf_pred,
                    tsdf_gt=tsdf_gt_sub,
                    alpha=tsdf_alpha,
                    sigma=tsdf_tau / 3.0,
                    # Normalize by sqrt(area) when training multiple labels to balance
                    # small vs large structures. Skip normalization in single-label mode.
                    label_area=label_area if not getattr(args, 'tsdf_single_label', False) else None,
                )
                tsdf_label_losses.append(loss_label)

        if tsdf_label_losses:
            tsdf_loss_val = torch.stack(tsdf_label_losses).mean()
            tsdf_loss_scalar = tsdf_loss_val.detach()

    # -------------------------------------------------------------------------
    # Phase 3: standard seg loss (unchanged from GVT v2)
    # -------------------------------------------------------------------------
    all_losses = []
    all_vol_losses = []

    for label in active_labels:
        # Per-label cut slice: slice with the most foreground for this label
        fg_counts_L = (masks_np == label).sum(axis=(1, 2))
        if fg_counts_L.max() == 0:
            continue
        cut_idx_L = int(np.argmax(fg_counts_L))
        v1_L = list(range(cut_idx_L, -1, -1))
        v2_L = [cut_idx_L] + list(range(cut_idx_L + 1, N_total))
        v1_L_train = subsample_slices(v1_L, max_slices)
        v2_L_train = subsample_slices(v2_L, max_slices)

        if len(v1_L_train) > 0:
            loss1, n1, vol_loss1 = process_video_with_memory_3dloss(
                model=base_model,
                image_embeds=image_embeds,
                high_res_feat_0s=high_res_feat_0s,
                high_res_feat_1s=high_res_feat_1s,
                masks=masks,
                slice_indices=v1_L_train,
                label=label,
                ori_h=ori_h,
                ori_w=ori_w,
                loss_fn=loss_fn,
                device=device,
                use_amp=args.use_amp,
                lambda_2d=args.lambda_2d,
                lambda_3d=args.lambda_3d,
            )
            if n1 > 0 and loss1.requires_grad:
                all_losses.append(loss1)
                all_vol_losses.append(vol_loss1.detach())

        if len(v2_L_train) > 1:
            v2_remaining = v2_L_train[1:]
            loss2, n2, vol_loss2 = process_video_with_memory_3dloss(
                model=base_model,
                image_embeds=image_embeds,
                high_res_feat_0s=high_res_feat_0s,
                high_res_feat_1s=high_res_feat_1s,
                masks=masks,
                slice_indices=v2_remaining,
                label=label,
                ori_h=ori_h,
                ori_w=ori_w,
                loss_fn=loss_fn,
                device=device,
                use_amp=args.use_amp,
                lambda_2d=args.lambda_2d,
                lambda_3d=args.lambda_3d,
            )
            if n2 > 0 and loss2.requires_grad:
                all_losses.append(loss2)
                all_vol_losses.append(vol_loss2.detach())

    # Clear GVT cache after processing volume
    if has_gvt(base_model):
        clear_gvt_cache(base_model)

    # -------------------------------------------------------------------------
    # Combine all losses: total = L_seg + lambda_aux * L_recon + lambda_tsdf * L_tsdf
    # All TSDF and recon losses are ADDITIVE (never averaged into seg losses)
    # -------------------------------------------------------------------------
    avg_vol_loss = (torch.stack(all_vol_losses).mean()
                   if all_vol_losses else torch.tensor(0.0, device=device))

    lambda_tsdf = getattr(args, 'lambda_tsdf', 0.1)

    if all_losses:
        seg_loss = torch.stack(all_losses).mean()
        total_loss = seg_loss

        if recon_loss_val is not None and recon_loss_val.requires_grad:
            current_lambda_aux = get_lambda_aux(epoch, args)
            total_loss = total_loss + current_lambda_aux * recon_loss_val

        if tsdf_loss_val is not None and tsdf_loss_val.requires_grad:
            total_loss = total_loss + lambda_tsdf * tsdf_loss_val

        return total_loss, avg_vol_loss, recon_loss_scalar, tsdf_loss_scalar

    # No seg losses — still train GVT encoder via recon/TSDF if available
    aux_loss = None
    if recon_loss_val is not None and recon_loss_val.requires_grad:
        current_lambda_aux = get_lambda_aux(epoch, args)
        aux_loss = current_lambda_aux * recon_loss_val
    if tsdf_loss_val is not None and tsdf_loss_val.requires_grad:
        tsdf_term = lambda_tsdf * tsdf_loss_val
        aux_loss = tsdf_term if aux_loss is None else aux_loss + tsdf_term

    if aux_loss is not None:
        return aux_loss, avg_vol_loss, recon_loss_scalar, tsdf_loss_scalar

    return (
        torch.tensor(0.0, device=device, requires_grad=True),
        torch.tensor(0.0, device=device),
        recon_loss_scalar,
        tsdf_loss_scalar,
    )


# =============================================================================
# Training Epoch
# =============================================================================

def train_epoch(
    model,
    predictor,
    dataloader,
    optimizer,
    loss_fn,
    args,
    device,
    rank,
    scaler=None,
    recon_decoder=None,
    tsdf_decoder=None,
    epoch: int = 0,
):
    """Train for one epoch with GVT, 3D loss, and optional TSDF loss."""
    model.train()
    if recon_decoder is not None:
        recon_decoder.train()
    if tsdf_decoder is not None:
        tsdf_decoder.train()

    epoch_loss = 0.0
    loss_2d_total = 0.0
    loss_3d_total = 0.0
    loss_vol_total = 0.0
    loss_recon_total = 0.0
    loss_tsdf_total = 0.0
    num_2d = 0
    num_3d = 0
    pending_grads = False

    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0))

    for batch_idx, batch in enumerate(pbar):
        samples_2d = batch["samples_2d"]
        samples_3d = batch["samples_3d"]

        batch_losses = []
        batch_vol_losses = []
        batch_recon_losses = []
        batch_tsdf_losses = []

        for sample in samples_2d:
            loss = train_2d_sample_from_embedding(model, sample, loss_fn, args, device)
            if not torch.isnan(loss) and loss.item() > 0:
                batch_losses.append(loss)
                loss_2d_total += loss.item()
                num_2d += 1

        for sample in samples_3d:
            loss, vol_loss, recon_loss, tsdf_loss_val = train_3d_sample_from_embedding_gvt_tsdf(
                model=model,
                predictor=predictor,
                sample=sample,
                loss_fn=loss_fn,
                args=args,
                device=device,
                recon_decoder=recon_decoder,
                tsdf_decoder=tsdf_decoder,
                epoch=epoch,
            )
            if not torch.isnan(loss) and loss.item() > 0:
                batch_losses.append(loss)
                batch_vol_losses.append(vol_loss.item())
                batch_recon_losses.append(recon_loss.item())
                batch_tsdf_losses.append(tsdf_loss_val.item())
                loss_3d_total += loss.item()
                loss_vol_total += vol_loss.item()
                loss_recon_total += recon_loss.item()
                loss_tsdf_total += tsdf_loss_val.item()
                num_3d += 1

        if not batch_losses:
            if rank == 0:
                pbar.set_postfix({"loss": "N/A", "2d": num_2d, "3d": num_3d})
            continue

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss = batch_loss / args.grad_accum_steps

        # Collect all trainable params
        all_params = list(model.parameters())
        if recon_decoder is not None:
            all_params = all_params + list(recon_decoder.parameters())
        if tsdf_decoder is not None:
            all_params = all_params + list(tsdf_decoder.parameters())

        if args.use_amp and scaler is not None:
            scaler.scale(batch_loss).backward()
            pending_grads = True
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pending_grads = False
        else:
            batch_loss.backward()
            pending_grads = True
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                pending_grads = False

        epoch_loss += batch_loss.item() * args.grad_accum_steps

        avg_vol = np.mean(batch_vol_losses) if batch_vol_losses else 0.0
        avg_recon = np.mean(batch_recon_losses) if batch_recon_losses else 0.0
        avg_tsdf = np.mean(batch_tsdf_losses) if batch_tsdf_losses else 0.0
        if rank == 0:
            postfix = {
                "loss": f"{batch_loss.item() * args.grad_accum_steps:.4f}",
                "vol": f"{avg_vol:.4f}",
                "2d": num_2d,
                "3d": num_3d,
            }
            if recon_decoder is not None:
                postfix["recon"] = f"{avg_recon:.4f}"
            if tsdf_decoder is not None:
                postfix["tsdf"] = f"{avg_tsdf:.4f}"
            pbar.set_postfix(postfix)

    if pending_grads:
        all_params = list(model.parameters())
        if recon_decoder is not None:
            all_params = all_params + list(recon_decoder.parameters())
        if tsdf_decoder is not None:
            all_params = all_params + list(tsdf_decoder.parameters())
        if args.use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = epoch_loss / max(len(dataloader), 1)
    avg_loss_2d = loss_2d_total / max(num_2d, 1)
    avg_loss_3d = loss_3d_total / max(num_3d, 1)
    avg_loss_vol = loss_vol_total / max(num_3d, 1)
    avg_loss_recon = loss_recon_total / max(num_3d, 1)
    avg_loss_tsdf = loss_tsdf_total / max(num_3d, 1)

    return avg_loss, avg_loss_2d, avg_loss_3d, avg_loss_vol, avg_loss_recon, avg_loss_tsdf


# =============================================================================
# Checkpoint helpers
# =============================================================================

def get_latest_checkpoint(save_path: str, initial_ckpt: str) -> Tuple[str, int]:
    """Find the latest GVT TSDF checkpoint in save_path."""
    pattern = os.path.join(save_path, "sam2_gvt_tsdf_epoch_*.pth")
    ckpt_files = glob(pattern)

    if not ckpt_files:
        return initial_ckpt, 0

    epochs = []
    for f in ckpt_files:
        try:
            basename = os.path.basename(f)
            epoch_str = basename.replace("sam2_gvt_tsdf_epoch_", "").replace(".pth", "")
            epochs.append((int(epoch_str), f))
        except (ValueError, IndexError):
            continue

    if not epochs:
        return initial_ckpt, 0

    epochs.sort(key=lambda x: x[0], reverse=True)
    return epochs[0][1], epochs[0][0]


# =============================================================================
# CLI and Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="GVT v3 Training: SAM2 from 3D embeddings with GVT + 3D Loss + TSDF shape prior"
    )

    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--pretrained-ckpt", type=str, required=True)
    parser.add_argument("--embedding-dir", type=str, required=True)

    # GVT configuration
    parser.add_argument("--use-gvt", action="store_true")
    parser.add_argument("--gvt-num-tokens", type=int, default=16)
    parser.add_argument("--gvt-num-layers", type=int, default=2)
    parser.add_argument("--gvt-num-heads", type=int, default=4)

    # GVT v2: auxiliary reconstruction loss
    parser.add_argument("--gvt-recon", action="store_true",
                        help="Enable reconstruction loss (GVT v2). Only effective with --use-gvt.")
    parser.add_argument("--gvt-recon-res", type=int, default=16,
                        help="Low-resolution target size for reconstruction (default: 16).")
    parser.add_argument("--lambda-aux", type=float, default=1.0,
                        help="Weight for the reconstruction loss (default: 1.0).")
    parser.add_argument("--lambda-aux-warmup", type=float, default=0.0,
                        help="λ_aux used during warmup epochs. 0 disables warmup (default: 0).")
    parser.add_argument("--lambda-aux-warmup-epochs", type=int, default=10,
                        help="Number of warmup epochs for λ_aux schedule (default: 10).")

    # GVT v3: TSDF shape prior auxiliary loss
    parser.add_argument("--use-tsdf", action="store_true",
                        help="Enable TSDF shape prior auxiliary loss (GVT v3).")
    parser.add_argument("--lambda-tsdf", type=float, default=0.1,
                        help="Weight for the TSDF auxiliary loss (default: 0.1, additive).")
    parser.add_argument("--tsdf-tau", type=float, default=32.0,
                        help="TSDF truncation threshold in pixels (default: 32.0).")
    parser.add_argument("--tsdf-alpha", type=float, default=4.0,
                        help="Boundary upweighting factor for TSDF loss (default: 4.0).")
    parser.add_argument("--tsdf-single-label", action="store_true",
                        help="Single-label mode: disable per-label sqrt(area) normalization "
                             "(normalization is only meaningful when multiple labels are active).")

    # Sampling
    parser.add_argument("--mix-ratio-3d", type=float, default=0.8)
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    parser.add_argument("--max-slices-per-video", type=int, default=32)
    parser.add_argument("--max-labels-per-sample", type=int, default=1)

    # Training
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Loss weights
    parser.add_argument("--lambda-2d", type=float, default=1.0)
    parser.add_argument("--lambda-3d", type=float, default=0.5)
    parser.add_argument("--lambda-dice", type=float, default=1.0)
    parser.add_argument("--lambda-focal", type=float, default=1.0)

    # Hardware
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--master-port", type=int, default=None)

    # Checkpointing
    parser.add_argument("--save-every", type=int, default=5)

    return parser.parse_args()


def main(rank, world_size, args):
    ddp_setup(rank, world_size)
    device = f"cuda:{rank}"

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    ckpt_path, start_epoch = get_latest_checkpoint(args.save_path, args.pretrained_ckpt)
    resume_training = start_epoch > 0

    if rank == 0:
        if resume_training:
            print(f"Resuming from epoch {start_epoch}, checkpoint: {ckpt_path}")
        else:
            print(f"Starting fresh training from checkpoint: {ckpt_path}")

    # Build SAM2 model
    if resume_training and args.use_gvt:
        model = build_sam2(
            config_file=args.sam2_config,
            ckpt_path=None,
            device=device,
            mode="train",
            apply_postprocessing=False,
        )
        freeze_unified_components(model)
        model = wrap_model_with_gvt(
            model,
            num_tokens=args.gvt_num_tokens,
            num_layers=args.gvt_num_layers,
            num_heads=args.gvt_num_heads,
            device=device,
        )
        load_gvt_checkpoint(model, ckpt_path, device=device)
        if rank == 0:
            print(f"Loaded GVT checkpoint: {ckpt_path}")
    else:
        model = build_sam2(
            config_file=args.sam2_config,
            ckpt_path=ckpt_path,
            device=device,
            mode="train",
            apply_postprocessing=False,
        )
        freeze_unified_components(model)
        if args.use_gvt:
            model = wrap_model_with_gvt(
                model,
                num_tokens=args.gvt_num_tokens,
                num_layers=args.gvt_num_layers,
                num_heads=args.gvt_num_heads,
                device=device,
            )

    if args.use_gvt and rank == 0:
        gvt_params = sum(p.numel() for p in model.memory_attention.gvt_module.parameters())
        print(f"GVT enabled: {args.gvt_num_tokens} tokens, "
              f"{args.gvt_num_layers} layers, {gvt_params:,} parameters")

    # GVT v2: reconstruction decoder
    recon_decoder = None
    if args.use_gvt and args.gvt_recon:
        mem_dim = model.mem_dim  # 64
        recon_decoder = GVTReconDecoder(
            C_mem=mem_dim,
            max_slices=256,
            recon_res=args.gvt_recon_res,
        ).to(device)
        if resume_training:
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
            if "gvt_recon_decoder" in sd:
                recon_decoder.load_state_dict(sd["gvt_recon_decoder"])
                if rank == 0:
                    print("Loaded GVTReconDecoder weights from checkpoint.")
            elif rank == 0:
                print("No GVTReconDecoder weights in checkpoint; starting decoder fresh.")
            del sd
        if rank == 0:
            recon_params = sum(p.numel() for p in recon_decoder.parameters())
            print(f"GVT v2 recon decoder enabled: res={args.gvt_recon_res}, "
                  f"{recon_params:,} parameters, λ_aux={args.lambda_aux}")
            if args.lambda_aux_warmup > 0:
                print(f"  λ_aux warmup: {args.lambda_aux_warmup} for first "
                      f"{args.lambda_aux_warmup_epochs} epochs")

    # GVT v3: TSDF decoder
    tsdf_decoder = None
    if args.use_gvt and args.use_tsdf:
        mem_dim = model.mem_dim  # 64
        tsdf_decoder = TSDFDecoder(
            num_tokens=args.gvt_num_tokens,
            token_dim=mem_dim,
            max_slices=256,
            seed_channels=64,
            seed_size=4,
            mem_feature_dim=mem_dim,
        ).to(device)
        if resume_training:
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
            if "tsdf_decoder" in sd:
                tsdf_decoder.load_state_dict(sd["tsdf_decoder"])
                if rank == 0:
                    print("Loaded TSDFDecoder weights from checkpoint.")
            elif rank == 0:
                print("No TSDFDecoder weights in checkpoint; starting decoder fresh.")
            del sd
        if rank == 0:
            tsdf_params = sum(p.numel() for p in tsdf_decoder.parameters())
            print(f"GVT v3 TSDF decoder enabled: tau={args.tsdf_tau}, alpha={args.tsdf_alpha}, "
                  f"{tsdf_params:,} parameters, λ_tsdf={args.lambda_tsdf}")

    if rank == 0:
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Loss weights: lambda_2d={args.lambda_2d}, lambda_3d={args.lambda_3d}")

    # Keep a reference to the unwrapped predictor for Phase 1 memory feature extraction
    # (before DDP wrapping, which complicates direct attribute access)
    predictor = model

    model = DDP(
        model,
        device_ids=[rank],
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
    )

    # Include all auxiliary decoder parameters in the optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    if recon_decoder is not None:
        trainable = trainable + list(recon_decoder.parameters())
    if tsdf_decoder is not None:
        trainable = trainable + list(tsdf_decoder.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if resume_training:
        sd_resume = torch.load(ckpt_path, map_location=device, weights_only=True)
        if "optimizer" in sd_resume:
            optimizer.load_state_dict(sd_resume["optimizer"])
            if rank == 0:
                print("Loaded optimizer state from checkpoint.")
        del sd_resume

    loss_fn = DiceFocalLoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
        lambda_dice=args.lambda_dice,
        lambda_focal=args.lambda_focal,
    )

    if args.use_amp:
        scaler = (torch.amp.GradScaler(device)
                  if hasattr(torch.amp, "GradScaler")
                  else torch.cuda.amp.GradScaler())
    else:
        scaler = None

    if rank == 0:
        print("\nLoading 3D embedding dataset...")

    emb_3d_dataset = Embedding3DDataset(
        embedding_dir=args.embedding_dir,
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
    )

    mixed_dataset = MixedEmbeddingDataset(
        embedding_3d_dataset=emb_3d_dataset,
        mix_ratio_3d=args.mix_ratio_3d,
        seed=args.seed,
    )

    train_sampler = DistributedSampler(
        mixed_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
    )

    train_loader = DataLoader(
        mixed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=mixed_emb_collate_fn,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    csv_path = os.path.join(args.save_path, "training_log.csv")
    csv_file = None
    csv_writer = None
    if rank == 0:
        csv_exists = os.path.exists(csv_path) and resume_training
        csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow([
                "epoch", "train_loss", "loss_2d", "loss_3d",
                "loss_vol", "loss_recon", "loss_tsdf"
            ])
        csv_file.flush()

    if rank == 0:
        print(f"\nStarting GVT v3 training for {args.num_epochs} epochs...")

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"{'='*60}")

        mixed_dataset.reshuffle(epoch)

        (avg_loss, avg_loss_2d, avg_loss_3d,
         avg_loss_vol, avg_loss_recon, avg_loss_tsdf) = train_epoch(
            model=model,
            predictor=predictor,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            args=args,
            device=device,
            rank=rank,
            scaler=scaler,
            recon_decoder=recon_decoder,
            tsdf_decoder=tsdf_decoder,
            epoch=epoch,
        )

        if rank == 0:
            recon_str = f", Recon={avg_loss_recon:.4f}" if recon_decoder is not None else ""
            tsdf_str = f", TSDF={avg_loss_tsdf:.4f}" if tsdf_decoder is not None else ""
            print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, 2D={avg_loss_2d:.4f}, "
                  f"3D={avg_loss_3d:.4f}, Vol={avg_loss_vol:.4f}{recon_str}{tsdf_str}")
            if recon_decoder is not None:
                current_lambda_aux = get_lambda_aux(epoch, args)
                print(f"  λ_aux={current_lambda_aux:.4f}")
            if tsdf_decoder is not None:
                print(f"  λ_tsdf={args.lambda_tsdf:.4f}")

            if csv_writer is not None:
                csv_writer.writerow([
                    epoch + 1,
                    f"{avg_loss:.6f}",
                    f"{avg_loss_2d:.6f}",
                    f"{avg_loss_3d:.6f}",
                    f"{avg_loss_vol:.6f}",
                    f"{avg_loss_recon:.6f}",
                    f"{avg_loss_tsdf:.6f}",
                ])
                csv_file.flush()

            if (epoch + 1) % args.save_every == 0:
                save_name = os.path.join(args.save_path, f"sam2_gvt_tsdf_epoch_{epoch + 1}.pth")
                ckpt = {
                    "model": model.module.state_dict(),
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                    "gvt_config": {
                        "use_gvt": args.use_gvt,
                        "num_tokens": args.gvt_num_tokens,
                        "num_layers": args.gvt_num_layers,
                        "num_heads": args.gvt_num_heads,
                    } if args.use_gvt else None,
                    "tsdf_config": {
                        "lambda_tsdf": args.lambda_tsdf,
                        "tsdf_tau": args.tsdf_tau,
                        "tsdf_alpha": args.tsdf_alpha,
                    } if args.use_tsdf else None,
                }
                if recon_decoder is not None:
                    ckpt["gvt_recon_decoder"] = recon_decoder.state_dict()
                if tsdf_decoder is not None:
                    ckpt["tsdf_decoder"] = tsdf_decoder.state_dict()
                torch.save(ckpt, save_name)
                print(f"Checkpoint saved: {save_name}")

    if rank == 0:
        save_name = os.path.join(args.save_path, "sam2_gvt_tsdf_final.pth")
        ckpt = {
            "model": model.module.state_dict(),
            "epoch": args.num_epochs,
            "optimizer": optimizer.state_dict(),
            "gvt_config": {
                "use_gvt": args.use_gvt,
                "num_tokens": args.gvt_num_tokens,
                "num_layers": args.gvt_num_layers,
                "num_heads": args.gvt_num_heads,
            } if args.use_gvt else None,
            "tsdf_config": {
                "lambda_tsdf": args.lambda_tsdf,
                "tsdf_tau": args.tsdf_tau,
                "tsdf_alpha": args.tsdf_alpha,
            } if args.use_tsdf else None,
        }
        if recon_decoder is not None:
            ckpt["gvt_recon_decoder"] = recon_decoder.state_dict()
        if tsdf_decoder is not None:
            ckpt["tsdf_decoder"] = tsdf_decoder.state_dict()
        torch.save(ckpt, save_name)
        print(f"Final checkpoint saved: {save_name}")

        if csv_file is not None:
            csv_file.close()

    destroy_process_group()


if __name__ == "__main__":
    import random as py_random

    args = parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")

    os.environ["MASTER_ADDR"] = "localhost"
    if args.master_port is not None:
        os.environ["MASTER_PORT"] = str(args.master_port)
    elif "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(py_random.randint(29500, 29999))

    gvt_str = " with GVT" if args.use_gvt else ""
    recon_str = " + recon (v2)" if (args.use_gvt and args.gvt_recon) else ""
    tsdf_str = " + TSDF (v3)" if (args.use_gvt and args.use_tsdf) else ""
    print(f"Starting GVT v3 training{gvt_str}{recon_str}{tsdf_str} with {world_size} GPUs")
    print(f"Loss weighting: lambda_2d={args.lambda_2d}, lambda_3d={args.lambda_3d}, "
          f"lambda_tsdf={args.lambda_tsdf if args.use_tsdf else 'N/A'}")

    mp.spawn(main, args=(world_size, args), nprocs=world_size)
