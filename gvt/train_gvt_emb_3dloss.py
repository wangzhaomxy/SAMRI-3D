#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_gvt_emb_3dloss.py

Training from pre-computed 3D embeddings with GVT and combined 2D + 3D volumetric loss.

Extends train_unified_emb_3dloss.py by adding GVT (Global Volume Token).

Usage:
    python gvt/train_gvt_emb_3dloss.py \\
        --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
        --pretrained-ckpt /path/to/pretrained.pth \\
        --embedding-dir /path/to/3D_embeddings/ \\
        --save-path /path/to/gvt_3dloss_checkpoints/ \\
        --use-gvt --gvt-num-tokens 4 --gvt-num-layers 2 \\
        --lambda-2d 1.0 --lambda-3d 0.5 \\
        --batch-size 2 --num-epochs 50 --lr 5e-6 --use-amp
"""

import os
import sys
import csv
import argparse
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

# Prefer local ./sam2 and project root
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2
from utils.losses import DiceFocalLoss
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
    dice_loss_3d,
    train_2d_sample_from_embedding,
    process_video_with_memory_3dloss,
)


def get_latest_checkpoint(save_path: str, initial_ckpt: str) -> Tuple[str, int]:
    """Find the latest GVT 3D-loss checkpoint in save_path."""
    pattern = os.path.join(save_path, "sam2_gvt3dloss_epoch_*.pth")
    ckpt_files = glob(pattern)

    if not ckpt_files:
        return initial_ckpt, 0

    epochs = []
    for f in ckpt_files:
        try:
            basename = os.path.basename(f)
            epoch_str = basename.replace("sam2_gvt3dloss_epoch_", "").replace(".pth", "")
            epochs.append((int(epoch_str), f))
        except (ValueError, IndexError):
            continue

    if not epochs:
        return initial_ckpt, 0

    epochs.sort(key=lambda x: x[0], reverse=True)
    return epochs[0][1], epochs[0][0]


# =============================================================================
# GVT-aware 3D Sample Training with 3D Loss
# =============================================================================

def train_3d_sample_from_embedding_gvt(
    model,
    sample: Dict,
    loss_fn,
    args,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Train on a 3D volume with GVT and combined 2D + 3D volumetric loss.
    """
    image_embeds = sample["image_embeds"]
    high_res_feat_0s = sample["high_res_feat_0s"]
    high_res_feat_1s = sample["high_res_feat_1s"]
    masks = sample["masks"]
    labels = sample["labels"]
    cut_idx = sample["cut_idx"]
    v1_slices = sample["v1_slices"]
    v2_slices = sample["v2_slices"]
    ori_h, ori_w = sample["ori_h"], sample["ori_w"]

    base_model = model.module if hasattr(model, 'module') else model

    # Compute GVT tokens from all slice embeddings
    if has_gvt(base_model):
        compute_gvt_tokens(base_model, image_embeds, device, use_amp=args.use_amp)

    # Subsample video slices if too long
    max_slices = getattr(args, 'max_slices_per_video', 32)

    def subsample_slices(slices: List[int], max_len: int) -> List[int]:
        if len(slices) <= max_len:
            return slices
        step = max(1, (len(slices) - 1) // (max_len - 1))
        subsampled = [slices[0]] + slices[1::step][:max_len - 1]
        return subsampled

    v1_slices_train = subsample_slices(v1_slices, max_slices)
    v2_slices_train = subsample_slices(v2_slices, max_slices)

    all_losses = []
    all_vol_losses = []
    total_slices = 0

    max_labels_per_sample = getattr(args, 'max_labels_per_sample', 1)
    if len(labels) > max_labels_per_sample:
        rng = np.random.RandomState()
        labels = rng.choice(labels, size=max_labels_per_sample, replace=False).tolist()

    for label in labels:
        if (masks[cut_idx] == label).sum() == 0:
            continue

        if len(v1_slices_train) > 0:
            loss1, n1, vol_loss1 = process_video_with_memory_3dloss(
                model=base_model,
                image_embeds=image_embeds,
                high_res_feat_0s=high_res_feat_0s,
                high_res_feat_1s=high_res_feat_1s,
                masks=masks,
                slice_indices=v1_slices_train,
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
                all_vol_losses.append(vol_loss1)
                total_slices += n1

        if len(v2_slices_train) > 1:
            v2_remaining = v2_slices_train[1:]
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
                all_vol_losses.append(vol_loss2)
                total_slices += n2

    # Clear GVT cache
    if has_gvt(base_model):
        clear_gvt_cache(base_model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if all_losses and total_slices > 0:
        total_loss = torch.stack(all_losses).mean()
        avg_vol_loss = torch.stack(all_vol_losses).mean() if all_vol_losses else torch.tensor(0.0, device=device)
        return total_loss, avg_vol_loss
    return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)


# =============================================================================
# Training Epoch
# =============================================================================

def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    args,
    device,
    rank,
    scaler=None,
):
    """Train for one epoch with GVT and 3D loss."""
    model.train()

    epoch_loss = 0.0
    loss_2d_total = 0.0
    loss_3d_total = 0.0
    loss_vol_total = 0.0
    num_2d = 0
    num_3d = 0
    pending_grads = False

    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0))

    for batch_idx, batch in enumerate(pbar):
        samples_2d = batch["samples_2d"]
        samples_3d = batch["samples_3d"]

        batch_losses = []
        batch_vol_losses = []

        for sample in samples_2d:
            loss = train_2d_sample_from_embedding(model, sample, loss_fn, args, device)
            if not torch.isnan(loss) and loss.item() > 0:
                batch_losses.append(loss)
                loss_2d_total += loss.item()
                num_2d += 1

        for sample in samples_3d:
            loss, vol_loss = train_3d_sample_from_embedding_gvt(model, sample, loss_fn, args, device)
            if not torch.isnan(loss) and loss.item() > 0:
                batch_losses.append(loss)
                batch_vol_losses.append(vol_loss.item())
                loss_3d_total += loss.item()
                loss_vol_total += vol_loss.item()
                num_3d += 1

        if not batch_losses:
            if rank == 0:
                pbar.set_postfix({"loss": "N/A", "2d": num_2d, "3d": num_3d})
            continue

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss = batch_loss / args.grad_accum_steps

        if args.use_amp and scaler is not None:
            scaler.scale(batch_loss).backward()
            pending_grads = True
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pending_grads = False
        else:
            batch_loss.backward()
            pending_grads = True
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                pending_grads = False

        epoch_loss += batch_loss.item() * args.grad_accum_steps

        avg_vol = np.mean(batch_vol_losses) if batch_vol_losses else 0.0
        if rank == 0:
            pbar.set_postfix({
                "loss": f"{batch_loss.item() * args.grad_accum_steps:.4f}",
                "vol": f"{avg_vol:.4f}",
                "2d": num_2d,
                "3d": num_3d,
            })

    if pending_grads:
        if args.use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = epoch_loss / max(len(dataloader), 1)
    avg_loss_2d = loss_2d_total / max(num_2d, 1)
    avg_loss_3d = loss_3d_total / max(num_3d, 1)
    avg_loss_vol = loss_vol_total / max(num_3d, 1)

    return avg_loss, avg_loss_2d, avg_loss_3d, avg_loss_vol


# =============================================================================
# CLI and Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="GVT Training with 3D Loss: Train SAM2 from 3D embeddings with GVT + volumetric Dice"
    )

    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--pretrained-ckpt", type=str, required=True)
    parser.add_argument("--embedding-dir", type=str, required=True)

    # GVT configuration
    parser.add_argument("--use-gvt", action="store_true")
    parser.add_argument("--gvt-num-tokens", type=int, default=4)
    parser.add_argument("--gvt-num-layers", type=int, default=2)
    parser.add_argument("--gvt-num-heads", type=int, default=4)

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

    # When resuming from a GVT checkpoint, build without ckpt, wrap, then load
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

    if rank == 0:
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Loss weights: lambda_2d={args.lambda_2d}, lambda_3d={args.lambda_3d}")

    model = DDP(
        model,
        device_ids=[rank],
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = DiceFocalLoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
        lambda_dice=args.lambda_dice,
        lambda_focal=args.lambda_focal,
    )

    scaler = torch.amp.GradScaler(device) if args.use_amp else None

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
            csv_writer.writerow(["epoch", "train_loss", "loss_2d", "loss_3d", "loss_vol"])
        csv_file.flush()

    if rank == 0:
        print(f"\nStarting training for {args.num_epochs} epochs...")

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"{'='*60}")

        mixed_dataset.reshuffle(epoch)

        avg_loss, avg_loss_2d, avg_loss_3d, avg_loss_vol = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            args=args,
            device=device,
            rank=rank,
            scaler=scaler,
        )

        if rank == 0:
            print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, 2D={avg_loss_2d:.4f}, "
                  f"3D={avg_loss_3d:.4f}, Vol={avg_loss_vol:.4f}")

            if csv_writer is not None:
                csv_writer.writerow([
                    epoch + 1,
                    f"{avg_loss:.6f}",
                    f"{avg_loss_2d:.6f}",
                    f"{avg_loss_3d:.6f}",
                    f"{avg_loss_vol:.6f}",
                ])
                csv_file.flush()

            if (epoch + 1) % args.save_every == 0:
                save_name = os.path.join(args.save_path, f"sam2_gvt3dloss_epoch_{epoch + 1}.pth")
                torch.save({
                    "model": model.module.state_dict(),
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                    "gvt_config": {
                        "use_gvt": args.use_gvt,
                        "num_tokens": args.gvt_num_tokens,
                        "num_layers": args.gvt_num_layers,
                        "num_heads": args.gvt_num_heads,
                    } if args.use_gvt else None,
                }, save_name)
                print(f"Checkpoint saved: {save_name}")

    if rank == 0:
        save_name = os.path.join(args.save_path, "sam2_gvt3dloss_final.pth")
        torch.save({
            "model": model.module.state_dict(),
            "epoch": args.num_epochs,
            "optimizer": optimizer.state_dict(),
            "gvt_config": {
                "use_gvt": args.use_gvt,
                "num_tokens": args.gvt_num_tokens,
                "num_layers": args.gvt_num_layers,
                "num_heads": args.gvt_num_heads,
            } if args.use_gvt else None,
        }, save_name)
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
    print(f"Starting GVT + 3D-loss training{gvt_str} with {world_size} GPUs")
    print(f"Loss weighting: lambda_2d={args.lambda_2d}, lambda_3d={args.lambda_3d}")

    mp.spawn(main, args=(world_size, args), nprocs=world_size)
