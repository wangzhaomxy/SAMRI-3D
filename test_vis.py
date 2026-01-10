# -*- coding: utf-8 -*-
"""
SAM2 evaluation script.

Design goals:
- Focus only on SAM2 checkpoints (future SAMRI2 can be tested the same way).

Folder layout expected:
  .
  ├── test_vis.py
  ├── sam2/                    # official SAM2 package folder
  └── utils/
      ├── dataloader.py
      ├── losses.py
      ├── utils.py
      └── visual.py            # NOT used here (it imports SamPredictor)

Example:
  python test_vis.py \
    --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --ckpt-path /path/to/sam2_checkpoint.pt \
    --save-path /path/to/save/results.pkl \
    --test-file-paths /path/to/dataset/main-folder\
    --device cuda
    
    
python test_vis.py \
--sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
--ckpt-path /home/s4670484/Desktop/Scratch/samri/pretrained_ckpt/sam2.1_hiera_tiny.pt \
--save-path /home/s4670484/Desktop/Scratch/samri/Eval_results/SAM2/sam2_hiera_t.pkl \
--test-file-paths /home/s4670484/Desktop/Scratch/samri/Datasets/SAMRI_train_test/ \
--device cuda
    
    sam2_hiera_t.yaml     sam2.1_hiera_tiny.pt
    sam2_hiera_s.yaml     sam2.1_hiera_small.pt
    sam2_hiera_b+.yaml    sam2.1_hiera_base_plus.pt
    sam2_hiera_l.yaml     sam2.1_hiera_large.pt

"""

from __future__ import annotations

from glob import glob
import os
import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# -----------------------------
# Reuse utils code
# -----------------------------
from utils.dataloader import NiiDataset
from utils.utils import MaskSplit, gen_points, gen_bboxes
from utils.losses import dice_similarity, sd_hausdorff_distance, sd_mean_surface_distance


# -----------------------------
# Make sure local ./sam2 is importable
# (sam2 folder is sibling of this script)
# -----------------------------
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# -----------------------------
# Small SAM2-specific helpers
# -----------------------------
def _ensure_hwc_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """
    NiiDataset docstring says it returns HWC uint8 already.
    This is just a safety net in case your pipeline changes.
    """
    if image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8:
        return image

    # CHW -> HWC
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    # grayscale -> 3ch
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # to uint8
    if image.dtype != np.uint8:
        vmin, vmax = float(image.min()), float(image.max())
        if 0.0 <= vmin and vmax <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    return image

def test_image_path(path):
    """substract images from the main dataset folder

    Args:
        path (str): The root path of the datasets.

    Returns:
        list: A list of paths to the testing subdirectories.
    """
    print("Generating test image paths from:", path)
    return [
        ds + "/testing/" for ds in sorted(glob(path + "*"))
        if os.path.isdir(ds)
    ]
TEST_IMAGE_PATH = [ds + "/testing/" for ds in sorted(glob(TEST_PATH + "*"))]

# -----------------------------
# SAM2-only evaluation core
# -----------------------------
@torch.no_grad()
def get_test_record_from_loader(
    sam2_model,
    test_loader: DataLoader,
) -> list[dict]:
    """
    Compute per-case metrics (point / box / box+point) using SAM2ImagePredictor.

    Reuses:
      - MaskSplit for label separation
      - gen_points / gen_bboxes for prompt generation
      - dice_similarity / sd_hausdorff_distance / sd_mean_surface_distance for metrics
    """
    predictor = SAM2ImagePredictor(sam2_model)

    final_record: list[dict] = []

    for image, mask, img_fullpath, mask_fullpath in tqdm(test_loader):
        # From your NiiDataset: image is HWC uint8, mask is (1,H,W) labeled ints
        image = image.squeeze(0).detach().cpu().numpy()
        mask = mask.squeeze(0).detach().cpu().numpy()  # (1,H,W) after squeeze(0) becomes (1,H,W) OR (H,W) depending; handle below

        if mask.ndim == 2:
            mask = mask[None, :, :]  # ensure (1,H,W)

        image = _ensure_hwc_uint8_rgb(image)
        predictor.set_image(image)

        H, W = mask.shape[-2], mask.shape[-1]
        total_pixels = H * W

        # Per-label metrics
        labels = []
        pixel_count = []
        area_percentage = []

        p_dice, b_dice, bp_dice = [], [], []
        p_hd, b_hd, bp_hd = [], [], []
        p_msd, b_msd, bp_msd = [], [], []

        # Split labeled mask into binary masks
        ms = MaskSplit(mask)

        for i in range(len(ms)):
            each_mask, label = ms[i]  # each_mask: (H,W) boolean

            # Prompts (reuse your utilities)
            point = gen_points(each_mask)
            point_label = np.array([1], dtype=np.int32)
            bbox = gen_bboxes(each_mask, jitter=0)  # [x0,y0,x1,y1]

            # SAM2 predicts masks as (N,H,W)
            pre_mask_p, _, _ = predictor.predict(
                point_coords=point,
                point_labels=point_label,
                box=None,
                multimask_output=False,
            )

            pre_mask_b, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(bbox, dtype=np.float32)[None, :],
                multimask_output=False,
            )

            pre_mask_bp, _, _ = predictor.predict(
                point_coords=point,
                point_labels=point_label,
                box=np.array(bbox, dtype=np.float32)[None, :],
                multimask_output=False,
            )

            # Save label + size stats
            labels.append(int(label))
            pc = int(np.sum(each_mask))
            pixel_count.append(pc)
            area_percentage.append(pc / float(total_pixels))

            # Metrics (reuse your losses.py helpers)
            gt = each_mask.astype(np.uint8)

            pm = pre_mask_p[0].astype(np.uint8)
            bm = pre_mask_b[0].astype(np.uint8)
            bpm = pre_mask_bp[0].astype(np.uint8)

            p_dice.append(dice_similarity(pm, gt))
            b_dice.append(dice_similarity(bm, gt))
            bp_dice.append(dice_similarity(bpm, gt))

            p_hd.append(sd_hausdorff_distance(pm, gt))
            b_hd.append(sd_hausdorff_distance(bm, gt))
            bp_hd.append(sd_hausdorff_distance(bpm, gt))

            p_msd.append(sd_mean_surface_distance(pm, gt))
            b_msd.append(sd_mean_surface_distance(bm, gt))
            bp_msd.append(sd_mean_surface_distance(bpm, gt))

        single_data_result = {
            "img_name": img_fullpath,
            "mask_name": mask_fullpath,
            "labels": labels,
            "p_dice": p_dice,
            "b_dice": b_dice,
            "bp_dice": bp_dice,
            "p_hd": p_hd,
            "b_hd": b_hd,
            "bp_hd": bp_hd,
            "p_msd": p_msd,
            "b_msd": b_msd,
            "bp_msd": bp_msd,
            "pixel_count": pixel_count,
            "area_percentage": area_percentage,
        }
        final_record.append(single_data_result)

    return final_record


def save_test_record(
    file_paths: list[str],
    sam2_model,
    save_path: str,
    num_workers: int = 8,
) -> dict:
    """
    Evaluate all datasets in file_paths, and pickle a dict: {ds_name: record_list}.
    This mirrors the intent of utils/visual.save_test_record, but for SAM2 only.
    """
    final_record: dict[str, list[dict]] = {}
    save_path = str(save_path)

    for file_path in file_paths:
        print(f"Processing dataset: {file_path}")
        ds_name = file_path.split("/")[-3]

        # Reuse your NiiDataset exactly
        test_dataset = NiiDataset([file_path], multi_mask=True, with_name=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
        )

        ds_record = get_test_record_from_loader(sam2_model=sam2_model, test_loader=test_loader)
        final_record[ds_name] = ds_record

    # Save pickle
    Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(final_record, f)

    return final_record


# -----------------------------
# CLI
# -----------------------------
def _collect_ckpts(ckpt_path: str) -> list[str]:
    p = Path(ckpt_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        ckpts = sorted([str(x) for x in p.glob("*.pt")] + [str(x) for x in p.glob("*.pth")])
        if not ckpts:
            raise FileNotFoundError(f"No .pt/.pth checkpoints found in: {ckpt_path}")
        return ckpts
    raise FileNotFoundError(f"ckpt-path not found: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-config", type=str, required=True,
                        help="SAM2 config yaml, e.g. configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Checkpoint file (.pt/.pth) or a directory containing checkpoints.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-path", type=str, required=True,
                        help="Output pickle path OR output folder if testing multiple ckpts.")
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--test-file-paths", type=str, nargs="+", required=True,
                        help="List of dataset folder paths (same format you used before).")

    args = parser.parse_args()

    device = args.device
    ckpts = _collect_ckpts(args.ckpt_path)

    save_root = Path(args.save_path)
    if len(ckpts) > 1:
        save_root.mkdir(parents=True, exist_ok=True)

    for ckpt in ckpts:
        print("\n======================================")
        print(f"SAM2 config : {args.sam2_config}")
        print(f"Checkpoint  : {ckpt}")
        print(f"Device      : {device}")

        t0 = time.time()

        sam2_model = build_sam2(args.sam2_config, ckpt, device=device)
        sam2_model.eval()

        if len(ckpts) == 1:
            out_path = str(save_root)
        else:
            out_path = str(save_root / (Path(ckpt).stem + ".pkl"))

        save_test_record(
            file_paths=test_image_path(args.test_file_paths),
            sam2_model=sam2_model,
            save_path=out_path,
            num_workers=args.num_workers,
        )

        print(f"Saved: {out_path}")
        print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
