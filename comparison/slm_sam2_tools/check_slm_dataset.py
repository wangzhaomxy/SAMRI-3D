#!/usr/bin/env python3
"""
check_slm_dataset.py

Sanity-check an exported SLM-SAM2 training folder by loading it through SLM-SAM2's OWN
data pipeline (PNGRawDataset + RandomUniformSampler + VOSDataset + the official training
transforms). If this passes, the training job will read the data the same way.

Run from anywhere; the SLM-SAM2 code root is put on sys.path automatically:
    python comparison/slm_sam2_tools/check_slm_dataset.py \
        --data-dir /scratch/user/s4670484/Datasets/SLM_SAM2_train_smoke --n 20
"""
import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SLM_ROOT = Path(__file__).resolve().parents[1] / "SLM-SAM2-main"
sys.path.insert(0, str(SLM_ROOT))

from PIL import Image  # noqa: E402
from training.dataset.vos_raw_dataset import PNGRawDataset  # noqa: E402
from training.dataset.vos_sampler import RandomUniformSampler  # noqa: E402
from training.dataset.vos_dataset import VOSDataset  # noqa: E402
from training.dataset import transforms as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=20, help="number of random clips to load")
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--max-num-objects", type=int, default=3)
    args = ap.parse_args()

    img_folder = os.path.join(args.data_dir, "JPEGImages")
    gt_folder = os.path.join(args.data_dir, "Annotations")
    file_list = os.path.join(args.data_dir, "train.txt")

    # 1) file-level checks
    vids = [l.strip() for l in open(file_list) if l.strip()]
    print(f"train.txt: {len(vids)} videos")
    bad = 0
    for v in vids[: min(len(vids), 200)]:
        nj = len([f for f in os.listdir(os.path.join(img_folder, v)) if f.endswith(".jpg")])
        npng = len([f for f in os.listdir(os.path.join(gt_folder, v)) if f.endswith(".png")])
        if nj != npng or nj == 0:
            bad += 1
            print(f"  [BAD] {v}: {nj} jpg vs {npng} png")
    print(f"frame/mask count mismatch in first {min(len(vids), 200)} videos: {bad}")

    # 2) palette indices survive the loader's .convert('P')
    v = vids[0]
    pngs = sorted(os.listdir(os.path.join(gt_folder, v)))
    pngs = [p for p in pngs if p.endswith(".png")]
    ids = Counter()
    for p in pngs:
        m = Image.open(os.path.join(gt_folder, v, p))
        assert m.mode == "P", f"mask mode is {m.mode}, expected P"
        ids.update(np.unique(np.array(m.convert("P"))).tolist())
    print(f"label ids in {v}: {sorted(ids)} (0 = background)")

    # 3) load clips through SLM-SAM2's own pipeline with the official transforms
    train_tf = [T.ComposeAPI(transforms=[
        T.RandomHorizontalFlip(consistent_transform=True),
        T.RandomAffine(degrees=25, shear=20, image_interpolation="bilinear",
                       consistent_transform=True),
        T.RandomResizeAPI(sizes=1024, square=True, consistent_transform=True),
        T.ColorJitter(consistent_transform=True, brightness=0.1, contrast=0.03,
                      saturation=0.03, hue=None),
        T.RandomGrayscale(p=0.05, consistent_transform=True),
        T.ColorJitter(consistent_transform=False, brightness=0.1, contrast=0.05,
                      saturation=0.05, hue=None),
        T.ToTensorAPI(),
        T.NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])]
    ds = VOSDataset(
        transforms=train_tf, training=True,
        video_dataset=PNGRawDataset(img_folder=img_folder, gt_folder=gt_folder,
                                    file_list_txt=file_list),
        sampler=RandomUniformSampler(num_frames=args.num_frames,
                                     max_num_objects=args.max_num_objects),
        multiplier=1,
    )
    random.seed(0)
    n_obj, fg_frac = [], []
    for k in range(args.n):
        dp = ds[random.randrange(len(ds))]
        assert len(dp.frames) == args.num_frames
        f0 = dp.frames[0]
        assert tuple(f0.data.shape) == (3, 1024, 1024), f0.data.shape
        n_obj.append(len(f0.objects))
        assert len(f0.objects) >= 1, "first frame must contain a target"
        assert any(o.segment.sum() > 0 for o in f0.objects), "first-frame target empty"
        fg_frac.append(float(np.mean([o.segment.float().mean().item() for o in f0.objects])))
    print(f"loaded {args.n} clips OK: frames={args.num_frames}, image 3x1024x1024, "
          f"objects/clip {min(n_obj)}-{max(n_obj)}, mean first-frame fg fraction "
          f"{np.mean(fg_frac):.4f}")
    print("CHECK PASSED")


if __name__ == "__main__":
    main()
