#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_slm_dataset.py

Export the SAMRI-3D training volumes to the SAM2/SLM-SAM2 video-folder format so that
SLM-SAM2 can be trained with its official recipe on exactly the same data as SAMRI-3D.

Same data as SAMRI-3D
---------------------
* Volume + label selection is identical to generate_3d_embeddings.py
  (`collect_volumes_from_training_list`): rows of training_list_3d.pkl with
  dice >= --dice-threshold (default 0.2), grouped per volume. Labels that fail the
  threshold are set to background, exactly as SAMRI-3D never trains on them.
* Slicing axis = shortest axis (`choose_primary_axis`), the same "slice-as-frame"
  convention used for training and for evaluation in test_vis3d_multi.py.
* Each slice is min-max normalised to uint8 (`normalize_to_uint8`) and written as an
  RGB JPEG (quality 95) -- the same preprocessing SAMRI-3D uses.

Output layout (what SLM-SAM2's PNGRawDataset expects)
-----------------------------------------------------
<out-dir>/
  JPEGImages/<video_id>/00000.jpg ...      one JPEG per slice (all slices, incl. empty)
  Annotations/<video_id>/00000.png ...     palettised PNG, pixel value = label id
  train.txt                                one video_id per line
  export_manifest.csv                      video_id, dataset, paths, axis, shape, labels
  export_summary.csv                       per-dataset counts
  _export_done/<video_id>                  resume markers (kept out of Annotations/)

video_id = <dataset>__<image file stem>

Usage
-----
# Smoke test (2 volumes per dataset) -- check the output before the full export
python comparison/slm_sam2_tools/export_slm_dataset.py \
    --training-list /scratch/user/s4670484/training_lists/training_list_3d.pkl \
    --out-dir /scratch/user/s4670484/Datasets/SLM_SAM2_train_smoke \
    --max-volumes-per-dataset 2 --num-workers 8

# Full export (resumable: finished volumes are skipped)
python comparison/slm_sam2_tools/export_slm_dataset.py \
    --training-list /scratch/user/s4670484/training_lists/training_list_3d.pkl \
    --out-dir /scratch/user/s4670484/Datasets/SLM_SAM2_train \
    --num-workers 32
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]  # SAMRI-3D repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_vis3d_multi import read_3d_image, choose_primary_axis, normalize_to_uint8  # noqa: E402

# Resume markers live OUTSIDE Annotations/: SLM-SAM2's mask loader int()-parses every
# filename in a video's mask folder, so no extra files may be placed there.
DONE_DIR = "_export_done"

# Simple, distinct palette; index 0 = background. Only the indices matter for training.
_PALETTE = [0, 0, 0]
_rng = np.random.RandomState(0)
for _i in range(1, 256):
    _PALETTE.extend(_rng.randint(40, 256, size=3).tolist())


def collect_volumes(training_list: str, dice_threshold: float) -> List[Dict]:
    """Same grouping/filter as generate_3d_embeddings.collect_volumes_from_training_list."""
    with open(training_list, "rb") as f:
        rows = pickle.load(f)
    vols: Dict = defaultdict(lambda: {"labels": [], "dices": []})
    for r in rows:
        if r["dice"] >= dice_threshold:
            key = (r["img_path"], r["mask_path"], r["dataset"])
            v = vols[key]
            v["labels"].append(int(r["label"]))
            v["dices"].append(float(r["dice"]))
            v["img_path"], v["mask_path"], v["dataset"] = key
    out = list(vols.values())
    out.sort(key=lambda v: (v["dataset"], v["img_path"]))
    return out


def image_stem(path: str) -> str:
    name = os.path.basename(path.rstrip("/"))
    for ext in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def assign_video_ids(vols: List[Dict]) -> None:
    seen: Dict[str, int] = {}
    for v in vols:
        base = f"{v['dataset']}__{image_stem(v['img_path'])}"
        base = base.replace("/", "_").replace(" ", "_")
        n = seen.get(base, 0)
        seen[base] = n + 1
        v["video_id"] = base if n == 0 else f"{base}__{n}"


def export_one(v: Dict, out_dir: str, min_slices: int, jpeg_quality: int) -> Dict:
    vid = v["video_id"]
    img_dir = os.path.join(out_dir, "JPEGImages", vid)
    ann_dir = os.path.join(out_dir, "Annotations", vid)
    res = {"video_id": vid, "dataset": v["dataset"], "img_path": v["img_path"],
           "mask_path": v["mask_path"], "status": "", "axis": -1, "shape": "",
           "labels": " ".join(str(x) for x in sorted(set(v["labels"])))}
    done_flag = os.path.join(out_dir, DONE_DIR, vid)
    if os.path.exists(done_flag):
        res["status"] = "exists"
        return res
    try:
        img, _ = read_3d_image(v["img_path"])
        gt, _ = read_3d_image(v["mask_path"])
        if img.shape != gt.shape:
            res["status"] = f"skip_shape_mismatch {img.shape} vs {gt.shape}"
            return res
        axis = choose_primary_axis(img.shape)
        vol = np.moveaxis(img, axis, 0)
        seg = np.moveaxis(gt, axis, 0)
        N, H, W = vol.shape
        res["axis"], res["shape"] = axis, f"{N}x{H}x{W}"
        if N < min_slices:
            res["status"] = f"skip_too_few_slices({N}<{min_slices})"
            return res

        keep = sorted(set(int(x) for x in v["labels"]))
        if max(keep) > 255:
            res["status"] = "skip_label_id_gt_255"
            return res
        seg_out = np.zeros(seg.shape, dtype=np.uint8)
        for lb in keep:
            seg_out[seg == lb] = lb
        if not seg_out.any():
            res["status"] = "skip_no_kept_foreground"
            return res

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        for z in range(N):
            s2d = normalize_to_uint8(vol[z])
            Image.fromarray(np.stack([s2d, s2d, s2d], axis=-1)).save(
                os.path.join(img_dir, f"{z:05d}.jpg"), quality=jpeg_quality)
            m = Image.fromarray(seg_out[z])          # mode "L"
            m.putpalette(_PALETTE)                   # -> mode "P", indices unchanged
            m.save(os.path.join(ann_dir, f"{z:05d}.png"))
        os.makedirs(os.path.join(out_dir, DONE_DIR), exist_ok=True)
        Path(done_flag).touch()
        res["status"] = "ok"
        return res
    except Exception as e:  # keep going; reported in the manifest
        res["status"] = f"error {type(e).__name__}: {e}"
        return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training-list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dice-threshold", type=float, default=0.2,
                    help="Same threshold as SAMRI-3D embedding generation (default 0.2).")
    ap.add_argument("--min-slices", type=int, default=8,
                    help="Skip volumes shorter than SLM-SAM2's training clip (num_frames=8).")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-volumes-per-dataset", type=int, default=None,
                    help="For smoke tests only.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Optional subset of dataset names.")
    args = ap.parse_args()

    vols = collect_volumes(args.training_list, args.dice_threshold)
    if args.datasets:
        vols = [v for v in vols if v["dataset"] in set(args.datasets)]
    if args.max_volumes_per_dataset:
        per = defaultdict(int)
        sel = []
        for v in vols:
            if per[v["dataset"]] < args.max_volumes_per_dataset:
                sel.append(v)
                per[v["dataset"]] += 1
        vols = sel
    assign_video_ids(vols)
    n_ds = len(set(v["dataset"] for v in vols))
    print(f"Volumes to export: {len(vols)} from {n_ds} datasets "
          f"(dice >= {args.dice_threshold}) -> {args.out_dir}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = [ex.submit(export_one, v, args.out_dir, args.min_slices, args.jpeg_quality)
                for v in vols]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 200 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done", flush=True)

    results.sort(key=lambda r: r["video_id"])
    with open(os.path.join(args.out_dir, "export_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    ok = [r for r in results if r["status"] in ("ok", "exists")]
    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        for r in ok:
            f.write(r["video_id"] + "\n")

    summ = defaultdict(lambda: {"ok": 0, "skipped": 0, "errors": 0, "slices": 0})
    for r in results:
        s = summ[r["dataset"]]
        if r["status"] in ("ok", "exists"):
            s["ok"] += 1
            s["slices"] += int(r["shape"].split("x")[0]) if r["shape"] else 0
        elif r["status"].startswith("error"):
            s["errors"] += 1
        else:
            s["skipped"] += 1
    with open(os.path.join(args.out_dir, "export_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "volumes_ok", "skipped", "errors", "slices"])
        for ds in sorted(summ):
            s = summ[ds]
            w.writerow([ds, s["ok"], s["skipped"], s["errors"], s["slices"]])

    n_err = sum(1 for r in results if r["status"].startswith("error"))
    n_skip = len(results) - len(ok) - n_err
    print(f"\nExported {len(ok)} volumes ({sum(s['slices'] for s in summ.values())} slices); "
          f"skipped {n_skip}; errors {n_err}.")
    print(f"train.txt, export_manifest.csv, export_summary.csv written to {args.out_dir}")
    if n_err:
        print("Errors present -- see the 'status' column of export_manifest.csv.")


if __name__ == "__main__":
    main()
