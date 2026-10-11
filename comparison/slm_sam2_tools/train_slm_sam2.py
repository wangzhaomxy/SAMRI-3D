#!/usr/bin/env python3
"""
train_slm_sam2.py

Thin launcher for SLM-SAM2's OFFICIAL training (comparison/SLM-SAM2-main/training/train.py)
that injects our paths as Hydra overrides, so the official config
`configs/sam2.1_training/slm_sam2_hiera_t_finetune.yaml` is used unmodified
(SAM2.1-Tiny init, full fine-tuning, 8-slice clips, 1024 px, lr 5e-6 / vision 3e-6,
100 epochs, recent_n = 1).

Only these fields are overridden:
  dataset.img_folder / gt_folder / file_list_txt      -> the exported SAMRI-3D training data
  ...model_weight_initializer.state_dict.checkpoint_path -> sam2.1_hiera_tiny.pt
  launcher.experiment_log_dir                         -> where logs + checkpoints go
  launcher.gpus_per_node (--num-gpus)
  optional: scratch.num_epochs, scratch.num_train_workers, and any --override key=value

Output: <exp-dir>/checkpoints/checkpoint.pt  ({"model": state_dict, ...}, last epoch),
        <exp-dir>/logs/log.txt, <exp-dir>/tensorboard/, <exp-dir>/config_resolved.yaml

Usage (from anywhere):
python comparison/slm_sam2_tools/train_slm_sam2.py \
    --data-dir /scratch/user/s4670484/Datasets/SLM_SAM2_train \
    --pretrained-ckpt /scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_tiny.pt \
    --exp-dir /scratch/user/s4670484/Model_save/slm_sam2_samri3d \
    --num-gpus 1
"""
import argparse
import os
import sys
from pathlib import Path

SLM_ROOT = Path(__file__).resolve().parents[1] / "SLM-SAM2-main"
SAMRI_ROOT = Path(__file__).resolve().parents[2]

# SLM-SAM2 ships its own `sam2` and `training` packages that clash with SAMRI-3D's.
# Put SLM-SAM2 first and drop the SAMRI-3D root so every import (incl. in spawned
# DDP workers, which inherit sys.path) resolves to SLM-SAM2's code.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != SAMRI_ROOT]
sys.path.insert(0, str(SLM_ROOT))
os.environ["PYTHONPATH"] = str(SLM_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")


_ORIG_SINGLE_PROC_RUN = None


def _per_rank_run(local_rank, main_port, cfg, world_size):
    """Wrap SLM-SAM2's single_proc_run: give each DDP rank its own MIOpen cache dir
    (MIOpen's SQLite caches must not be shared between processes)."""
    base = os.environ.get("MIOPEN_USER_DB_PATH")
    if base:
        d = f"{base.rstrip('/')}-r{local_rank}"
        os.makedirs(d, exist_ok=True)
        os.environ["MIOPEN_USER_DB_PATH"] = d
        os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = d
    fn = _ORIG_SINGLE_PROC_RUN
    if fn is None:  # spawned worker: training.train is unpatched here
        import training.train as tt
        fn = tt.single_proc_run
    return fn(local_rank, main_port, cfg, world_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/sam2.1_training/slm_sam2_hiera_t_finetune.yaml")
    ap.add_argument("--data-dir", required=True,
                    help="Output of export_slm_dataset.py (JPEGImages/, Annotations/, train.txt)")
    ap.add_argument("--file-list", default=None, help="Default: <data-dir>/train.txt")
    ap.add_argument("--pretrained-ckpt", required=True, help="sam2.1_hiera_tiny.pt")
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--num-epochs", type=int, default=None, help="Default: official (100)")
    ap.add_argument("--num-workers", type=int, default=None, help="Default: official (8)")
    ap.add_argument("--override", action="append", default=[],
                    help="Extra Hydra override key=value (repeatable)")
    args = ap.parse_args()

    import training.train as tt  # SLM-SAM2's train.py
    assert Path(tt.__file__).resolve().is_relative_to(SLM_ROOT), tt.__file__
    import sam2
    assert Path(sam2.__file__).resolve().is_relative_to(SLM_ROOT), sam2.__file__

    file_list = args.file_list or os.path.join(args.data_dir, "train.txt")
    for p in (os.path.join(args.data_dir, "JPEGImages"),
              os.path.join(args.data_dir, "Annotations"), file_list, args.pretrained_ckpt):
        if not os.path.exists(p):
            sys.exit(f"[ERROR] not found: {p}")

    overrides = [
        f"dataset.img_folder={os.path.join(args.data_dir, 'JPEGImages')}",
        f"dataset.gt_folder={os.path.join(args.data_dir, 'Annotations')}",
        f"dataset.file_list_txt={file_list}",
        "trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path="
        f"{args.pretrained_ckpt}",
        f"launcher.experiment_log_dir={args.exp_dir}",
    ]
    if args.num_epochs is not None:
        overrides.append(f"scratch.num_epochs={args.num_epochs}")
    if args.num_workers is not None:
        overrides.append(f"scratch.num_train_workers={args.num_workers}")
    overrides += args.override
    print("Hydra overrides:\n  " + "\n  ".join(overrides), flush=True)

    _compose = tt.compose
    tt.compose = lambda config_name: _compose(config_name=config_name, overrides=overrides)
    global _ORIG_SINGLE_PROC_RUN
    _ORIG_SINGLE_PROC_RUN = tt.single_proc_run
    tt.single_proc_run = _per_rank_run

    # Importing SLM-SAM2's `sam2` package already initialises Hydra with module "sam2"
    # (sam2/__init__.py), which is exactly what train.py's __main__ would do.
    from hydra.core.global_hydra import GlobalHydra
    if not GlobalHydra.instance().is_initialized():
        tt.initialize_config_module("sam2", version_base="1.2")
    tt.register_omegaconf_resolvers()
    tt.main(argparse.Namespace(config=args.config, use_cluster=False, partition=None,
                               account=None, qos=None, num_gpus=args.num_gpus, num_nodes=1))


if __name__ == "__main__":
    main()
