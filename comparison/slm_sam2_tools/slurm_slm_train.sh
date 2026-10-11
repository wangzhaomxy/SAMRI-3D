#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --job-name=SLM_train
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi300x:1
#SBATCH --account=a_ai_collab
#SBATCH --qos=gpu
#SBATCH -o /scratch/user/s4670484/SAMRI2_log/slurm-slm-train-%j.output
#SBATCH -e /scratch/user/s4670484/SAMRI2_log/slurm-slm-train-%j.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhao.wang1@uq.edu.au
#
# Train SLM-SAM2 with its OFFICIAL recipe (SAM2.1-Tiny init, full fine-tuning,
# 8-slice clips at 1024 px, 100 epochs, recent_n=1) on the exported SAMRI-3D training split.
#
# Submit FROM THE SAMRI-3D REPO ROOT:
#   1) smoke (1 epoch on the smoke export, ~10-30 min):
#        sbatch --time=0-01:00:00 comparison/slm_sam2_tools/slurm_slm_train.sh
#   2) full (official: 1 GPU, batch 1):
#        sbatch --export=ALL,MODE=full comparison/slm_sam2_tools/slurm_slm_train.sh
#      multi-GPU alternative (changes the effective batch -- decide after the smoke timing):
#        sbatch --gres=gpu:mi300x:4 --export=ALL,MODE=full comparison/slm_sam2_tools/slurm_slm_train.sh
#
# Auto-resume: the trainer restarts from <EXP>/checkpoints/checkpoint.pt (overwritten
# every epoch), so if the 4-day limit is hit just submit the same command again.

set -euo pipefail
MODE=${MODE:-smoke}

mkdir -p /scratch/user/s4670484/SAMRI2_log
module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate samri-mi300

cd "$SLURM_SUBMIT_DIR"
[ -f test_vis3d_multi.py ] && [ -d comparison/SLM-SAM2-main ] || {
  echo "[ERROR] submit from the SAMRI-3D repo root"; exit 1; }

export MASTER_ADDR=localhost
export MASTER_PORT=$((27000 + RANDOM % 1000))

CKPT=/scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_tiny.pt
if [ "$MODE" = "smoke" ]; then
  DATA=/scratch/user/s4670484/Datasets/SLM_SAM2_train_smoke
  EXP=/scratch/user/s4670484/Model_save/slm_sam2_smoke_${SLURM_JOB_ID}
  EXTRA="--num-epochs 1"
else
  DATA=/scratch/user/s4670484/Datasets/SLM_SAM2_train
  EXP=/scratch/user/s4670484/Model_save/slm_sam2_samri3d
  EXTRA=""
fi

# ---- preflight: fail in seconds, not hours ----
python - <<EOF
import importlib, os, sys, torch
missing = [m for m in ("hydra", "iopath", "fvcore", "submitit", "tensordict", "pandas",
                       "tensorboard") if importlib.util.find_spec(m) is None]
if missing: sys.exit(f"[PREFLIGHT] missing python packages: {missing} (pip install them)")
n = torch.cuda.device_count()
print(f"[PREFLIGHT] torch {torch.__version__} | GPUs visible: {n} | "
      f"{torch.cuda.get_device_name(0) if n else 'none'}")
if n == 0: sys.exit("[PREFLIGHT] no GPU visible")
for p in ("$CKPT", "$DATA/train.txt", "$DATA/JPEGImages", "$DATA/Annotations"):
    if not os.path.exists(p): sys.exit(f"[PREFLIGHT] not found: {p}")
nv = sum(1 for l in open("$DATA/train.txt") if l.strip())
print(f"[PREFLIGHT] training videos: {nv}")
EOF

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "MODE=$MODE  NUM_GPUS=$NUM_GPUS  DATA=$DATA  EXP=$EXP  started $(date)"

python comparison/slm_sam2_tools/train_slm_sam2.py \
  --data-dir "$DATA" --pretrained-ckpt "$CKPT" --exp-dir "$EXP" \
  --num-gpus "$NUM_GPUS" $EXTRA

echo "finished $(date)"
ls -la "$EXP/checkpoints" || true
