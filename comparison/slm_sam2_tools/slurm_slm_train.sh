#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=SLM_train
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/SAMRI2_log/slurm-slm-train-%j.output
#SBATCH -e /scratch/user/s4670484/SAMRI2_log/slurm-slm-train-%j.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhao.wang1@uq.edu.au
#
# Train SLM-SAM2 with its OFFICIAL recipe (SAM2.1-Tiny init, full fine-tuning, 8-slice
# clips at 1024 px, 100 epochs, recent_n=1, 1 GPU x batch 1) on the exported SAMRI-3D
# training split.
#
# Card: MI355X x1 under qos=gpu (3-day cap, never cut so far; ~2.8x an MI300X). The env
# is picked automatically from the card (bunya_common.sh). MI300X alternative:
#   sbatch --gres=gpu:mi300x:1 --time=0-12:00:00 ...   (practical 12 h cut -> resubmit; resumes)
#
#   1) smoke: 1 epoch on the smoke export (gives seconds/iteration):
#      sbatch --time=0-01:00:00 /scratch/user/s4670484/SAMRI2/comparison/slm_sam2_tools/slurm_slm_train.sh
#   2) full:
#      sbatch --export=ALL,MODE=full /scratch/user/s4670484/SAMRI2/comparison/slm_sam2_tools/slurm_slm_train.sh
#
# Resume: the trainer restarts from <EXP>/checkpoints/checkpoint.pt (rewritten every
# epoch), so a cut or timed-out job is continued by submitting the same command.
# Speed: judge from epoch 1+, never epoch 0 (MIOpen tunes kernels during the first epoch).

set -uo pipefail
MODE=${MODE:-smoke}
source /scratch/user/s4670484/SAMRI2/comparison/slm_sam2_tools/bunya_common.sh

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

gpu_preflight hydra iopath fvcore submitit tensordict pandas tensorboard PIL || exit 1
require_paths "$CKPT" "$DATA/train.txt" "$DATA/JPEGImages" "$DATA/Annotations" || exit 1
echo "[PREFLIGHT] training videos: $(grep -c . "$DATA/train.txt")"

NG=$(( NGPU > 0 ? NGPU : 1 ))
WORKERS=$(( CPUS / NG - 1 )); [ "$WORKERS" -gt 12 ] && WORKERS=12; [ "$WORKERS" -lt 1 ] && WORKERS=1
echo "MODE=$MODE  NGPU=$NG  WORKERS/GPU=$WORKERS  DATA=$DATA  EXP=$EXP"
[ -f "$EXP/checkpoints/checkpoint.pt" ] && echo "[RESUME] found $EXP/checkpoints/checkpoint.pt"

python comparison/slm_sam2_tools/train_slm_sam2.py \
  --data-dir "$DATA" --pretrained-ckpt "$CKPT" --exp-dir "$EXP" \
  --num-gpus "$NG" --num-workers "$WORKERS" $EXTRA
RC=$?
echo "finished $(date) rc=$RC"
ls -la "$EXP/checkpoints" 2>/dev/null
grep "Train Epoch" "$EXP/logs/log.txt" 2>/dev/null | tail -3
exit $RC
