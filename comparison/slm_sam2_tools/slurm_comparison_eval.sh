#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=CMP_eval
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi300x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/SAMRI2_log/slurm-cmp-eval-%j.output
#SBATCH -e /scratch/user/s4670484/SAMRI2_log/slurm-cmp-eval-%j.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhao.wang1@uq.edu.au
#
# Evaluate on the same 34-dataset test set and protocol as SAMRI-3D (comparison_test.py):
#   TARGET=medsam2   MedSAM2 rerun with the box-coordinate fix
#   TARGET=slm_box   SLM-SAM2, GT box on the cut slice   (headline protocol)
#   TARGET=slm_mask  SLM-SAM2, GT mask on the cut slice  (its native prompt, secondary)
#
# Card: MI300X x1 (samri-mi300, the env used for the earlier SAM2-based evaluations).
# --time=12h because MI300X jobs have been cut at ~12 h; the run saves after every dataset
# and --resume skips finished datasets, so a cut job is continued by resubmitting it.
# If the earlier MedSAM2 run used another env, add ENV=<env> to --export.
#
#   S=/scratch/user/s4670484/SAMRI2/comparison/slm_sam2_tools/slurm_comparison_eval.sh
#   sbatch --time=0-01:00:00 --export=ALL,TARGET=medsam2 $S        # smoke: 1 case/dataset
#   sbatch --export=ALL,TARGET=medsam2,MODE=full $S               # full

set -uo pipefail
MODE=${MODE:-smoke}
TARGET=${TARGET:-}
[ -n "$TARGET" ] || { echo "[ERROR] set TARGET=medsam2|slm_box|slm_mask"; exit 1; }
source /scratch/user/s4670484/SAMRI2/comparison/slm_sam2_tools/bunya_common.sh

DATA_SEEN=/scratch/user/s4670484/Datasets/SAMRI_datasets_3D/
DATA_UNSEEN=/scratch/user/s4670484/Datasets/Zeroshot3D/
WEIGHTS=/scratch/user/s4670484/comparison_ckpt/
SLM_CKPT=${SLM_CKPT:-/scratch/user/s4670484/Model_save/slm_sam2_samri3d/checkpoints/checkpoint.pt}
OUT_DIR=/scratch/user/s4670484/Eval_results/comparison

case "$TARGET" in
  medsam2)  ARGS=(--model medsam2);                                          RUN=medsam2_boxfix; NEED=("$WEIGHTS/MedSAM2") ;;
  slm_box)  ARGS=(--model slm_sam2 --slm-sam2-ckpt "$SLM_CKPT" --slm-prompt box);  RUN=slm_sam2_box;  NEED=("$SLM_CKPT") ;;
  slm_mask) ARGS=(--model slm_sam2 --slm-sam2-ckpt "$SLM_CKPT" --slm-prompt mask); RUN=slm_sam2_mask; NEED=("$SLM_CKPT") ;;
  *) echo "[ERROR] unknown TARGET=$TARGET"; exit 1 ;;
esac
EXTRA=()
if [ "$MODE" = "smoke" ]; then RUN=${RUN}_smoke; EXTRA=(--max-cases 1); fi
SAVE="$OUT_DIR/$RUN/$RUN.pkl"

gpu_preflight hydra PIL SimpleITK || exit 1
require_paths "$DATA_SEEN" "$DATA_UNSEEN" "${NEED[@]}" || exit 1
echo "TARGET=$TARGET MODE=$MODE SAVE=$SAVE"
[ -f "$SAVE.partial.pkl" ] && echo "[RESUME] partial results found: $SAVE.partial.pkl"

python comparison_test.py "${ARGS[@]}" \
  --dataset-path "$DATA_SEEN" "$DATA_UNSEEN" \
  --weights-dir "$WEIGHTS" \
  --save-path "$SAVE" \
  --device cuda --resume "${EXTRA[@]}"
RC=$?
echo "finished $(date) rc=$RC"
N_ERR=$(grep -c '\[ERROR\]' "$OUT_DIR/$RUN/log.txt" 2>/dev/null) || true
echo "[ERROR] lines in log: ${N_ERR:-0}"
exit $RC
