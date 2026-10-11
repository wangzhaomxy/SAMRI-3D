#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=CMP_eval
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi300x:1
#SBATCH --account=a_ai_collab
#SBATCH --qos=gpu
#SBATCH -o /scratch/user/s4670484/SAMRI2_log/slurm-cmp-eval-%j.output
#SBATCH -e /scratch/user/s4670484/SAMRI2_log/slurm-cmp-eval-%j.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhao.wang1@uq.edu.au
#
# Evaluate on the same 34-dataset test set and protocol as SAMRI-3D (comparison_test.py):
#   TARGET=slm_box   SLM-SAM2, GT box on the cut slice     (headline protocol)
#   TARGET=slm_mask  SLM-SAM2, GT mask on the cut slice    (its native prompt, secondary)
#   TARGET=medsam2   MedSAM2 rerun with the box-coordinate fix
#
# Submit FROM THE SAMRI-3D REPO ROOT, one job per target. Smoke first (1 case/dataset):
#   sbatch --time=0-01:00:00 --export=ALL,TARGET=medsam2 comparison/slm_sam2_tools/slurm_comparison_eval.sh
#   sbatch --export=ALL,TARGET=medsam2,MODE=full         comparison/slm_sam2_tools/slurm_comparison_eval.sh
#
# Each run writes <OUT_DIR>/<run>/<run>.pkl plus its own log.txt (separate folders so
# parallel runs do not overwrite each other's log).

set -euo pipefail
MODE=${MODE:-smoke}
TARGET=${TARGET:?set TARGET=slm_box|slm_mask|medsam2}
# Conda env used for the earlier comparison runs (MedSAM2 needs its deps; SLM-SAM2 only
# needs the SAM2 stack, so samri-mi300 is enough for it).
COMP_ENV=${COMP_ENV:-samri-mi300}

mkdir -p /scratch/user/s4670484/SAMRI2_log
module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate "$COMP_ENV"

cd "$SLURM_SUBMIT_DIR"
[ -f comparison_test.py ] || { echo "[ERROR] submit from the SAMRI-3D repo root"; exit 1; }

DATA_SEEN=/scratch/user/s4670484/Datasets/SAMRI_datasets_3D/
DATA_UNSEEN=/scratch/user/s4670484/Datasets/Zeroshot3D/
WEIGHTS=/scratch/user/s4670484/comparison_ckpt/
SLM_CKPT=${SLM_CKPT:-/scratch/user/s4670484/Model_save/slm_sam2_samri3d/checkpoints/checkpoint.pt}
OUT_DIR=/scratch/user/s4670484/Eval_results/comparison

case "$TARGET" in
  slm_box)  ARGS="--model slm_sam2 --slm-sam2-ckpt $SLM_CKPT --slm-prompt box";  RUN=slm_sam2_box ;;
  slm_mask) ARGS="--model slm_sam2 --slm-sam2-ckpt $SLM_CKPT --slm-prompt mask"; RUN=slm_sam2_mask ;;
  medsam2)  ARGS="--model medsam2"; RUN=medsam2_boxfix ;;
  *) echo "[ERROR] unknown TARGET=$TARGET"; exit 1 ;;
esac
EXTRA=""
if [ "$MODE" = "smoke" ]; then RUN=${RUN}_smoke; EXTRA="--max-cases 1"; fi
SAVE="$OUT_DIR/$RUN/$RUN.pkl"

echo "TARGET=$TARGET MODE=$MODE ENV=$COMP_ENV SAVE=$SAVE started $(date)"
python comparison_test.py $ARGS \
  --dataset-path "$DATA_SEEN" "$DATA_UNSEEN" \
  --weights-dir "$WEIGHTS" \
  --save-path "$SAVE" \
  --device cuda $EXTRA
echo "finished $(date)"
N_ERR=$(grep -c "\[ERROR\]" "$OUT_DIR/$RUN/log.txt" || true)
echo "[ERROR] lines in $OUT_DIR/$RUN/log.txt: ${N_ERR:-0}"
