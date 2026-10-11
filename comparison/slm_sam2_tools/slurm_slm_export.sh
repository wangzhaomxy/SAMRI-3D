#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --job-name=SLM_export
#SBATCH --time=0-08:00:00
#SBATCH --partition=general            # Bunya CPU partition (no GPU needed)
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/SAMRI2_log/slurm-slm-export-%j.output
#SBATCH -e /scratch/user/s4670484/SAMRI2_log/slurm-slm-export-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au
#
# Export the SAMRI-3D training split (training_list_3d.pkl, dice >= 0.2 -- same as
# SAMRI-3D) to SLM-SAM2's JPEG/PNG video-folder format, then load it back through
# SLM-SAM2's own data pipeline as a check.
#
# Submit FROM THE SAMRI-3D REPO ROOT:
#   sbatch comparison/slm_sam2_tools/slurm_slm_export.sh                      # smoke (2 vols/dataset)
#   sbatch --export=ALL,MODE=full comparison/slm_sam2_tools/slurm_slm_export.sh
#
# The full export is resumable: re-submitting skips finished volumes.

set -euo pipefail
MODE=${MODE:-smoke}

mkdir -p /scratch/user/s4670484/SAMRI2_log
module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate samri-mi300

cd "$SLURM_SUBMIT_DIR"
[ -f test_vis3d_multi.py ] && [ -d comparison/SLM-SAM2-main ] || {
  echo "[ERROR] submit from the SAMRI-3D repo root"; exit 1; }

TRAIN_LIST=/scratch/user/s4670484/training_lists/training_list_3d.pkl
if [ "$MODE" = "smoke" ]; then
  OUT=/scratch/user/s4670484/Datasets/SLM_SAM2_train_smoke
  EXTRA="--max-volumes-per-dataset 2"
else
  OUT=/scratch/user/s4670484/Datasets/SLM_SAM2_train
  EXTRA=""
fi
echo "MODE=$MODE  OUT=$OUT  started $(date)"

python comparison/slm_sam2_tools/export_slm_dataset.py \
  --training-list "$TRAIN_LIST" --out-dir "$OUT" \
  --dice-threshold 0.2 --num-workers ${SLURM_CPUS_PER_TASK:-16} $EXTRA

echo "---- per-dataset summary ----"
column -s, -t < "$OUT/export_summary.csv" || cat "$OUT/export_summary.csv"
echo "---- non-ok rows (first 20) ----"
grep -v ",ok,\|,exists," "$OUT/export_manifest.csv" | head -20 || true

echo "---- loader check (SLM-SAM2's own pipeline) ----"
python comparison/slm_sam2_tools/check_slm_dataset.py --data-dir "$OUT" --n 50
du -sh "$OUT"
echo "finished $(date)"
