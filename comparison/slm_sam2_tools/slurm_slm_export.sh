#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --job-name=SLM_export
#SBATCH --time=0-08:00:00
#SBATCH --partition=general
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/SAMRI2_log/slurm-slm-export-%j.output
#SBATCH -e /scratch/user/s4670484/SAMRI2_log/slurm-slm-export-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au
#
# CPU job. Export the SAMRI-3D training split (training_list_3d.pkl, dice >= 0.2, i.e. the
# same volumes + labels SAMRI-3D trained on) to SLM-SAM2's JPEG/PNG video-folder format,
# then load it back through SLM-SAM2's own data pipeline as a check.
#
#   sbatch /scratch/user/s4670484/SAMRI2/comparison/slm_sam2_tools/slurm_slm_export.sh                       # smoke: 2 vols/dataset
#   sbatch --export=ALL,MODE=full /scratch/user/s4670484/SAMRI2/comparison/slm_sam2_tools/slurm_slm_export.sh
#
# Resumable: re-submitting skips volumes that were already exported.

set -uo pipefail
MODE=${MODE:-smoke}
source /scratch/user/s4670484/SAMRI2/comparison/slm_sam2_tools/bunya_common.sh

TRAIN_LIST=/scratch/user/s4670484/training_lists/training_list_3d.pkl
if [ "$MODE" = "smoke" ]; then
  OUT=/scratch/user/s4670484/Datasets/SLM_SAM2_train_smoke; EXTRA="--max-volumes-per-dataset 2"
else
  OUT=/scratch/user/s4670484/Datasets/SLM_SAM2_train;       EXTRA=""
fi
require_paths "$TRAIN_LIST" || exit 1
echo "MODE=$MODE  OUT=$OUT"

python comparison/slm_sam2_tools/export_slm_dataset.py \
  --training-list "$TRAIN_LIST" --out-dir "$OUT" \
  --dice-threshold 0.2 --num-workers "$CPUS" $EXTRA || { echo "[ERROR] export failed"; exit 1; }

echo "---- per-dataset summary ----"
cat "$OUT/export_summary.csv"
echo "---- rows not ok (first 20) ----"
grep -v ",ok,\|,exists," "$OUT/export_manifest.csv" | head -20

echo "---- loader check (SLM-SAM2's own pipeline) ----"
python comparison/slm_sam2_tools/check_slm_dataset.py --data-dir "$OUT" --n 50 \
  || { echo "[ERROR] loader check failed"; exit 1; }
du -sh "$OUT"
echo "finished $(date)"
