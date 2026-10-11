# bunya_common.sh -- sourced by the slurm_*.sh jobs in this folder (not run directly).
# Encodes the Bunya rules: card -> conda env, ROCm env vars, GPU preflight.
#
# Card -> env (one row of the Bunya combination table):
#   mi300x, mi210 -> samri-mi300                                  (torch 2.7.0+rocm6.3)
#   mi355x        -> /scratch/user/s4670484/conda/samri-mi355x    (torch 2.13.0+rocm7.2, path-based)
#   h100          -> samri                                        (torch 2.10.0+cu128)
#   no GPU        -> samri-mi300
# Override with --export=ALL,ENV=<env-name-or-path>.

CODE_ROOT=${CODE_ROOT:-/scratch/user/s4670484/SAMRI2}

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"

CARD=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null \
       | grep -oiE 'gpu:(mi300x|mi355x|mi210|h100)' | head -1 | cut -d: -f2 | tr 'A-Z' 'a-z')
CARD=${CARD:-none}
if [ -z "${ENV:-}" ]; then
  case "$CARD" in
    mi300x|mi210) ENV=samri-mi300 ;;
    mi355x)       ENV=/scratch/user/s4670484/conda/samri-mi355x ;;
    h100)         ENV=samri ;;
    *)            ENV=samri-mi300 ;;
  esac
fi
conda activate "$ENV" || { echo "[ERROR] conda activate $ENV failed"; exit 1; }

cd "$CODE_ROOT" || { echo "[ERROR] cannot cd to CODE_ROOT=$CODE_ROOT"; exit 1; }
[ -f test_vis3d_multi.py ] && [ -f comparison_test.py ] && [ -d comparison/SLM-SAM2-main ] \
  && [ -d comparison/slm_sam2_tools ] || {
  echo "[ERROR] $CODE_ROOT is missing comparison_test.py / comparison/SLM-SAM2-main / comparison/slm_sam2_tools"
  exit 1; }

# ROCm / threading (harmless on CUDA and CPU nodes)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MIOPEN_USER_DB_PATH="/dev/shm/miopen-${SLURM_JOB_ID:-$$}"   # SQLite needs POSIX locks: not GPFS
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

NGPU="${SLURM_GPUS_ON_NODE:-0}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"

echo "==== $(date) | job ${SLURM_JOB_ID:-?} on $(hostname) | card=$CARD NGPU=$NGPU CPUS=$CPUS"
echo "==== env=$ENV | python=$(command -v python) | code=$CODE_ROOT"
echo "==== visibility: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-}"

# gpu_preflight <python modules...> : fail in seconds if the env cannot see the card
gpu_preflight() {
  python - "$CARD" "$@" <<'PY'
import importlib.util, sys
card, mods = sys.argv[1], sys.argv[2:]
import torch
print(f"[PREFLIGHT] {sys.executable} | torch {torch.__version__} | hip {torch.version.hip} | cuda {torch.version.cuda}")
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"[PREFLIGHT] missing python packages: {missing}  ->  pip install {' '.join(missing)}")
if not torch.cuda.is_available():
    sys.exit("[PREFLIGHT] no GPU visible -- wrong env for this card?")
if card.startswith("mi") and torch.version.hip is None:
    sys.exit(f"[PREFLIGHT] card {card} needs a ROCm torch build, got {torch.__version__}")
print(f"[PREFLIGHT] {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
PY
}

# require_paths <path...>
require_paths() {
  for p in "$@"; do
    [ -e "$p" ] || { echo "[PREFLIGHT] not found: $p"; return 1; }
  done
}
