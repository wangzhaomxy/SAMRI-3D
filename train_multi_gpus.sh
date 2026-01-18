#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --job-name=SAMRI2_S1
#SBATCH --time=168:00:00
#SBATCH --partition=gpu_rocm        # gpu_rocm      gpu_rocm    gpu_cuda
#SBATCH --gres=gpu:mi210:2           # gpu:mi300x:8  gpu:mi210:2 gpu:h100:2 
#SBATCH --account=a_ai_collab
#SBATCH --qos=gpu                   # sdf           gpu         gpu
#SBATCH -o /home/s4670484/Documents/SAMRI2_log/slurm-samri2phase1-%j.output
#SBATCH -e /home/s4670484/Documents/SAMRI2_log/slurm-samri2phase1-%j.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate samri-mi300

# Dynamically assign port from job ID to avoid collisions
export MASTER_ADDR=localhost
export MASTER_PORT=$((26000 + RANDOM % 1000))  # Pick a port between 26000 ~ 26999

# ============================================================================
# SAMRI2 Stage 1: Slice-wise mask decoder training
# ============================================================================

# Paths - modify these according to your setup
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT="/scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt"
SAVE_PATH="/scratch/project/samri/Model_save/samri2_stage1/"

# Data mode: "embeddings" or "images"
DATA_MODE="embeddings"

# For embedding mode (faster training with pre-computed embeddings)
EMBEDDING_FILE="/scratch/project/samri/train_list.pkl"
SUBSET="60_up"  # Options: "60_up", "60_down", "all"

# For image mode (raw NIfTI images)
TRAIN_DATA_PATHS="/scratch/project/samri/Datasets/SAMRI_train_test/*/training/"

# Training hyperparameters
NUM_EPOCHS=200
BATCH_SIZE=256
LR=1e-5
PROMPT_MODE="mixed"
SAVE_EVERY=5

if [ "$DATA_MODE" == "embeddings" ]; then
    python train_stage_one_slicewise.py \
        --sam2-config ${SAM2_CONFIG} \
        --ckpt-path ${SAM2_CKPT} \
        --use-embeddings \
        --embedding-file ${EMBEDDING_FILE} \
        --subset ${SUBSET} \
        --save-path ${SAVE_PATH} \
        --num-epochs ${NUM_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --prompt-mode ${PROMPT_MODE} \
        --save-every ${SAVE_EVERY} \
        --num-workers 8
else
    python train_stage_one_slicewise.py \
        --sam2-config ${SAM2_CONFIG} \
        --ckpt-path ${SAM2_CKPT} \
        --train-data-paths ${TRAIN_DATA_PATHS} \
        --save-path ${SAVE_PATH} \
        --num-epochs ${NUM_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --prompt-mode ${PROMPT_MODE} \
        --save-every ${SAVE_EVERY} \
        --num-workers 8
fi

