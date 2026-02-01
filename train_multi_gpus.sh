#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1024G
#SBATCH --job-name=SAMRI2
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu_rocm        # gpu_rocm      gpu_rocm    gpu_cuda
#SBATCH --gres=gpu:mi300x:4         # gpu:mi300x:8  gpu:mi210:2 gpu:h100:2
#SBATCH --account=a_ai_collab
#SBATCH --qos=gpu                   # sdf           gpu         gpu
#SBATCH -o /home/s4670484/Documents/SAMRI2_log/slurm-samri2-%j.output
#SBATCH -e /home/s4670484/Documents/SAMRI2_log/slurm-samri2-%j.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate samri-mi300  # samri samri-mi300

# Dynamically assign port from job ID to avoid collisions
export MASTER_ADDR=localhost
export MASTER_PORT=$((26000 + RANDOM % 1000))  # Pick a port between 26000 ~ 26999

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Training stage: "1" for slice-wise, "2" for 3D-wise with memory bank
TRAINING_STAGE="2"

# Common paths
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT="/scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt"

# ============================================================================
# Stage 1 Configuration
# ============================================================================
# Data mode: "embeddings", "images", or "training_list"
DATA_MODE="training_list"

# For embedding mode (faster training with pre-computed embeddings)
EMBEDDING_DIR="/scratch/project/samri/SAM2_embeddings/"

# For image mode (raw NIfTI images)
TRAIN_DATA_PATHS="/scratch/project/samri/Datasets/SAMRI_train_test/*/training/"

# For training_list mode (recommended - stratified sampling with dice filtering)
TRAINING_LIST_2D="/scratch/project/samri/training_lists/training_list_2d.pkl"

# Stage 1 hyperparameters
SAVE_PATH_S1="/scratch/project/samri/Model_save/samri2_stage1/"
NUM_EPOCHS_S1=100
BATCH_SIZE_S1=512
LR_S1=1e-5
PROMPT_MODE="mixed"
DICE_THRESHOLD_S1=0.4
SAMPLES_PER_DATASET_S1=1024
SAVE_EVERY_S1=1
NUM_WORKERS_S1=16

# ============================================================================
# Stage 2 Configuration
# ============================================================================
# Stage 1 checkpoint (required for stage 2)
STAGE1_CKPT="/scratch/project/samri/Model_save/samri2_stratified/sam2_maskdecoder_epoch_50.pth"

# Training lists (required for stage 2)
TRAINING_LIST_3D="/scratch/project/samri/training_lists/training_list_3d.pkl"

# Stage 2 hyperparameters
SAVE_PATH_S2="/scratch/project/samri/Model_save/samri2_stage2/"
NUM_EPOCHS_S2=50
BATCH_SIZE_S2=4            # Increased from 2 (GPU mem underutilized)
LR_S2=2e-5                 # Scale LR with batch size (sqrt scaling)
GRAD_ACCUM=1               # Reduced from 2 (larger batch already)
MIX_RATIO_3D=0.8
DICE_THRESHOLD_2D=0.4
DICE_THRESHOLD_3D=0.2
SAMPLES_PER_DATASET_2D=50
SAMPLES_PER_DATASET_3D=10
MAX_SLICES=64
MAX_SLICES_PER_VIDEO=48    # Increased from 32 (more GPU mem available)
SAVE_EVERY_S2=1
NUM_WORKERS_S2=64          # Increased from 32 (more CPU available)
PREFETCH_FACTOR=4          # New: prefetch more batches

# ============================================================================
# Training Execution
# ============================================================================

if [ "$TRAINING_STAGE" == "1" ]; then
    echo "=========================================="
    echo "SAMRI2 Stage 1: Slice-wise Training"
    echo "=========================================="

    if [ "$DATA_MODE" == "embeddings" ]; then
        echo "Using pre-computed embeddings from: ${EMBEDDING_DIR}"
        python train_stage_one_slicewise.py \
            --sam2-config ${SAM2_CONFIG} \
            --ckpt-path ${SAM2_CKPT} \
            --use-embeddings --sam2-embeddings \
            --embedding-dir ${EMBEDDING_DIR} \
            --save-path ${SAVE_PATH_S1} \
            --num-epochs ${NUM_EPOCHS_S1} \
            --batch-size ${BATCH_SIZE_S1} \
            --lr ${LR_S1} \
            --prompt-mode ${PROMPT_MODE} \
            --save-every ${SAVE_EVERY_S1} \
            --num-workers ${NUM_WORKERS_S1}

    elif [ "$DATA_MODE" == "training_list" ]; then
        echo "Using stratified training list: ${TRAINING_LIST_2D}"
        python train_stage_one_slicewise.py \
            --sam2-config ${SAM2_CONFIG} \
            --ckpt-path ${SAM2_CKPT} \
            --training-list ${TRAINING_LIST_2D} \
            --dice-threshold ${DICE_THRESHOLD_S1} \
            --samples-per-dataset ${SAMPLES_PER_DATASET_S1} \
            --save-path ${SAVE_PATH_S1} \
            --num-epochs ${NUM_EPOCHS_S1} \
            --batch-size ${BATCH_SIZE_S1} \
            --lr ${LR_S1} \
            --prompt-mode ${PROMPT_MODE} \
            --save-every ${SAVE_EVERY_S1} \
            --num-workers ${NUM_WORKERS_S1}

    else
        echo "Using raw images from: ${TRAIN_DATA_PATHS}"
        python train_stage_one_slicewise.py \
            --sam2-config ${SAM2_CONFIG} \
            --ckpt-path ${SAM2_CKPT} \
            --train-data-paths ${TRAIN_DATA_PATHS} \
            --save-path ${SAVE_PATH_S1} \
            --num-epochs ${NUM_EPOCHS_S1} \
            --batch-size ${BATCH_SIZE_S1} \
            --lr ${LR_S1} \
            --prompt-mode ${PROMPT_MODE} \
            --save-every ${SAVE_EVERY_S1} \
            --num-workers ${NUM_WORKERS_S1}
    fi

elif [ "$TRAINING_STAGE" == "2" ]; then
    echo "=========================================="
    echo "SAMRI2 Stage 2: 3D-wise Training"
    echo "=========================================="
    echo "Stage 1 checkpoint: ${STAGE1_CKPT}"
    echo "2D training list: ${TRAINING_LIST_2D}"
    echo "3D training list: ${TRAINING_LIST_3D}"

    python train_stage_two_3dwise.py \
        --sam2-config ${SAM2_CONFIG} \
        --stage1-ckpt ${STAGE1_CKPT} \
        --training-list-2d ${TRAINING_LIST_2D} \
        --training-list-3d ${TRAINING_LIST_3D} \
        --save-path ${SAVE_PATH_S2} \
        --dice-threshold-2d ${DICE_THRESHOLD_2D} \
        --dice-threshold-3d ${DICE_THRESHOLD_3D} \
        --mix-ratio-3d ${MIX_RATIO_3D} \
        --samples-per-dataset-2d ${SAMPLES_PER_DATASET_2D} \
        --samples-per-dataset-3d ${SAMPLES_PER_DATASET_3D} \
        --batch-size ${BATCH_SIZE_S2} \
        --grad-accum-steps ${GRAD_ACCUM} \
        --lr ${LR_S2} \
        --num-epochs ${NUM_EPOCHS_S2} \
        --max-slices ${MAX_SLICES} \
        --max-slices-per-video ${MAX_SLICES_PER_VIDEO} \
        --num-workers ${NUM_WORKERS_S2} \
        --use-amp \
        --save-every ${SAVE_EVERY_S2}

else
    echo "Error: TRAINING_STAGE must be '1' or '2'"
    echo "Current value: ${TRAINING_STAGE}"
    exit 1
fi

echo "=========================================="
echo "Training completed!"
echo "=========================================="
