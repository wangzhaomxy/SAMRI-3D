#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --job-name=SAMRI2
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu_rocm        # gpu_rocm      gpu_rocm    gpu_cuda
#SBATCH --gres=gpu:mi300x:8         # gpu:mi300x:8  gpu:mi210:2 gpu:h100:2
#SBATCH --account=a_ai_collab
#SBATCH --qos=sdf                   # sdf           gpu         gpu
#SBATCH -o /scratch/user/s4670484/SAMRI2_log/slurm-samri2-%j.output
#SBATCH -e /scratch/user/s4670484/SAMRI2_log/slurm-samri2-%j.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

# make sure log directory exists
mkdir -p /scratch/user/s4670484/SAMRI2_log

module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate samri-mi300  # samri samri-mi300

# Dynamically assign port from job ID to avoid collisions
export MASTER_ADDR=localhost
export MASTER_PORT=$((26000 + RANDOM % 1000))  # Pick a port between 26000 ~ 26999

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Training mode:
#   "2d_emb"   - 2D loss from pre-computed embeddings (train_unified_emb.py)
#   "3d_emb"   - 2D + 3D loss from pre-computed embeddings (train_unified_emb_3dloss.py)
#   "2d_img"   - 2D loss from raw images (train_unified.py)
#   "gvt_tsdf" - GVT v3: 2D + 3D + recon + TSDF shape prior (gvt/train_gvt_emb_tsdf.py)
TRAINING_MODE="gvt_tsdf" # Change this to switch between training modes

# Common paths
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_s.yaml"
INITIAL_CKPT="/scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_small.pt"

# ============================================================================
# Common Hyperparameters (shared across modes)
# ============================================================================
NUM_EPOCHS=50
BATCH_SIZE=2
LR=5e-6
GRAD_ACCUM=1
MIX_RATIO_3D=0.8           # 80% 3D samples, 20% 2D samples
SAMPLES_PER_DATASET=100     # Volumes per dataset per epoch
MAX_SLICES_PER_VIDEO=32
SAVE_EVERY=1
NUM_WORKERS=12

# ============================================================================
# 2D Loss Embedding Mode Configuration (train_unified_emb.py)
# ============================================================================
EMBEDDING_DIR_3D="/scratch/user/s4670484/SAM2_embeddings/"
SAVE_PATH_2D_EMB="/scratch/user/s4670484/Model_save/samri2_2dloss_all3d_emb/"

# ============================================================================
# 3D Loss Embedding Mode Configuration (train_unified_emb_3dloss.py)
# ============================================================================
SAVE_PATH_3D_EMB="/scratch/user/s4670484/Model_save/samri2_5x_3d_emb_256/"
LAMBDA_2D=1.0               # Weight for 2D slice loss
LAMBDA_3D=0               # Weight for 3D volume loss

# ============================================================================
# 2D Loss Image Mode Configuration (train_unified.py)
# ============================================================================
TRAINING_LIST_3D="/scratch/user/s4670484/training_lists/training_list_3d.pkl"
SAVE_PATH_2D_IMG="/scratch/user/s4670484/Model_save/samri2_2d_img/"

# ============================================================================
# GVT v3 Configuration (gvt/train_gvt_emb_tsdf.py)
# ============================================================================
SAVE_PATH_GVT_TSDF="/scratch/user/s4670484/Model_save/samri2_gvt3_tsdf_K16_rec32_AUX1/"

# GVT encoder
GVT_NUM_TOKENS=16
GVT_NUM_LAYERS=2
GVT_NUM_HEADS=4

# Auxiliary reconstruction loss (GVTReconDecoder)
# Set GVT_RECON=false to disable (ablation: TSDF-only)
GVT_RECON=false
GVT_RECON_RES=32            # Low-res reconstruction target (R×R)
LAMBDA_AUX=1.0              # Weight for reconstruction loss (additive)
LAMBDA_AUX_WARMUP=5.0       # λ_aux used during warmup epochs
LAMBDA_AUX_WARMUP_EPOCHS=10

# TSDF shape prior loss (TSDFDecoder)
# Set USE_TSDF=false to disable (ablation: recon-only)
USE_TSDF=true
TSDF_RES=256             # Resolution for TSDF target generation (R×R)
LAMBDA_TSDF=1.0             # Weight for TSDF loss (additive)
TSDF_TAU=32.0               # Truncation threshold in pixels
TSDF_ALPHA=4.0              # Boundary upweighting factor
# Add --tsdf-single-label to the python call below to disable per-label sqrt(area) normalization

# Additional GVT v3 training args
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
MAX_LABELS_PER_SAMPLE=1     # Labels sampled per volume per epoch
LAMBDA_DICE=1.0             # Dice component weight within 2D loss
LAMBDA_FOCAL=1.0            # Focal component weight within 2D loss
SEED=42

# ============================================================================
# Training Execution
# ============================================================================

if [ "$TRAINING_MODE" == "2d_emb" ]; then
    echo "=========================================="
    echo "SAMRI2: 2D Loss Training (from embeddings)"
    echo "=========================================="
    echo "Initial checkpoint: ${INITIAL_CKPT}"
    echo "3D embeddings dir: ${EMBEDDING_DIR_3D}"

    python train_unified_emb.py \
        --sam2-config ${SAM2_CONFIG} \
        --pretrained-ckpt ${INITIAL_CKPT} \
        --embedding-dir ${EMBEDDING_DIR_3D} \
        --save-path ${SAVE_PATH_2D_EMB} \
        --mix-ratio-3d ${MIX_RATIO_3D} \
        --samples-per-dataset ${SAMPLES_PER_DATASET} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum-steps ${GRAD_ACCUM} \
        --lr ${LR} \
        --num-epochs ${NUM_EPOCHS} \
        --max-slices-per-video ${MAX_SLICES_PER_VIDEO} \
        --num-workers ${NUM_WORKERS} \
        --use-amp \
        --save-every ${SAVE_EVERY}

elif [ "$TRAINING_MODE" == "3d_emb" ]; then
    echo "=========================================="
    echo "SAMRI2: 2D + 3D Loss Training (from embeddings)"
    echo "=========================================="
    echo "Initial checkpoint: ${INITIAL_CKPT}"
    echo "3D embeddings dir: ${EMBEDDING_DIR_3D}"
    echo "Lambda 2D: ${LAMBDA_2D}, Lambda 3D: ${LAMBDA_3D}"

    python train_unified_emb_3dloss.py \
        --sam2-config ${SAM2_CONFIG} \
        --pretrained-ckpt ${INITIAL_CKPT} \
        --embedding-dir ${EMBEDDING_DIR_3D} \
        --save-path ${SAVE_PATH_3D_EMB} \
        --mix-ratio-3d ${MIX_RATIO_3D} \
        --samples-per-dataset ${SAMPLES_PER_DATASET} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum-steps ${GRAD_ACCUM} \
        --lr ${LR} \
        --num-epochs ${NUM_EPOCHS} \
        --max-slices-per-video ${MAX_SLICES_PER_VIDEO} \
        --lambda-2d ${LAMBDA_2D} \
        --lambda-3d ${LAMBDA_3D} \
        --num-workers ${NUM_WORKERS} \
        --use-amp \
        --save-every ${SAVE_EVERY}

elif [ "$TRAINING_MODE" == "2d_img" ]; then
    echo "=========================================="
    echo "SAMRI2: 2D Loss Training (from raw images)"
    echo "=========================================="
    echo "Initial checkpoint: ${INITIAL_CKPT}"
    echo "Training list: ${TRAINING_LIST_3D}"

    python train_unified.py \
        --sam2-config ${SAM2_CONFIG} \
        --pretrained-ckpt ${INITIAL_CKPT} \
        --training-list ${TRAINING_LIST_3D} \
        --save-path ${SAVE_PATH_2D_IMG} \
        --mix-ratio-3d ${MIX_RATIO_3D} \
        --samples-per-dataset ${SAMPLES_PER_DATASET} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum-steps ${GRAD_ACCUM} \
        --lr ${LR} \
        --num-epochs ${NUM_EPOCHS} \
        --max-slices-per-video ${MAX_SLICES_PER_VIDEO} \
        --num-workers ${NUM_WORKERS} \
        --use-amp \
        --save-every ${SAVE_EVERY}

elif [ "$TRAINING_MODE" == "gvt_tsdf" ]; then
    echo "=========================================="
    echo "SAMRI2: GVT v3 TSDF Training (from embeddings)"
    echo "=========================================="
    echo "Initial checkpoint: ${INITIAL_CKPT}"
    echo "3D embeddings dir: ${EMBEDDING_DIR_3D}"
    echo "Seg loss:    lambda_2d=${LAMBDA_2D}  lambda_3d=${LAMBDA_3D}"
    echo "GVT encoder: K=${GVT_NUM_TOKENS}  layers=${GVT_NUM_LAYERS}  heads=${GVT_NUM_HEADS}"
    if [ "$GVT_RECON" == "true" ]; then
        echo "Recon loss:  enabled  res=${GVT_RECON_RES}  lambda_aux=${LAMBDA_AUX}  warmup=${LAMBDA_AUX_WARMUP_EPOCHS}ep"
    else
        echo "Recon loss:  DISABLED"
    fi
    if [ "$USE_TSDF" == "true" ]; then
        echo "TSDF loss:   enabled  tau=${TSDF_TAU}  alpha=${TSDF_ALPHA}  lambda_tsdf=${LAMBDA_TSDF}"
    else
        echo "TSDF loss:   DISABLED"
    fi

    python gvt/train_gvt_emb_tsdf.py \
        --sam2-config ${SAM2_CONFIG} \
        --pretrained-ckpt ${INITIAL_CKPT} \
        --embedding-dir ${EMBEDDING_DIR_3D} \
        --save-path ${SAVE_PATH_GVT_TSDF} \
        --use-gvt \
        --gvt-num-tokens ${GVT_NUM_TOKENS} \
        --gvt-num-layers ${GVT_NUM_LAYERS} \
        --gvt-num-heads ${GVT_NUM_HEADS} \
        $([ "$GVT_RECON" == "true" ] && echo "--gvt-recon") \
        --gvt-recon-res ${GVT_RECON_RES} \
        --lambda-aux ${LAMBDA_AUX} \
        --lambda-aux-warmup ${LAMBDA_AUX_WARMUP} \
        --lambda-aux-warmup-epochs ${LAMBDA_AUX_WARMUP_EPOCHS} \
        $([ "$USE_TSDF" == "true" ] && echo "--use-tsdf") \
        --lambda-tsdf ${LAMBDA_TSDF} \
        --tsdf-tau ${TSDF_TAU} \
        --tsdf-alpha ${TSDF_ALPHA} \
        --mix-ratio-3d ${MIX_RATIO_3D} \
        --samples-per-dataset ${SAMPLES_PER_DATASET} \
        --max-slices-per-video ${MAX_SLICES_PER_VIDEO} \
        --max-labels-per-sample ${MAX_LABELS_PER_SAMPLE} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum-steps ${GRAD_ACCUM} \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --max-grad-norm ${MAX_GRAD_NORM} \
        --num-epochs ${NUM_EPOCHS} \
        --lambda-2d ${LAMBDA_2D} \
        --lambda-3d ${LAMBDA_3D} \
        --lambda-dice ${LAMBDA_DICE} \
        --lambda-focal ${LAMBDA_FOCAL} \
        --num-workers ${NUM_WORKERS} \
        --seed ${SEED} \
        --use-amp \
        --save-every ${SAVE_EVERY}

else
    echo "Error: TRAINING_MODE must be '2d_emb', '3d_emb', '2d_img', or 'gvt_tsdf'"
    echo "Current value: ${TRAINING_MODE}"
    exit 1
fi

echo "=========================================="
echo "Training completed!"
echo "=========================================="
