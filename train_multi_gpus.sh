#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --job-name=SAMRI2
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu_rocm        # gpu_rocm      gpu_rocm    gpu_cuda
#SBATCH --gres=gpu:mi300x:3         # gpu:mi300x:8  gpu:mi210:2 gpu:h100:2
#SBATCH --account=a_ai_collab
#SBATCH --qos=gpu                   # sdf           gpu         gpu
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
#   "gvt_v1"   - GVT v1: GVT encoder only (seg loss, no aux decoder)
#   "gvt_v2"   - GVT v2: GVT encoder + GVTReconDecoder (label-conditioned recon loss)
#   "gvt_v3"   - GVT v3: GVT encoder + GVTReconDecoder + TSDFDecoder (recon + TSDF)
TRAINING_MODE="gvt_v1"  # Change this to switch between training modes

# Common paths
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_s.yaml"
INITIAL_CKPT="/scratch/user/s4670484/pretrained_ckpt/sam2.1_hiera_small.pt"

# ============================================================================
# Common Hyperparameters (shared across modes)
# ============================================================================
NUM_EPOCHS=50
BATCH_SIZE=1
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
LAMBDA_3D=0.0               # Weight for 3D volume loss

# ============================================================================
# 2D Loss Image Mode Configuration (train_unified.py)
# ============================================================================
TRAINING_LIST_3D="/scratch/user/s4670484/training_lists/training_list_3d.pkl"
SAVE_PATH_2D_IMG="/scratch/user/s4670484/Model_save/samri2_2d_img/"

# ============================================================================
# GVT Configuration (gvt/train_gvt_emb_tsdf.py) — used by gvt_v1/v2/v3
# ============================================================================
SAVE_PATH_GVT_V1="/scratch/user/s4670484/Model_save/samri2_gvt1_K16_perlabel/"
SAVE_PATH_GVT_V2="/scratch/user/s4670484/Model_save/samri2_gvt2_recon_K16_rec32_new/"
SAVE_PATH_GVT_TSDF="/scratch/user/s4670484/Model_save/samri2_gvt3_tsdf_K16_AUX1_gated_3d/"

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

# Gated Attention (Priority 1 experiment)
# Set to "true" to enable Y' = Y ⊙ σ(X·W_G) gating on memory attention output
USE_GATED_ATTENTION=false

# Additional GVT v3 training args
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
MAX_LABELS_PER_SAMPLE=1     # Labels sampled per volume per epoch
LAMBDA_DICE=1.0             # Dice component weight within 2D loss
LAMBDA_FOCAL=1.0            # Focal component weight within 2D loss
SEED=42

# ============================================================================
# GVT Version Normalization
# Translate gvt_v1/v2/v3 aliases into gvt_tsdf mode with appropriate flag overrides.
# This avoids duplicating the python invocation for each GVT variant.
# ============================================================================
GVT_VERSION_LABEL=""  # Human-readable label printed in job header

if [ "$TRAINING_MODE" == "gvt_v1" ]; then
    GVT_VERSION_LABEL="v1 (encoder only — seg loss, no aux decoder)"
    GVT_RECON=false
    USE_TSDF=false
    SAVE_PATH_GVT_TSDF="${SAVE_PATH_GVT_V1}"
    TRAINING_MODE="gvt_tsdf"

elif [ "$TRAINING_MODE" == "gvt_v2" ]; then
    GVT_VERSION_LABEL="v2 (encoder + GVTReconDecoder — label-conditioned recon loss)"
    GVT_RECON=true
    USE_TSDF=false
    SAVE_PATH_GVT_TSDF="${SAVE_PATH_GVT_V2}"
    TRAINING_MODE="gvt_tsdf"

elif [ "$TRAINING_MODE" == "gvt_v3" ]; then
    GVT_VERSION_LABEL="v3 (encoder + GVTReconDecoder + TSDFDecoder)"
    GVT_RECON=true
    USE_TSDF=true
    # SAVE_PATH_GVT_TSDF already set above
    TRAINING_MODE="gvt_tsdf"
fi

# ============================================================================
# Training Execution
# ============================================================================

# Print common job and hardware info
echo ""
echo "############################################################"
echo "#                  SAMRI2 Training Job                     #"
echo "############################################################"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Node         : $(hostname)"
echo "  GPUs         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | tr '\n' ' ' || echo 'N/A')"
echo "  CPU cores    : ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "  Memory       : ${SLURM_MEM_PER_NODE:-N/A} MB"
echo "  Started      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Log output   : ${SLURM_STDOUT:-stdout}"
echo "############################################################"
echo ""
echo "  Training mode    : ${TRAINING_MODE}"
echo "  SAM2 config      : ${SAM2_CONFIG}"
echo "  Initial ckpt     : ${INITIAL_CKPT}"
echo ""
echo "  --- Common Hyperparameters ---"
echo "  Epochs           : ${NUM_EPOCHS}"
echo "  Batch size       : ${BATCH_SIZE}"
echo "  Learning rate    : ${LR}"
echo "  Grad accum steps : ${GRAD_ACCUM}"
echo "  Effective batch  : $((BATCH_SIZE * GRAD_ACCUM)) (batch × accum)"
echo "  Mix ratio 3D     : ${MIX_RATIO_3D}  (3D fraction of batches)"
echo "  Samples/dataset  : ${SAMPLES_PER_DATASET}  (volumes per dataset per epoch)"
echo "  Max slices/video : ${MAX_SLICES_PER_VIDEO}"
echo "  Save every       : ${SAVE_EVERY} epoch(s)"
echo "  DataLoader wkrs  : ${NUM_WORKERS}"
echo "  AMP              : enabled"
echo "############################################################"
echo ""

if [ "$TRAINING_MODE" == "2d_emb" ]; then
    echo "  *** Mode: 2D Loss from Pre-computed Embeddings ***"
    echo "  Script           : train_unified_emb.py"
    echo "  Embeddings dir   : ${EMBEDDING_DIR_3D}"
    echo "  Save path        : ${SAVE_PATH_2D_EMB}"
    echo "  Loss             : 2D Dice-Focal (per-slice)"
    echo "  Lambda 2D        : 1.0  (fixed)"
    echo "############################################################"
    echo ""

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
    echo "  *** Mode: 2D + 3D Loss from Pre-computed Embeddings ***"
    echo "  Script           : train_unified_emb_3dloss.py"
    echo "  Embeddings dir   : ${EMBEDDING_DIR_3D}"
    echo "  Save path        : ${SAVE_PATH_3D_EMB}"
    echo "  Loss             : 2D Dice-Focal + 3D volumetric Dice"
    echo "  Lambda 2D        : ${LAMBDA_2D}"
    echo "  Lambda 3D        : ${LAMBDA_3D}"
    echo "############################################################"
    echo ""

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
    echo "  *** Mode: 2D Loss from Raw Images ***"
    echo "  Script           : train_unified.py"
    echo "  Training list    : ${TRAINING_LIST_3D}"
    echo "  Save path        : ${SAVE_PATH_2D_IMG}"
    echo "  Loss             : 2D Dice-Focal (per-slice, no pre-computed embeddings)"
    echo "############################################################"
    echo ""

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
    echo "  *** Mode: GVT ${GVT_VERSION_LABEL} ***"
    echo "  Script           : gvt/train_gvt_emb_tsdf.py"
    echo "  Embeddings dir   : ${EMBEDDING_DIR_3D}"
    echo "  Save path        : ${SAVE_PATH_GVT_TSDF}"
    echo ""
    echo "  --- GVT Encoder ---"
    echo "  Tokens (K)       : ${GVT_NUM_TOKENS}"
    echo "  Layers           : ${GVT_NUM_LAYERS}"
    echo "  Heads            : ${GVT_NUM_HEADS}"
    echo ""
    echo "  --- Segmentation Loss ---"
    echo "  Lambda 2D        : ${LAMBDA_2D}  (Dice-Focal, per-slice)"
    echo "  Lambda 3D        : ${LAMBDA_3D}  (volumetric Dice; 0 = disabled)"
    echo "  Lambda Dice      : ${LAMBDA_DICE}  (Dice component of 2D loss)"
    echo "  Lambda Focal     : ${LAMBDA_FOCAL}  (Focal component of 2D loss)"
    echo ""
    echo "  --- Auxiliary Losses ---"
    if [ "$GVT_RECON" == "true" ]; then
        echo "  Recon loss       : ENABLED"
        echo "  Recon res        : ${GVT_RECON_RES}x${GVT_RECON_RES}"
        echo "  Lambda aux       : ${LAMBDA_AUX}  (warmup: ${LAMBDA_AUX_WARMUP} for first ${LAMBDA_AUX_WARMUP_EPOCHS} ep)"
    else
        echo "  Recon loss       : DISABLED"
    fi
    if [ "$USE_TSDF" == "true" ]; then
        echo "  TSDF loss        : ENABLED"
        echo "  TSDF tau         : ${TSDF_TAU}  (truncation threshold, pixels)"
        echo "  TSDF alpha       : ${TSDF_ALPHA}  (boundary upweighting)"
        echo "  Lambda TSDF      : ${LAMBDA_TSDF}"
    else
        echo "  TSDF loss        : DISABLED"
    fi
    if [ "$USE_GATED_ATTENTION" == "true" ]; then
        echo "  Gated attention  : ENABLED  (Y' = Y ⊙ σ(X·W_G), init_bias=-2.0)"
    else
        echo "  Gated attention  : DISABLED"
    fi
    echo ""
    echo "  --- Sampling ---"
    echo "  Max labels/sample: ${MAX_LABELS_PER_SAMPLE}  (labels per volume per step)"
    echo "  Weight decay     : ${WEIGHT_DECAY}"
    echo "  Max grad norm    : ${MAX_GRAD_NORM}"
    echo "  Seed             : ${SEED}"
    echo "############################################################"
    echo ""

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
        $([ "$USE_GATED_ATTENTION" == "true" ] && echo "--use-gated-attention") \
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
    echo "Error: TRAINING_MODE must be one of: '2d_emb', '3d_emb', '2d_img', 'gvt_v1', 'gvt_v2', 'gvt_v3'"
    echo "Current value: ${TRAINING_MODE}"
    exit 1
fi

echo ""
echo "############################################################"
echo "  Training completed : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Mode               : ${TRAINING_MODE}"
echo "############################################################"
