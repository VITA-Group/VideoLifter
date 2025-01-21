#!/bin/bash

# Change the absolute path first!
# <Absolute_Path>/VideoLifter
DATA_ROOT_DIR="<Absolute_Path>/VideoLifter/data"
OUTPUT_DIR="output"
DATASETS=(
    Tanks
)

SCENES=(
    Museum
)

N_VIEWS=(
    44
)

gs_train_iter=11000

# Function to get the id of an available GPU
get_available_gpu() {
    local mem_threshold=500
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
    $2 < threshold { print $1; exit }
    '
}

# Function: Run task on specified GPU
run_on_gpu() {
    local GPU_ID=$1
    local DATASET=$2
    local SCENE=$3
    local N_VIEW=$4
    local gs_train_iter=$5
    SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    IMAGE_PATH=${SOURCE_PATH}images
    MODEL_PATH=./${OUTPUT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views_${gs_train_iter}/

    # Create necessary directories
    mkdir -p ${MODEL_PATH}

    echo "======================================================="
    echo "Starting process: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="

    # (1) Sparse Point-Based Fragment Registration with 3D Priors
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./sparse_tt.py \
    -s ${SOURCE_PATH} \
    --n_views ${N_VIEW}
 
    # (2) Hierarchical 3D Gaussian Alignment
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train_tt.py \
    --sparse_image_folder ${SOURCE_PATH} \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    --n_views ${N_VIEW}  \
    --scene ${SCENE} \
    --iter ${gs_train_iter}


    # (3) Render Test Views
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render_tt.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    --n_views ${N_VIEW}  \
    --scene ${SCENE} \
    --optim_test_pose_iter 1000 \
    --iter ${gs_train_iter} \
    --eval 

    # (4) Compute Metrics
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./metrics.py \
    -m ${MODEL_PATH}  \
    --iter ${gs_train_iter} \
    --n_views ${N_VIEW}  \


    echo "======================================================="
    echo "Task completed: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="
}

# Main loop
total_tasks=$((${#DATASETS[@]} * ${#SCENES[@]} * ${#N_VIEWS[@]} * ${#gs_train_iter[@]}))
current_task=0

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do
            for gs_train_iter in "${gs_train_iter[@]}"; do
                current_task=$((current_task + 1))
                echo "Processing task $current_task / $total_tasks"

                # Get available GPU
                GPU_ID=$(get_available_gpu)

                # If no GPU is available, wait for a while and retry
                while [ -z "$GPU_ID" ]; do
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU available, waiting 60 seconds before retrying..."
                    sleep 60
                    GPU_ID=$(get_available_gpu)
                done

                # Run the task in the background
                (run_on_gpu $GPU_ID "$DATASET" "$SCENE" "$N_VIEW" "$gs_train_iter") &

                # Wait for 20 seconds before trying to start the next task
                sleep 10
            done
        done
    done
done

# Wait for all background tasks to complete
wait

echo "======================================================="
echo "All tasks completed! Processed $total_tasks tasks in total."
echo "======================================================="