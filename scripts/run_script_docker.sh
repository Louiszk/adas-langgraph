#!/bin/bash

# Change working directory to project root
cd "$(dirname "$0")/.."
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --job-name=adas_run
#SBATCH --output=adas_run_%j.out

# --- Configuration ---
if [ $# -lt 2 ]; then
    echo "Usage: $0 <type> <benchmark> [range]"
    exit 1
fi

TYPE=$1
BENCHMARK=$2
RANGE=${3:-"1-16"}
UNIQUE_ID=${SLURM_JOB_ID:-$$}
BASE_IMAGE_NAME="adas-base-job:${UNIQUE_ID}"

# --- Argument Parsing and Validation ---
if [[ $RANGE == *-* ]]; then
    IFS='-' read -ra RANGE_PARTS <<< "$RANGE"
    START=${RANGE_PARTS[0]}
    END=${RANGE_PARTS[1]}
else
    START=$RANGE
    END=$RANGE
fi

echo "Running with TYPE=$TYPE, BENCHMARK=$BENCHMARK, RANGE=$START-$END"

# --- Environment Setup ---
source $HOME/.bashrc
eval "$(conda shell.bash hook)"
conda activate py311

TARGET_DIR="ADAS_$TYPE"
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

cd "$TARGET_DIR"
echo "Changed directory to $(pwd)"

PROBLEM_PATH=""
if [ "$BENCHMARK" == "gsm" ]; then
    PROBLEM_PATH="generated_systems/GSMHard/prompts.txt"
elif [ "$BENCHMARK" == "mmlu" ]; then
    PROBLEM_PATH="generated_systems/MMLUPro/prompts.txt"
elif [ "$BENCHMARK" == "fever" ]; then
    PROBLEM_PATH="generated_systems/FEVER/prompts.txt"
fi

# ==============================================================================
#  DOCKER CLEANUP MANAGEMENT
# ==============================================================================

# This function will be called to clean up all job-specific resources on exit.
docker_cleanup() {
  echo "--- Cleaning up all resources for Job $UNIQUE_ID ---"
  
  # Remove any leftover containers
  echo "--> Removing leftover containers..."
  docker container prune -f
  
  # Remove the unique base image created for this job.
  echo "--> Removing base image: $BASE_IMAGE_NAME..."
  docker image rm -f "$BASE_IMAGE_NAME" 2>/dev/null || true
}

# 'trap' ensures cleanup happens even if the script fails or is cancelled.
trap docker_cleanup EXIT INT TERM

# ==============================================================================
#  PRE-FLIGHT DEEP CLEANUP
# ==============================================================================
echo "--- Performing Pre-flight Deep Cleanup for Job $UNIQUE_ID ---"

# 1. Stop and remove ONLY containers starting with "adas-"
#    This prevents deleting unrelated containers on the host machine.
docker ps -a --filter "name=adas-" -q | xargs -r docker stop >/dev/null 2>&1 || true
docker ps -a --filter "name=adas-" -q | xargs -r docker rm -f >/dev/null 2>&1 || true

# 2. Remove the specific base image for this job
docker image rm -f "$BASE_IMAGE_NAME" 2>/dev/null || true

# 3. Prune dangling images (safe)
docker image prune -f >/dev/null 2>&1

echo "--- Pre-flight Cleanup Complete ---"

# ==============================================================================
#  IMAGE PREPARATION STAGE
# ==============================================================================
echo "--- Preparing Custom Base Image '$BASE_IMAGE_NAME' for Job $UNIQUE_ID ---"

# Use the unique ID for the builder container name.
BUILDER_NAME="adas-builder-${UNIQUE_ID}"

echo "--> Step 1: Starting builder container ('$BUILDER_NAME')..."
cid=$(docker run -d --name $BUILDER_NAME python:3.11-slim sleep 3600)
if [ -z "$cid" ]; then echo "!!ERROR: Failed to start builder container."; exit 1; fi

echo "--> Step 2: Installing core dependencies..."
docker exec $cid pip install "langgraph==0.4.8" "langchain_openai==0.3.32" "python-dotenv==1.0.1" "dill==0.3.9"

echo "--> Step 3: Committing to image '$BASE_IMAGE_NAME'..."
docker commit $cid $BASE_IMAGE_NAME

echo "--> Step 4: Cleaning up builder container..."
docker stop $cid
docker rm $cid

echo "--- Custom Base Image is Ready ---"

# --- Main Experiment Loop ---
for i in $(seq $START $END)
do
    echo "========================================================="
    echo "           Running Experiment $i of $END"
    echo "           Name: ${TYPE}_${BENCHMARK}${i}_gpt"
    echo "========================================================="
    
    python run_design.py \
        --problem "$(cat $PROBLEM_PATH)" \
        --name "${TYPE}_${BENCHMARK}${i}_gpt" \
        --base-image "$BASE_IMAGE_NAME" \
        --container "docker"
    
    echo "========================================================="
    echo "           Completed Experiment $i of $END"
    echo "========================================================="
    
    sleep 5
done

echo "All runs for Job $UNIQUE_ID completed successfully."
