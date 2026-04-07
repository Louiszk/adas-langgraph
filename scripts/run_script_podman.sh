#!/bin/bash

# Change working directory to project root
cd "$(dirname "$0")/.."
# SLURM CONFIGURATIONS HERE

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
#  PODMAN SERVICE MANAGEMENT & ROBUST CLEANUP
# ==============================================================================

# Create a job-specific runtime directory for the socket.
unset XDG_RUNTIME_DIR
export XDG_RUNTIME_DIR=/tmp/$USER/podman-run-${UNIQUE_ID}
mkdir -p $XDG_RUNTIME_DIR/podman
PODMAN_SOCKET="unix://$XDG_RUNTIME_DIR/podman/podman.sock"
export ADAS_PODMAN_SOCKET=$PODMAN_SOCKET

# This function will be called to clean up all job-specific resources on exit.
podman_service_cleanup() {
  echo "--- Cleaning up all resources for Job $UNIQUE_ID ---"
  if [ ! -z "$PODMAN_PID" ]; then
    echo "--> Killing Podman service (PID: $PODMAN_PID)..."
    kill $PODMAN_PID
    sleep 10 # Wait a moment for the process to die
  fi

  # Remove any leftover containers managed by this job's isolated service.
  echo "--> Removing leftover containers..."
  podman rm -af --ignore

  # Remove the unique base image created for this job.
  echo "--> Removing base image: $BASE_IMAGE_NAME..."
  podman image rm -f "$BASE_IMAGE_NAME" --ignore

  # Remove the unique runtime directory.
  echo "--> Removing runtime directory: $XDG_RUNTIME_DIR"
  rm -rf "$XDG_RUNTIME_DIR"
}

# 'trap' ensures cleanup happens even if the script fails or is cancelled.
trap podman_service_cleanup EXIT INT TERM

# Start a job-specific Podman service in the background.
echo "--- Starting Podman REST API service for Job $UNIQUE_ID ---"
echo "--> Service socket: $PODMAN_SOCKET"
podman system service --time=0 $PODMAN_SOCKET &
PODMAN_PID=$! # Capture the Process ID of the background service

# CRITICAL: Wait for the service to initialize before trying to connect.
echo "--> Waiting 4 seconds for the service to be ready (PID: $PODMAN_PID)..."
sleep 4

# ==============================================================================
#  PRE-FLIGHT DEEP CLEANUP
# ==============================================================================
echo "--- Performing Pre-flight Deep Cleanup for Job $UNIQUE_ID ---"
podman stop -a --ignore
podman rm -af --ignore
podman image rm -f "$BASE_IMAGE_NAME" --ignore
echo "--- Pre-flight Cleanup Complete ---"

# ==============================================================================
#  IMAGE PREPARATION STAGE
# ==============================================================================
echo "--- Preparing Custom Base Image '$BASE_IMAGE_NAME' for Job $UNIQUE_ID ---"
# Use the unique ID for the builder container name.
BUILDER_NAME="adas-builder-${UNIQUE_ID}"
echo "--> Step 1: Starting builder container ('$BUILDER_NAME')..."
cid=$(podman run -d --name $BUILDER_NAME python:3.11-slim sleep 3600)
if [ -z "$cid" ]; then echo "!!ERROR: Failed to start builder container."; exit 1; fi

echo "--> Step 2: Installing core dependencies..."
podman exec $cid pip install "langgraph==0.4.8" "langchain_openai==0.3.32" "python-dotenv==1.0.1" "dill==0.3.9"

echo "--> Step 3: Committing to image '$BASE_IMAGE_NAME'..."
podman commit $cid $BASE_IMAGE_NAME

echo "--> Step 4: Cleaning up builder container..."
podman stop $cid
podman rm $cid
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
        --container "podman"

    echo "========================================================="
    echo "           Completed Experiment $i of $END"
    echo "========================================================="
    sleep 20
done

echo "All runs for Job $UNIQUE_ID completed successfully."