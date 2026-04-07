#!/bin/bash

# Change working directory to project root
cd "$(dirname "$0")/.."
# SLURM CONFIGURATIONS HERE

# 1. Names of the generated systems to run (Space separated list inside parentheses)
#    The script will iterate through each one using the exact same state.

SYSTEM_NAMES=(
    "data_analyst_gpt5_v0"
    "data_analyst_gpt5_v1"
    "data_analyst_gpt5_v2"
    "data_analyst_gpt5_v3"
    "data_analyst_gpt5_v4"
    "data_analyst_gpt4_v0"
    "data_analyst_gpt4_v1"
    "data_analyst_gpt4_v2"
    "data_analyst_gpt4_v3"
    "data_analyst_gpt4_v4"
)

# 2. Initial State JSON (as a string)
INPUT_STATE='{"analysis_task": "Create a heatmap (viridis colormap) showing total minutes watched for each hour of the day (x-axis) and each weekday (y-axis)."}'

# 3. Optional: Data Generation Script to run before the systems
DATA_GEN_SCRIPT="" # "generated_systems/use_cases/generate_data.py"

# ==============================================================================

UNIQUE_ID=${SLURM_JOB_ID:-$$}
BASE_IMAGE_NAME="adas-base-job:${UNIQUE_ID}"

# --- Environment Setup ---
source $HOME/.bashrc
eval "$(conda shell.bash hook)"
conda activate py311

# Ensure jq is installed (crucial for parsing metrics)
if ! command -v jq &> /dev/null; then
    echo "ERROR: 'jq' command not found. Please install jq." >&2
    exit 1
fi

# --- Data Preparation ---
# We run this ONCE so all systems see the exact same generated data
mkdir -p data/input
mkdir -p data/output

if [ -n "$DATA_GEN_SCRIPT" ] && [ -f "$DATA_GEN_SCRIPT" ]; then
    echo "--- Running Data Generation Script: $DATA_GEN_SCRIPT ---"
    python "$DATA_GEN_SCRIPT"
    if [ $? -ne 0 ]; then
        echo "Error: Data generation failed."
        exit 1
    fi
    echo "Data generated in data/input/"
fi

#  PODMAN SERVICE MANAGEMENT
unset XDG_RUNTIME_DIR
export XDG_RUNTIME_DIR=/tmp/$USER/podman-run-${UNIQUE_ID}
mkdir -p $XDG_RUNTIME_DIR/podman
PODMAN_SOCKET="unix://$XDG_RUNTIME_DIR/podman/podman.sock"
export ADAS_PODMAN_SOCKET=$PODMAN_SOCKET

podman_service_cleanup() {
  echo "--- Cleaning up resources ---"
  if [ ! -z "$PODMAN_PID" ]; then
    kill $PODMAN_PID 2>/dev/null
    sleep 2
  fi
  podman rm -af --ignore
  # Remove all temp images created during the loop
  podman images --filter "reference=adas-temp-image-*" -q | xargs podman rmi -f --ignore
  podman image rm -f "$BASE_IMAGE_NAME" --ignore
  if [ -d "${XDG_RUNTIME_DIR:-}" ]; then rm -rf "$XDG_RUNTIME_DIR"; fi
}
trap podman_service_cleanup EXIT INT TERM

echo "--- Starting Podman Service ---"
podman system service --time=0 $PODMAN_SOCKET &
PODMAN_PID=$!
sleep 5

echo "--- Preparing Base Image '$BASE_IMAGE_NAME' ---"
# 1. Create Standard Base Image (Created once, reused for all systems)
cid=$(podman run -d --name "adas-base-${UNIQUE_ID}" python:3.11-slim sleep 3600)
podman exec $cid pip install "langgraph==0.4.8" "langchain_openai==0.3.32" "python-dotenv==1.0.1" "dill==0.3.9"
podman commit $cid $BASE_IMAGE_NAME
podman stop $cid; podman rm $cid

# Iterate through the array of systems
for SYSTEM_NAME in "${SYSTEM_NAMES[@]}"; do

    echo "-------------------------------------------------"
    echo " STARTING RUN FOR: $SYSTEM_NAME"
    echo "-------------------------------------------------"

    # --- Check for System-Specific Dependencies ---
    METRICS_FILE="generated_systems/metrics/${SYSTEM_NAME}.json"
    FINAL_IMAGE_TO_USE="$BASE_IMAGE_NAME"
    TEMP_IMAGE_NAME=""

    if [ -f "$METRICS_FILE" ]; then
        DEPENDENCIES=$(jq -r '.installed_packages // empty' "$METRICS_FILE")

        if [ -n "$DEPENDENCIES" ] && [ "$DEPENDENCIES" != "null" ]; then
            echo "--- Found custom dependencies for $SYSTEM_NAME: $DEPENDENCIES ---"

            # Create a unique temp image name for this specific system run
            TEMP_IMAGE_NAME="adas-temp-image-${UNIQUE_ID}-${SYSTEM_NAME}"

            # Start builder from base image
            builder_cid=$(podman run -d --name "adas-builder-${UNIQUE_ID}-${SYSTEM_NAME}" "$BASE_IMAGE_NAME" sleep 3600)

            # Install specific packages (split string into array)
            read -ra REQS <<< "$DEPENDENCIES"
            echo "--> Installing: ${REQS[*]}"
            if podman exec "$builder_cid" pip install "${REQS[@]}"; then
                podman commit "$builder_cid" "$TEMP_IMAGE_NAME"
                FINAL_IMAGE_TO_USE="$TEMP_IMAGE_NAME"
                echo "--> Created custom image: $TEMP_IMAGE_NAME"
            else
                echo "WARNING: Failed to install dependencies. Falling back to base image."
            fi
            podman stop "$builder_cid"; podman rm "$builder_cid"
        else
            echo "--- No custom dependencies found. Using base image. ---"
        fi
    else
        echo "--- Metrics file not found. Using base image. ---"
    fi

    # --- Run the Test ---
    echo "--> Running Target System: $SYSTEM_NAME"
    echo "--> Image: $FINAL_IMAGE_TO_USE"

    python test_target.py \
        --system_name "$SYSTEM_NAME" \
        --state "$INPUT_STATE" \
        --base-image "$FINAL_IMAGE_TO_USE" \
        --container "podman" \
        --keep-template

    TEST_EXIT_CODE=$?

    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "--> Success: $SYSTEM_NAME"
    else
        echo "--> Failure: $SYSTEM_NAME exited with code $TEST_EXIT_CODE"
    fi

    # Optional: Clean up the specific temp image immediately to save space,
    # or let the trap handle it at the very end.
    if [ -n "$TEMP_IMAGE_NAME" ]; then
        podman image rm "$TEMP_IMAGE_NAME" --ignore > /dev/null 2>&1
    fi

    echo "-------------------------------------------------"
    echo " COMPLETED RUN FOR: $SYSTEM_NAME"
    echo "-------------------------------------------------"
    echo ""

done

echo "================================================="
echo "   All System Tests Complete"
echo "================================================="