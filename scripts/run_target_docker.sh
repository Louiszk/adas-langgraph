#!/bin/bash

# Change working directory to project root
cd "$(dirname "$0")/.." || exit

# 1. Names of the generated systems to run
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

# 2. Initial State JSON
INPUT_STATE='{"analysis_task": "Create a heatmap (viridis colormap) showing total minutes watched for each hour of the day (x-axis) and each weekday (y-axis)."}'

# 3. Optional: Data Generation Script
DATA_GEN_SCRIPT="" # "generated_systems/use_cases/generate_data.py"

# ==============================================================================

# Generate a unique ID for this run (using timestamp)
UNIQUE_ID=$(date +%s)
BASE_IMAGE_NAME="adas-base-job:${UNIQUE_ID}"

# --- Environment Setup ---
# Adjust this section to match your local environment (Conda or Venv)
# shellcheck disable=SC1091
source "$HOME"/.bashrc
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate py311 || echo "Warning: Conda env 'py311' not found, using current python."
fi

# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "ERROR: 'jq' command not found. Please install jq (e.g., sudo apt install jq or brew install jq)." >&2
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop or the docker daemon." >&2
    exit 1
fi

# --- Data Preparation ---
mkdir -p data/input
mkdir -p data/output

if [ -n "$DATA_GEN_SCRIPT" ] && [ -f "$DATA_GEN_SCRIPT" ]; then
    echo "--- Running Data Generation Script: $DATA_GEN_SCRIPT ---"
    if ! python "$DATA_GEN_SCRIPT"; then
        echo "Error: Data generation failed."
        exit 1
    fi
    echo "Data generated in data/input/"
fi

# ==============================================================================
#  DOCKER CLEANUP HANDLER
# ==============================================================================
docker_cleanup() {
  echo "--- Cleaning up resources ---"
  
  # Remove temp images created during the loop
  # Docker doesn't support regex in rmi directly, so we filter by name/reference
  docker images --format "{{.Repository}}:{{.Tag}}" | grep "adas-temp-image-${UNIQUE_ID}" | xargs -r docker rmi -f
  
  # Remove the base image
  docker rmi -f "$BASE_IMAGE_NAME" >/dev/null 2>&1
  
  # Prune stopped containers created by this script (optional, but good for hygiene)
  docker container prune -f >/dev/null 2>&1
}
trap docker_cleanup EXIT INT TERM

echo "--- Preparing Base Image '$BASE_IMAGE_NAME' ---"
# 1. Start a temporary container to build dependencies
cid=$(docker run -d --name "adas-base-${UNIQUE_ID}" python:3.11-slim sleep 3600)

# 2. Install core libraries
docker exec "$cid" pip install "langgraph==0.4.8" "langchain_openai==0.3.32" "python-dotenv==1.0.1" "dill==0.3.9"

# 3. Commit to create the base image
docker commit "$cid" "$BASE_IMAGE_NAME"

# 4. Cleanup builder container
docker stop "$cid" >/dev/null
docker rm "$cid" >/dev/null

# ==============================================================================
#  MAIN LOOP
# ==============================================================================

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

            TEMP_IMAGE_NAME="adas-temp-image-${UNIQUE_ID}-${SYSTEM_NAME}"

            # Start builder from base image
            builder_cid=$(docker run -d --name "adas-builder-${UNIQUE_ID}-${SYSTEM_NAME}" "$BASE_IMAGE_NAME" sleep 3600)

            # Install specific packages
            read -ra REQS <<< "$DEPENDENCIES"
            echo "--> Installing: ${REQS[*]}"
            if docker exec "$builder_cid" pip install "${REQS[@]}"; then
                docker commit "$builder_cid" "$TEMP_IMAGE_NAME"
                FINAL_IMAGE_TO_USE="$TEMP_IMAGE_NAME"
                echo "--> Created custom image: $TEMP_IMAGE_NAME"
            else
                echo "WARNING: Failed to install dependencies. Falling back to base image."
            fi
            docker stop "$builder_cid" >/dev/null
            docker rm "$builder_cid" >/dev/null
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
        --container "docker" \
        --keep-template

    TEST_EXIT_CODE=$?

    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "--> Success: $SYSTEM_NAME"
    else
        echo "--> Failure: $SYSTEM_NAME exited with code $TEST_EXIT_CODE"
    fi

    # Clean up the specific temp image immediately
    if [ -n "$TEMP_IMAGE_NAME" ]; then
        docker rmi "$TEMP_IMAGE_NAME" > /dev/null 2>&1
    fi

    echo "-------------------------------------------------"
    echo " COMPLETED RUN FOR: $SYSTEM_NAME"
    echo "-------------------------------------------------"
    echo ""

done

echo "================================================="
echo "   All System Tests Complete"
echo "================================================="