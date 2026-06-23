#!/bin/bash

# Change working directory to project root
cd "$(dirname "$0")/.." || exit

# --- Configuration ---
UNIQUE_ID=${RANDOM} # Use Random ID for local execution
RESULTS_DIR="$PWD/local_benchmark_results"
DOCKER_CLEANUP_FREQUENCY=8
BASE_IMAGE_NAME="adas-base-job:${UNIQUE_ID}"

# Define your combinations here
COMBINATIONS=(
"" # e.g., ablationC fever 29
)

echo "Starting Local Docker Benchmark Run ($(date)) for Job ID: $UNIQUE_ID"
echo "Configuration:"
echo "  Results Directory: $RESULTS_DIR"
echo "  Cleanup Frequency: Every $DOCKER_CLEANUP_FREQUENCY iterations"
echo "-----------------------------------------"

# Check for Docker
if ! docker info > /dev/null 2>&1; then
    echo "!!ERROR: Docker is not running or you do not have permissions. Aborting."
    exit 1
fi

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "!!ERROR: 'jq' command not found. Please install jq (e.g., 'apt-get install jq' or 'brew install jq'). Aborting."
    exit 1
fi

# Setup Python Environment
# NOTE: Adjust this line to match your local environment name or use 'source venv/bin/activate'
echo "Loading environment..."
# shellcheck disable=SC1091
source "$HOME"/.bashrc # Or path to your conda hook
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate py311 || echo "Warning: Conda env 'py311' not found, using current python."
fi
echo "Environment loaded."

mkdir -p "$RESULTS_DIR"

# --- Cleanup Logic ---

job_cleanup() {
    echo "--- Cleaning up resources for Job $UNIQUE_ID ---"
    
    # Remove the base image created by this script
    if docker image inspect "$BASE_IMAGE_NAME" > /dev/null 2>&1; then
        echo "--> Removing base image $BASE_IMAGE_NAME..."
        docker rmi -f "$BASE_IMAGE_NAME" >/dev/null 2>&1
    fi

    # Cleanup any leftover builder containers specific to this ID
    echo "--> Removing temporary builder containers..."
    docker ps -a --filter "name=adas-builder-${UNIQUE_ID}" -q | xargs -r docker rm -f
    
    # Cleanup any dangling images created during dynamic builds
    echo "--> Cleaning up dangling images..."
    docker image prune -f >/dev/null 2>&1
    
    echo "--- Cleanup Complete ---"
}
trap job_cleanup EXIT INT TERM

# --- Helper Functions ---

parse_combinations() {
    EXPANDED_COMBINATIONS=()
    for combo in "${COMBINATIONS[@]}"; do
        if [ -z "$combo" ]; then continue; fi
        read -r type benchmark iter_range <<< "$combo"
        if [[ "$iter_range" == *-* ]]; then
            local start_iter=${iter_range%-*}
            local end_iter=${iter_range#*-}
            for ((iter=start_iter; iter<=end_iter; iter++)); do
                EXPANDED_COMBINATIONS+=("$type $benchmark $iter")
            done
        else
            EXPANDED_COMBINATIONS+=("$type $benchmark $iter_range")
        fi
    done
}

create_base_image() {
    if ! docker image inspect "$BASE_IMAGE_NAME" > /dev/null 2>&1; then
        echo "--> Custom base image '$BASE_IMAGE_NAME' not found. Creating it..." >&2
        local cid
        local builder_image="python:3.11-slim"
        
        docker pull "$builder_image" >/dev/null
        
        # Run a container to install packages
        cid=$(docker run -d --name "adas-builder-${UNIQUE_ID}" "$builder_image" sleep 3600)

        if [ -z "$cid" ]; then
            echo "!!ERROR: Failed to start the builder container. Aborting." >&2
            exit 1
        fi

        echo "--> Installing core dependencies in builder container..." >&2
        docker exec "$cid" pip install \
            "langgraph==0.4.8" \
            "langchain_openai==0.3.32" \
            "python-dotenv==1.0.1" \
            "dill==0.3.9"

        echo "--> Committing container to create image '$BASE_IMAGE_NAME'..." >&2
        docker commit "$cid" "$BASE_IMAGE_NAME"

        echo "--> Cleaning up builder container..." >&2
        docker stop "$cid" >/dev/null
        docker rm "$cid" >/dev/null

        echo "--> Custom base image created successfully." >&2
    else
        echo "--> Custom base image '$BASE_IMAGE_NAME' already exists. Skipping creation." >&2
    fi
}

perform_maintenance() {
    echo "--- Performing Docker maintenance/cleanup ---" >&2
    # Prune stopped containers and unused networks to free resources
    docker container prune -f >/dev/null 2>&1
    docker network prune -f >/dev/null 2>&1
    echo "--- Maintenance complete ---" >&2
}

# ==============================================================================

test_system() {
    local TYPE=$1
    local BENCHMARK=$2
    local ITERATION=$3

    local SYSTEM_NAME_BASE="${TYPE}_${BENCHMARK}${ITERATION}_gpt"
    local SYSTEM_MODULE_PATH="generated_systems.${SYSTEM_NAME_BASE}"

    echo "-----------------------------------------"
    echo "Testing System: $SYSTEM_NAME_BASE (in $PWD)"
    echo "  Benchmark: $BENCHMARK, Iteration: $ITERATION"

    local BENCH_DIR=""
    local BENCH_NAME_FULL=""
    if [ "$BENCHMARK" == "gsm" ]; then
        BENCH_DIR="GSMHard"; BENCH_NAME_FULL="gsmhard";
    elif [ "$BENCHMARK" == "mmlu" ]; then
        BENCH_DIR="MMLUPro"; BENCH_NAME_FULL="mmlupro";
    elif [ "$BENCHMARK" == "fever" ]; then
        BENCH_DIR="FEVER"; BENCH_NAME_FULL="fever";
    else
        echo "  ERROR: Unknown benchmark value: $BENCHMARK"; return 1;
    fi

    local MAIN_BENCH_SCRIPT="benchmark/${BENCH_DIR}/main_${BENCH_NAME_FULL}_bench.py"
    local SYSTEM_FILE_CHECK="${SYSTEM_MODULE_PATH//.//}.py"

    if [ ! -f "$SYSTEM_FILE_CHECK" ]; then
        echo "  ERROR: System file not found: $SYSTEM_FILE_CHECK (in $PWD)"; return 1;
    fi
    echo "  Found system file: $SYSTEM_FILE_CHECK"

    local image_to_use="$BASE_IMAGE_NAME"
    local temp_image_name=""
    local metrics_file="generated_systems/metrics/${SYSTEM_NAME_BASE}.json"

    # Check for custom dependencies in the metrics file
    if [ -f "$metrics_file" ]; then
        local dependencies
        dependencies=$(jq -r '.installed_packages' "$metrics_file")

        if [ -n "$dependencies" ] && [ "$dependencies" != "null" ]; then
            echo "  Found dependencies via 'installed_packages': $dependencies"
            temp_image_name="adas-temp-image-${SYSTEM_NAME_BASE}-${UNIQUE_ID}"

            echo "  --> Creating temporary image '$temp_image_name' with custom dependencies..."
            
            # Start a builder based on our base image
            local builder_cid
            builder_cid=$(docker run -d --name "adas-builder-${SYSTEM_NAME_BASE}-${UNIQUE_ID}" "$BASE_IMAGE_NAME" sleep 3600)
            
            if [ -z "$builder_cid" ]; then
                echo "  !!ERROR: Failed to start builder container for temp image."
                return 1
            fi

            read -ra reqs <<< "$dependencies"
            if ! docker exec "$builder_cid" pip install "${reqs[@]}"; then
                echo "  !!ERROR: pip install failed inside the builder container. Stopping build."
                docker stop "$builder_cid"; docker rm "$builder_cid"
                return 1
            fi

            docker commit "$builder_cid" "$temp_image_name"
            docker stop "$builder_cid"; docker rm "$builder_cid"

            image_to_use="$temp_image_name"
            echo "  --> Temporary image created."
        else
            echo "  INFO: No extra dependencies found for $SYSTEM_NAME_BASE."
        fi
    else
        echo "  INFO: No metrics file for $SYSTEM_NAME_BASE. Using base image."
    fi

    # Construct the command
    # Note: container="docker" explicitly tells sandbox.py to use Docker backend
    local CMD="python ${MAIN_BENCH_SCRIPT} --system=\"${SYSTEM_MODULE_PATH}\" --base-image=\"${image_to_use}\" --container=\"docker\""

    echo "  Executing: $CMD"
    local LOG_FILE="${RESULTS_DIR}/${SYSTEM_NAME_BASE}_log.txt"
    
    # Run the Python script on the HOST. The Python script spawns Docker containers.
    # We use 'timeout' command (standard on Linux/Mac) to enforce the limit
    # If using Mac without coreutils, you might need 'gtimeout' or remove the timeout logic.
    
    if command -v timeout &> /dev/null; then
        timeout 1200s bash -c "$CMD" > >(tee "$LOG_FILE") 2>&1
        BENCHMARK_EXIT_CODE=$?
    else
        # Fallback if 'timeout' command missing
        eval "$CMD" > >(tee "$LOG_FILE") 2>&1
        BENCHMARK_EXIT_CODE=$?
    fi

    echo "Exit code: $BENCHMARK_EXIT_CODE" >> "$LOG_FILE"
    
    if [ $BENCHMARK_EXIT_CODE -eq 124 ]; then
         echo "  ERROR: Benchmark execution timed out."
         echo "Exit code: 124 (TIMEOUT)" >> "$LOG_FILE"
         # Cleanup specifically if timeout occurred
         if [ -n "$temp_image_name" ]; then docker rmi "$temp_image_name" >/dev/null 2>&1; fi
         return 1
    fi

    # Cleanup temp image if it was created
    if [ -n "$temp_image_name" ]; then
        echo "  --> Cleaning up temporary image '$temp_image_name'..."
        docker rmi "$temp_image_name" >/dev/null 2>&1 || echo "  WARNING: Failed to remove temp image."
    fi

    local RESULT_FILE_CHECK="benchmark/${BENCH_DIR}/results/benchmark_results_${SYSTEM_MODULE_PATH}.json"
    local RESULT_FILE_COPY_DEST="${RESULTS_DIR}/${SYSTEM_NAME_BASE}_results.json"
    
    if [ -f "$RESULT_FILE_CHECK" ]; then
        cp "$RESULT_FILE_CHECK" "$RESULT_FILE_COPY_DEST"
        echo "  ✓ Results file found and copied."
        return 0
    else
        echo "  ✗ Results file not found at expected location: $RESULT_FILE_CHECK"
        return 1
    fi
}

# ==============================================================================

# --- Main Execution Logic ---

create_base_image

parse_combinations
echo "Expanded ${#COMBINATIONS[@]} combinations into ${#EXPANDED_COMBINATIONS[@]} specific test cases"

SUMMARY_FILE="${RESULTS_DIR}/test_summary.txt"
echo "Local Benchmark Test Summary ($(date))" > "$SUMMARY_FILE"
echo "Job ID: ${UNIQUE_ID}" >> "$SUMMARY_FILE"
if [ ${#COMBINATIONS[@]} -gt 0 ]; then
    echo "Testing specific combinations:" >> "$SUMMARY_FILE"
    for combo in "${COMBINATIONS[@]}"; do
        echo "  - $combo" >> "$SUMMARY_FILE"
    done
fi
echo "=========================================" >> "$SUMMARY_FILE"

declare -A test_statuses
initial_dir=$PWD

declare -A type_benchmark_map
for combo in "${EXPANDED_COMBINATIONS[@]}"; do
    read -r type benchmark iter <<< "$combo"
    key="${type}_${benchmark}"
    if [[ -z "${type_benchmark_map[$key]}" ]]; then
        type_benchmark_map[$key]="$iter"
    else
        type_benchmark_map[$key]="${type_benchmark_map[$key]} $iter"
    fi
done

for key in "${!type_benchmark_map[@]}"; do
    benchmark=${key##*_}
    type=${key%_"$benchmark"}
    read -ra iterations <<< "${type_benchmark_map[$key]}"
    ADAS_DIR="ADAS_$type"

    if [ ! -d "$initial_dir/$ADAS_DIR" ]; then
        echo "!!! ERROR: Directory for type '$type' not found: $initial_dir/$ADAS_DIR. Skipping." | tee -a "$SUMMARY_FILE"
        for iter in "${iterations[@]}"; do
            test_statuses["${type}_${benchmark}${iter}_gpt"]="FAILED_DIR_NOT_FOUND"
        done
        continue
    fi

    pushd "$initial_dir/$ADAS_DIR" > /dev/null || exit
    echo ""
    echo ">>> Testing Approach: ${type}, Benchmark: ${benchmark} (in $PWD) <<<" | tee -a "$SUMMARY_FILE"

    cleanup_counter=0
    for iter in "${iterations[@]}"; do
        cleanup_counter=$((cleanup_counter + 1))
        if [ $cleanup_counter -gt $DOCKER_CLEANUP_FREQUENCY ]; then
            perform_maintenance
            cleanup_counter=1
        fi

        SYSTEM_NAME="${type}_${benchmark}${iter}_gpt"
        echo -n "  - ${SYSTEM_NAME}: " | tee -a "$SUMMARY_FILE"

        test_system "$type" "$benchmark" "$iter"
        test_exit_code=$?

        if [ $test_exit_code -eq 0 ]; then
            echo "WORKING" >> "$SUMMARY_FILE"
            test_statuses["$SYSTEM_NAME"]="WORKING"
        else
            echo "FAILED" >> "$SUMMARY_FILE"
            test_statuses["$SYSTEM_NAME"]="FAILED"
        fi
        sleep 5
    done
    popd > /dev/null || exit
done

echo "Generating CSV summary..."
CSV_FILE="${RESULTS_DIR}/results_summary.csv"
echo "approach,benchmark,iteration,status" > "$CSV_FILE"

for combo in "${EXPANDED_COMBINATIONS[@]}"; do
    read -r type benchmark iter <<< "$combo"
    SYSTEM_NAME="${type}_${benchmark}${iter}_gpt"
    STATUS=${test_statuses["$SYSTEM_NAME"]:-"FAILED_UNKNOWN"}
    echo "${type},${benchmark},${iter},${STATUS}" >> "$CSV_FILE"
done

echo "CSV summary generated: $CSV_FILE"

echo "-----------------------------------------"
echo "Testing completed. Results and summary in ${RESULTS_DIR}"
echo "Final Statuses:"
cat "$SUMMARY_FILE"
echo "-----------------------------------------"