#!/bin/bash

# Change working directory to project root
cd "$(dirname "$0")/.." || exit
# SLURM CONFIGURATION HERE

# --- Configuration ---
UNIQUE_ID=${SLURM_JOB_ID:-$$}
RESULTS_DIR="$PWD/small_benchmark_results"
PODMAN_RESET_FREQUENCY=8
BASE_IMAGE_NAME="adas-base-job:${UNIQUE_ID}"

COMBINATIONS=(
"" # e.g., ablationC fever 29
)

echo "Starting Refactored Small Benchmark Test Run ($(date)) for Job ID: $UNIQUE_ID"
echo "Configuration:"
echo "  Results Directory: $RESULTS_DIR"
echo "  Podman Reset Frequency: Every $PODMAN_RESET_FREQUENCY iterations"
echo "  Using provided combinations:"
for combo in "${COMBINATIONS[@]}"; do
    echo "    - $combo"
done
echo "-----------------------------------------"

echo "Loading environment..."
source "$HOME"/.bashrc
eval "$(conda shell.bash hook)"
conda activate py311
echo "Environment loaded."

# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "!!ERROR: 'jq' command not found. Please install jq to parse JSON metrics. Aborting."
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# --- Podman and Cleanup Setup ---
unset XDG_RUNTIME_DIR
export XDG_RUNTIME_DIR="/tmp/${USER}/podman-run-${UNIQUE_ID}"

job_cleanup() {
    echo "--- Cleaning up all resources for Job $UNIQUE_ID ---"
    if [ ! -z "$PODMAN_PID" ]; then
        echo "--> Killing Podman service (PID: $PODMAN_PID)..."
        kill "$PODMAN_PID" 2>/dev/null
        wait "$PODMAN_PID" 2>/dev/null
    fi
    echo "--> Removing any leftover containers and images for this job..."
    podman rm -af --ignore
    podman image rm -f "$BASE_IMAGE_NAME" --ignore
    echo "--> Removing runtime directory: $XDG_RUNTIME_DIR"
    rm -rf "$XDG_RUNTIME_DIR"
    echo "--- Cleanup for Job $UNIQUE_ID Complete ---"
}
trap job_cleanup EXIT INT TERM

# --- Helper Functions ---
parse_combinations() {
    EXPANDED_COMBINATIONS=()
    for combo in "${COMBINATIONS[@]}"; do
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
    if ! podman image exists "$BASE_IMAGE_NAME"; then
        echo "--> Custom base image '$BASE_IMAGE_NAME' not found. Creating it..." >&2
        local cid
        local builder_image="docker.io/library/python:3.11-slim"
        podman pull "$builder_image" >/dev/null
        cid=$(podman run -d --name "adas-builder-${UNIQUE_ID}" "$builder_image" sleep 3600)

        if [ -z "$cid" ]; then
            echo "!!ERROR: Failed to start the builder container. Aborting." >&2
            exit 1
        fi

        echo "--> Installing core dependencies in builder container..." >&2
        podman exec "$cid" pip install \
            "langgraph==0.4.8" \
            "langchain_openai==0.3.32" \
            "python-dotenv==1.0.1" \
            "dill==0.3.9"

        echo "--> Committing container to create image '$BASE_IMAGE_NAME'..." >&2
        podman commit "$cid" "$BASE_IMAGE_NAME"

        echo "--> Cleaning up builder container..." >&2
        podman stop "$cid" >/dev/null
        podman rm "$cid" >/dev/null

        echo "--> Custom base image created successfully." >&2
    else
        echo "--> Custom base image '$BASE_IMAGE_NAME' already exists. Skipping creation." >&2
    fi
}

reset_podman() {
    echo "--- Performing deep podman environment reset for Job $UNIQUE_ID... ---" >&2
    if [ ! -z "${PODMAN_PID:-}" ]; then
        echo "Killing previous podman service PID: $PODMAN_PID" >&2
        kill "$PODMAN_PID" 2>/dev/null || true; wait "$PODMAN_PID" 2>/dev/null || true
    fi
    podman stop --all >/dev/null 2>&1 || true
    podman rm --all --force >/dev/null 2>&1 || true

    echo "--> Removing job-specific base image before rebuild..." >&2
    podman image rm -f "$BASE_IMAGE_NAME" --ignore >/dev/null 2>&1 || true

    mkdir -p "$XDG_RUNTIME_DIR/podman"
    PODMAN_SOCKET="unix://$XDG_RUNTIME_DIR/podman/podman.sock"
    export ADAS_PODMAN_SOCKET=$PODMAN_SOCKET
    create_base_image

    echo "Starting new podman service..." >&2
    podman system service --time=0 "$PODMAN_SOCKET" &
    PODMAN_PID=$!
    echo "Waiting for podman service (PID: $PODMAN_PID) to initialize..." >&2

    sleep 15
    if ! podman info > /dev/null 2>&1; then
        echo "ERROR: Podman service failed to start." >&2; exit 1;
    fi
    echo "--- Podman service is ready. ---" >&2
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

    if [ -f "$metrics_file" ]; then
        # Use jq to directly and reliably parse the installed_packages key from the JSON metrics file.
        local dependencies
        dependencies=$(jq -r '.installed_packages' "$metrics_file")

        # Check if dependencies string is not empty and not the literal string "null".
        if [ -n "$dependencies" ] && [ "$dependencies" != "null" ]; then
            echo "  Found dependencies via 'installed_packages': $dependencies"
            temp_image_name="adas-temp-image-${SYSTEM_NAME_BASE}-${UNIQUE_ID}"

            echo "  --> Creating temporary image '$temp_image_name' with custom dependencies..."
            local builder_cid
            builder_cid=$(podman run -d --name "adas-builder-${SYSTEM_NAME_BASE}-${UNIQUE_ID}" "$BASE_IMAGE_NAME" sleep 3600)
            if [ -z "$builder_cid" ]; then
                echo "  !!ERROR: Failed to start builder container for temp image."
                return 1
            fi

            read -ra reqs <<< "$dependencies"
            if ! podman exec "$builder_cid" pip install "${reqs[@]}"; then
                echo "  !!ERROR: pip install failed inside the builder container. Stopping build for this system."
                podman stop "$builder_cid"; podman rm "$builder_cid"
                return 1
            fi

            podman commit "$builder_cid" "$temp_image_name"
            podman stop "$builder_cid"; podman rm "$builder_cid"

            image_to_use="$temp_image_name"
            echo "  --> Temporary image created."
        else
            echo "  INFO: No dependencies found in 'installed_packages' key in metrics for $SYSTEM_NAME_BASE."
        fi
    else
        echo "  INFO: No metrics file for $SYSTEM_NAME_BASE. Using base image."
    fi

    local CMD="python ${MAIN_BENCH_SCRIPT} --system=\"${SYSTEM_MODULE_PATH}\" --base-image=\"${image_to_use}\""

    echo "  Executing: $CMD"
    local LOG_FILE="${RESULTS_DIR}/${SYSTEM_NAME_BASE}_log.txt"
    local BENCHMARK_EXIT_CODE=0

    eval "$CMD" > >(tee "$LOG_FILE") 2>&1 &
    local CMD_PID=$!

    local timeout_seconds=1200
    local waited=0
    local interval=30

    echo "  Command running with PID: $CMD_PID (timeout: ${timeout_seconds}s)"

    while kill -0 $CMD_PID 2>/dev/null; do
        sleep $interval
        waited=$((waited + interval))
        if [ $waited -ge $timeout_seconds ]; then
            echo "  ERROR: Benchmark execution timed out after ${timeout_seconds} seconds" | tee -a "$LOG_FILE"
            echo "  Killing process $CMD_PID..." | tee -a "$LOG_FILE"
            kill -9 $CMD_PID 2>/dev/null || true
            wait $CMD_PID 2>/dev/null || true
            local BENCHMARK_EXIT_CODE=124
            echo "" >> "$LOG_FILE"
            echo "Exit code: $BENCHMARK_EXIT_CODE (TIMEOUT)" >> "$LOG_FILE"
            echo "  Benchmark execution TIMED OUT after ${timeout_seconds} seconds"
            return 1
        fi
        echo "  Still running... (${waited}/${timeout_seconds}s)"
    done

    wait $CMD_PID
    BENCHMARK_EXIT_CODE=$?
    echo "Exit code: $BENCHMARK_EXIT_CODE" >> "$LOG_FILE"
    echo "  Benchmark execution finished with exit code: $BENCHMARK_EXIT_CODE"

    if [ -n "$temp_image_name" ]; then
        echo "  --> Cleaning up temporary image '$temp_image_name'..."
        podman rmi "$temp_image_name" || echo "  WARNING: Failed to remove temp image."
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
reset_podman

parse_combinations
echo "Expanded ${#COMBINATIONS[@]} combinations into ${#EXPANDED_COMBINATIONS[@]} specific test cases"

SUMMARY_FILE="${RESULTS_DIR}/test_summary.txt"
echo "Refactored Small Benchmark Test Summary ($(date))" > "$SUMMARY_FILE"
echo "Job ID: ${UNIQUE_ID}" >> "$SUMMARY_FILE"
echo "Podman Reset Frequency: Every ${PODMAN_RESET_FREQUENCY} iterations" >> "$SUMMARY_FILE"
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

    podman_reset_counter=0
    for iter in "${iterations[@]}"; do
        podman_reset_counter=$((podman_reset_counter + 1))
        if [ $podman_reset_counter -gt $PODMAN_RESET_FREQUENCY ]; then
            echo "  === Performing scheduled Podman reset after $PODMAN_RESET_FREQUENCY iterations ==="
            reset_podman
            podman_reset_counter=1
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
        sleep 10
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
echo "CSV Summary:"
cat "$CSV_FILE"
echo "-----------------------------------------"