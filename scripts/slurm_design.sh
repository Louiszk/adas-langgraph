#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --job-name=adas_design
#SBATCH --output=adas_design_%j.out

cd "$(dirname "$0")/.." || exit 1

# shellcheck disable=SC1091
source scripts/slurm_common.sh

# 1. Load Conda/python environment first (where podman may be installed)
load_environment

# 2. Set up Podman socket service
setup_podman_service "$@" || exit 1

# Hand off to Python (defaulting to Podman container engine if not specified)
CONTAINER_ARGS=()
if [[ "$*" != *"--container"* ]]; then
    CONTAINER_ARGS=(--container podman)
fi

if [ $# -gt 0 ]; then
    python scripts/orchestrator.py --task design "${CONTAINER_ARGS[@]}" "$@"
else
    echo "Error: Design runs require an established TaskSpec (--task-spec)." >&2
    echo "Usage: sbatch scripts/slurm_design.sh --task-spec <path/to/task.json> [options]" >&2
    exit 1
fi
