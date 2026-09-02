#!/bin/bash
# Common helper functions for SLURM job wrappers

load_environment() {
    # shellcheck disable=SC1091
    source "$HOME/.bashrc"
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate py311 || echo "Warning: Conda env 'py311' not found, using current python."
    fi
}

setup_podman_service() {
    # Default SLURM runs to Podman unless Docker is explicitly requested
    if [[ "$*" == *"--container docker"* ]] || [[ "$*" == *"--container=docker"* ]]; then
        return 0
    fi

    UNIQUE_ID=${SLURM_JOB_ID:-$$}
    unset XDG_RUNTIME_DIR
    export XDG_RUNTIME_DIR="/tmp/${USER}/podman-run-${UNIQUE_ID}"
    mkdir -p "$XDG_RUNTIME_DIR/podman"
    PODMAN_SOCKET="unix://$XDG_RUNTIME_DIR/podman/podman.sock"
    export ADAS_PODMAN_SOCKET=$PODMAN_SOCKET

    # shellcheck disable=SC2329
    cleanup_podman() {
        if [ -n "${PODMAN_PID:-}" ]; then
            kill "$PODMAN_PID" 2>/dev/null || true
            wait "$PODMAN_PID" 2>/dev/null || true
        fi
        podman rm -af --ignore >/dev/null 2>&1 || true
        rm -rf "$XDG_RUNTIME_DIR"
    }
    trap cleanup_podman EXIT INT TERM

    podman system service --time=0 "$PODMAN_SOCKET" &
    PODMAN_PID=$!

    # Readiness wait loop
    for _ in {1..30}; do
        if [ -S "$XDG_RUNTIME_DIR/podman/podman.sock" ]; then
            break
        fi
        sleep 1
    done

    if [ ! -S "$XDG_RUNTIME_DIR/podman/podman.sock" ]; then
        echo "ERROR: Podman socket did not become ready: $PODMAN_SOCKET" >&2
        return 1
    fi
}
