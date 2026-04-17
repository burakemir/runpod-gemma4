#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-unsloth/gemma-4-31B-it-GGUF}"
MODEL_FILE="${MODEL_FILE:-gemma-4-31B-it-Q8_0.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
CTX_SIZE="${CTX_SIZE:-8192}"
LLAMA_PORT="${LLAMA_PORT:-8080}"

# Use the persistent RunPod volume for model caching when available,
# so the 33 GB model doesn't re-download on every cold start.
if [ -d "/runpod-volume" ] && [ -w "/runpod-volume" ]; then
    export LLAMA_CACHE="/runpod-volume/llama-cache"
    echo "Using persistent cache: $LLAMA_CACHE"
fi
mkdir -p "$LLAMA_CACHE"

# Check whether RunPod's model caching already placed the GGUF on disk.
# RunPod's cache follows HF layout:
#   /runpod-volume/huggingface-cache/hub/models--<org>--<name>/snapshots/<hash>/<file>
# The org/name in the path uses whatever casing the RunPod Model field has
# (typically lowercase), which may differ from MODEL_REPO.  We try both
# the original casing and lowercase.
resolve_snapshot() {
    local hf_cache="/runpod-volume/huggingface-cache/hub"
    local repo="$1"
    local org name model_root snapshots_dir hash candidate
    org="${repo%%/*}"
    name="${repo#*/}"
    model_root="${hf_cache}/models--${org}--${name}"
    snapshots_dir="${model_root}/snapshots"
    # Try refs/main first
    if [ -f "${model_root}/refs/main" ]; then
        hash=$(cat "${model_root}/refs/main" | tr -d '[:space:]')
        candidate="${snapshots_dir}/${hash}"
        if [ -d "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    fi
    # Fall back to first snapshot directory
    if [ -d "$snapshots_dir" ]; then
        candidate=$(ls -1d "${snapshots_dir}"/*/ 2>/dev/null | head -1)
        if [ -n "$candidate" ]; then
            echo "${candidate%/}"
            return 0
        fi
    fi
    return 1
}

PRECACHED=""
# Try original casing from MODEL_REPO
SNAPSHOT_DIR=$(resolve_snapshot "$MODEL_REPO" 2>/dev/null) || true
# Try lowercase if that didn't work
if [ -z "$SNAPSHOT_DIR" ]; then
    REPO_LOWER=$(echo "$MODEL_REPO" | tr '[:upper:]' '[:lower:]')
    SNAPSHOT_DIR=$(resolve_snapshot "$REPO_LOWER" 2>/dev/null) || true
fi
# Look for the model file in the snapshot
if [ -n "$SNAPSHOT_DIR" ] && [ -f "${SNAPSHOT_DIR}/${MODEL_FILE}" ]; then
    PRECACHED="${SNAPSHOT_DIR}/${MODEL_FILE}"
fi

if [ -n "$PRECACHED" ]; then
    echo "Found RunPod pre-cached model: $PRECACHED"
    MODEL_ARGS=(--model "$PRECACHED")
else
    echo "No pre-cached model found, will download via --hf-repo"
    MODEL_ARGS=(--hf-repo "$MODEL_REPO" --hf-file "$MODEL_FILE")
fi

echo "Starting llama-server: ${MODEL_ARGS[*]}"
echo "  GPU layers: $N_GPU_LAYERS  CTX: $CTX_SIZE  Port: $LLAMA_PORT  Cache: $LLAMA_CACHE"

/app/llama-server \
    "${MODEL_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "$LLAMA_PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --flash-attn on \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    &

SERVER_PID=$!

echo "Waiting for llama-server (pid $SERVER_PID) ..."
READY=0
for _ in $(seq 1 360); do          # up to 12 min for download + load
    if curl -sf "http://localhost:${LLAMA_PORT}/health" >/dev/null 2>&1; then
        echo "llama-server is ready"
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: llama-server exited unexpectedly"
        exit 1
    fi
    sleep 2
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: llama-server did not become healthy within timeout"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

# ---- start RunPod handler ----
exec python3 -u /handler.py
