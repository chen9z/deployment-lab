#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "$UV_BIN" ]]; then
  echo "uv not found in PATH."
  echo "Install uv first, e.g. https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN"
  echo "Set PYTHON_BIN or create .venv first."
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-/home/looper/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-27B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
LIMIT_MM_PER_PROMPT_VIDEO="${LIMIT_MM_PER_PROMPT_VIDEO:-0}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-off}"
SPECULATIVE_ARGS=()
PREFIX_CACHING_ARGS=()
KV_CACHE_DTYPE_ARGS=()

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  PREFIX_CACHING_ARGS=(--enable-prefix-caching)
fi

if [[ -n "$KV_CACHE_DTYPE" ]]; then
  KV_CACHE_DTYPE_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH"
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && [[ "$GPU_COUNT" -gt 0 ]] && [[ "$TP_SIZE" -gt "$GPU_COUNT" ]]; then
    echo "TP_SIZE=$TP_SIZE is larger than visible GPUs ($GPU_COUNT). Falling back to TP_SIZE=$GPU_COUNT."
    TP_SIZE="$GPU_COUNT"
  fi

  GPU_MEM_LINE="$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)"
  if [[ -n "$GPU_MEM_LINE" ]]; then
    GPU_MEM_TOTAL_MB="$(echo "$GPU_MEM_LINE" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
    GPU_MEM_FREE_MB="$(echo "$GPU_MEM_LINE" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
    if [[ "$GPU_MEM_TOTAL_MB" =~ ^[0-9]+$ ]] && [[ "$GPU_MEM_FREE_MB" =~ ^[0-9]+$ ]] && [[ "$GPU_MEM_TOTAL_MB" -gt 0 ]]; then
      CLAMP_NEEDED="$(awk -v u="$GPU_MEMORY_UTILIZATION" -v f="$GPU_MEM_FREE_MB" -v t="$GPU_MEM_TOTAL_MB" 'BEGIN{
        free_ratio=f/t;
        max_util=free_ratio-0.01;
        if (max_util < 0.50) max_util=0.50;
        if (u > max_util) print 1;
        else print 0;
      }')"
      if [[ "$CLAMP_NEEDED" == "1" ]]; then
        CLAMPED_UTIL="$(awk -v f="$GPU_MEM_FREE_MB" -v t="$GPU_MEM_TOTAL_MB" 'BEGIN{
          free_ratio=f/t;
          max_util=free_ratio-0.01;
          if (max_util < 0.50) max_util=0.50;
          printf "%.4f", max_util;
        }')"
        echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION is too high for current free VRAM (${GPU_MEM_FREE_MB}/${GPU_MEM_TOTAL_MB} MiB). Falling back to $CLAMPED_UTIL."
        GPU_MEMORY_UTILIZATION="$CLAMPED_UTIL"
      fi
    fi
  fi
fi

if [[ -n "$SPECULATIVE_CONFIG" ]] && [[ "${SPECULATIVE_CONFIG,,}" != "off" ]]; then
  SPECULATIVE_ARGS=(--speculative-config "$SPECULATIVE_CONFIG")
fi

exec "$UV_BIN" run --python "$PYTHON_BIN" vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  "${KV_CACHE_DTYPE_ARGS[@]}" \
  --attention-backend FLASHINFER \
  --limit-mm-per-prompt.video "$LIMIT_MM_PER_PROMPT_VIDEO" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "${PREFIX_CACHING_ARGS[@]}" \
  "${SPECULATIVE_ARGS[@]}" \
  --disable-log-requests \
  "$@"
