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

MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/Lorbus/Qwen3.6-27B-int4-AutoRound}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-27B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
DTYPE="${DTYPE:-half}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
LIMIT_MM_PER_PROMPT_VIDEO="${LIMIT_MM_PER_PROMPT_VIDEO:-0}"
DEFAULT_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3}'
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-$DEFAULT_SPECULATIVE_CONFIG}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-none}"
SPECULATIVE_ARGS=()
PREFIX_CACHING_ARGS=()
KV_CACHE_DTYPE_ARGS=()
CUDAGRAPH_ARGS=()

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  PREFIX_CACHING_ARGS=(--enable-prefix-caching)
fi

if [[ -n "$KV_CACHE_DTYPE" ]]; then
  KV_CACHE_DTYPE_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

if [[ -n "$DTYPE" ]]; then
  DTYPE_ARGS=(--dtype "$DTYPE")
else
  DTYPE_ARGS=()
fi

if [[ -n "$CUDAGRAPH_MODE" ]]; then
  CUDAGRAPH_ARGS=(--compilation-config.cudagraph_mode "$CUDAGRAPH_MODE")
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
  "${DTYPE_ARGS[@]}" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  "${KV_CACHE_DTYPE_ARGS[@]}" \
  --attention-backend FLASHINFER \
  --limit-mm-per-prompt.video "$LIMIT_MM_PER_PROMPT_VIDEO" \
  --reasoning-parser "$REASONING_PARSER" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  "${CUDAGRAPH_ARGS[@]}" \
  "${PREFIX_CACHING_ARGS[@]}" \
  "${SPECULATIVE_ARGS[@]}" \
  --disable-log-requests \
  "$@"
