#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "$UV_BIN" ]]; then
  echo "uv not found in PATH." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python binary: $PYTHON_BIN" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-/home/looper/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-27B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.99}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
ENABLE_MULTIMODAL="${ENABLE_MULTIMODAL:-1}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-24}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-flashinfer}"
if [[ -z "${LIMIT_MM_DATA_PER_REQUEST+x}" ]]; then
  LIMIT_MM_DATA_PER_REQUEST='{"video": 0}'
fi

EXTRA_ARGS=()

if [[ -n "$MAX_RUNNING_REQUESTS" ]]; then
  EXTRA_ARGS+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
fi

if [[ -n "$MAX_TOTAL_TOKENS" ]]; then
  EXTRA_ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi

if [[ -n "$KV_CACHE_DTYPE" ]]; then
  EXTRA_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

if [[ "$ENABLE_MULTIMODAL" == "1" ]]; then
  EXTRA_ARGS+=(--enable-multimodal)
fi

if [[ -n "$LIMIT_MM_DATA_PER_REQUEST" ]]; then
  EXTRA_ARGS+=(--limit-mm-data-per-request "$LIMIT_MM_DATA_PER_REQUEST")
fi

if [[ "$DISABLE_RADIX_CACHE" == "1" ]]; then
  EXTRA_ARGS+=(--disable-radix-cache)
fi

if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
  EXTRA_ARGS+=(--disable-cuda-graph)
fi

exec "$UV_BIN" run --python "$PYTHON_BIN" sglang serve \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --attention-backend "$ATTENTION_BACKEND" \
  --sampling-backend "$SAMPLING_BACKEND" \
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
  "${EXTRA_ARGS[@]}" \
  "$@"
