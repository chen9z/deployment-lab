#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/jina-embeddings-v5-text-small}"
MODEL_NAME="${MODEL_NAME:-jina-embeddings-v5-text-small}"
BACKEND_PORT="${BACKEND_PORT:-8017}"
SERVER_PORT="${SERVER_PORT:-8016}"
export PYTHONPATH="/app${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  kill "${backend_pid:-}" "${gateway_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

vllm serve "${MODEL_PATH}" \
  --served-model-name jina-base \
  --runner pooling \
  --convert embed \
  --trust-remote-code \
  --hf-overrides '{"architectures":["Qwen3Model"],"model_type":"qwen3"}' \
  --pooler-config '{"pooling_type":"LAST"}' \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-128}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.26}" \
  --enable-lora \
  --max-loras 4 \
  --max-cpu-loras 4 \
  --max-lora-rank 32 \
  --lora-modules \
    retrieval="${MODEL_PATH}/adapters/retrieval" \
    text-matching="${MODEL_PATH}/adapters/text-matching" \
    classification="${MODEL_PATH}/adapters/classification" \
    clustering="${MODEL_PATH}/adapters/clustering" \
  --middleware vllm_middleware.patch_pooling_lora \
  --host 127.0.0.1 \
  --port "${BACKEND_PORT}" \
  --disable-uvicorn-access-log \
  --no-use-tqdm-on-load &
backend_pid=$!

for _ in $(seq 1 180); do
  if ! kill -0 "${backend_pid}" 2>/dev/null; then
    wait "${backend_pid}"
  fi
  if curl --fail --silent "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null

BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}" \
MODEL_NAME="${MODEL_NAME}" \
uvicorn gateway:app \
  --host 0.0.0.0 \
  --port "${SERVER_PORT}" \
  --no-access-log \
  --workers 1 &
gateway_pid=$!

wait -n "${backend_pid}" "${gateway_pid}"
