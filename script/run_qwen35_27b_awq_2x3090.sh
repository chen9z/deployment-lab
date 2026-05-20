#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-$ROOT_DIR/.venv/bin/vllm}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Lorbus/Qwen3.6-27B-int4-AutoRound}"
PORT="${PORT:-8010}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b-autoround}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
SERVICE_DIR="${SERVICE_DIR:-$ROOT_DIR/qwen3.5-27b/dual_3090_qwen36_service}"
PID_FILE="$SERVICE_DIR/server.pid"
LOG_FILE="$SERVICE_DIR/server.log"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

if [[ -z "${SPECULATIVE_CONFIG+x}" ]]; then
  SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3}'
fi

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM CLI not found: $VLLM_BIN" >&2
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$SERVICE_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Server already running with PID $old_pid"
    echo "Log file: $LOG_FILE"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port $PORT is already in use. Stop the existing server first." >&2
    exit 1
  fi
fi

echo "Model dir:  $MODEL_DIR"
echo "Port:       $PORT"
echo "Model name: $SERVED_MODEL_NAME"
echo "GPUs:       CUDA_VISIBLE_DEVICES=1,2"
echo "TP/PP:      ${TENSOR_PARALLEL_SIZE}/${PIPELINE_PARALLEL_SIZE}"
echo "Max len:    $MAX_MODEL_LEN"
echo "Log file:   $LOG_FILE"
if [[ -n "${HOST_IP:-}" ]]; then
  echo "Reachable:  http://${HOST_IP}:${PORT}"
fi

custom_all_reduce_flag=()
if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
  custom_all_reduce_flag+=(--disable-custom-all-reduce)
fi

speculative_flag=()
if [[ "$SPECULATIVE_CONFIG" != "off" ]]; then
  speculative_flag+=(--speculative-config "$SPECULATIVE_CONFIG")
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2 \
RAY_memory_monitor_refresh_ms=0 \
NCCL_CUMEM_ENABLE=0 \
NCCL_P2P_DISABLE=1 \
VLLM_NO_USAGE_STATS=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}" \
nohup "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --quantization auto_round \
  --dtype float16 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --block-size "$BLOCK_SIZE" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype fp8_e5m2 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  "${speculative_flag[@]}" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${custom_all_reduce_flag[@]}" \
  --disable-uvicorn-access-log \
  --no-use-tqdm-on-load \
  -O3 \
  --host 0.0.0.0 --port "$PORT" \
  >"$LOG_FILE" 2>&1 &

server_pid=$!
echo "$server_pid" > "$PID_FILE"

echo
echo "Waiting for readiness ..."
for _ in $(seq 1 240); do
  if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server is ready at http://127.0.0.1:${PORT}"
    exit 0
  fi
  if [[ -n "${HOST_IP:-}" ]] && curl -fsS --max-time 5 "http://${HOST_IP}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server is ready at http://${HOST_IP}:${PORT}"
    echo "Server is ready."
    exit 0
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Server exited during startup." >&2
    tail -n 200 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Timed out waiting for server readiness." >&2
tail -n 200 "$LOG_FILE" >&2 || true
exit 1
