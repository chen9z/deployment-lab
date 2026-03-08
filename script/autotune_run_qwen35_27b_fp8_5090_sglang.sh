#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="$ROOT_DIR/script/run_qwen35_27b_fp8_5090_sglang.sh"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT" >&2
  exit 1
fi

if [[ -z "$UV_BIN" ]]; then
  echo "uv not found in PATH." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python binary: $PYTHON_BIN" >&2
  exit 1
fi

RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/script/sglang_autotune_results_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR"

PORT="${PORT:-8003}"
HOST="${HOST:-127.0.0.1}"
MODEL_PATH="${MODEL_PATH:-/home/looper/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B-FP8}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-FP8}"

CONTEXT_CANDIDATES=(${CONTEXT_CANDIDATES:-65536 49152 32768})
MEM_FRACTION_CANDIDATES=(${MEM_FRACTION_CANDIDATES:-0.88 0.90 0.92})
KV_CACHE_DTYPE_CANDIDATES=(${KV_CACHE_DTYPE_CANDIDATES:-auto fp8_e4m3})
MAX_RUNNING_REQUESTS_CANDIDATES=(${MAX_RUNNING_REQUESTS_CANDIDATES:-1 2})
CHUNKED_PREFILL_SIZE_CANDIDATES=(${CHUNKED_PREFILL_SIZE_CANDIDATES:-2048 4096})

BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-16}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-4096}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-256}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-2}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_TIMEOUT_SEC="${BENCH_TIMEOUT_SEC:-420}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-420}"

SERVER_PID=""

cleanup_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""

  for pid in $(pgrep -f "$RUN_SCRIPT" || true); do
    kill -KILL "$pid" 2>/dev/null || true
  done
  for pid in $(pgrep -f "sglang serve --model-path $MODEL_PATH" || true); do
    kill -KILL "$pid" 2>/dev/null || true
  done
}

trap cleanup_server EXIT

wait_for_server() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
  while (( SECONDS < deadline )); do
    if curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://$HOST:$PORT/v1/models" | grep -q '^200$'; then
      return 0
    fi
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
  return 1
}

start_server() {
  local log_file="$1"
  shift

  cleanup_server
  env PORT="$PORT" HOST="0.0.0.0" MODEL_PATH="$MODEL_PATH" SERVED_MODEL_NAME="$MODEL_NAME" "$@" \
    "$RUN_SCRIPT" >"$log_file" 2>&1 &
  SERVER_PID=$!
}

run_bench() {
  local out_json="$1"
  local out_log="$2"

  timeout "${BENCH_TIMEOUT_SEC}s" "$UV_BIN" run --python "$PYTHON_BIN" vllm bench serve \
    --backend openai \
    --base-url "http://$HOST:$PORT" \
    --model "$MODEL_NAME" \
    --tokenizer "$MODEL_PATH" \
    --dataset-name random \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --random-input-len "$BENCH_INPUT_LEN" \
    --random-output-len "$BENCH_OUTPUT_LEN" \
    --max-concurrency "$BENCH_MAX_CONCURRENCY" \
    --request-rate "$BENCH_REQUEST_RATE" \
    --save-result \
    --result-filename "$out_json" \
    >"$out_log" 2>&1
}

echo "Autotune results: $RESULT_DIR"

BEST_CONTEXT=""
WORKING_CONFIGS=()

for context in "${CONTEXT_CANDIDATES[@]}"; do
  echo
  echo "== Probing context_length=$context =="
  for mem in "${MEM_FRACTION_CANDIDATES[@]}"; do
    for kv in "${KV_CACHE_DTYPE_CANDIDATES[@]}"; do
      tag="probe_ctx${context}_mem${mem}_kv${kv}"
      log_file="$RESULT_DIR/${tag}.log"

      echo "Trying ctx=$context mem=$mem kv=$kv"
      if [[ "$kv" == "auto" ]]; then
        start_server "$log_file" \
          CONTEXT_LENGTH="$context" \
          MEM_FRACTION_STATIC="$mem" \
          CHUNKED_PREFILL_SIZE=2048 \
          MAX_RUNNING_REQUESTS=1 \
          MAX_TOTAL_TOKENS=
      else
        start_server "$log_file" \
          CONTEXT_LENGTH="$context" \
          MEM_FRACTION_STATIC="$mem" \
          CHUNKED_PREFILL_SIZE=2048 \
          MAX_RUNNING_REQUESTS=1 \
          MAX_TOTAL_TOKENS= \
          KV_CACHE_DTYPE="$kv"
      fi

      if wait_for_server; then
        echo "startup=ok ctx=$context mem=$mem kv=$kv"
        BEST_CONTEXT="$context"
        WORKING_CONFIGS+=("${mem}:${kv}")
        cleanup_server
      else
        echo "startup=fail ctx=$context mem=$mem kv=$kv"
        cleanup_server
      fi
    done
  done

  if [[ -n "$BEST_CONTEXT" ]]; then
    break
  fi
done

if [[ -z "$BEST_CONTEXT" ]]; then
  echo "No working startup config found." >&2
  exit 1
fi

printf '%s\n' "${WORKING_CONFIGS[@]}" >"$RESULT_DIR/working_configs.txt"

for cfg in "${WORKING_CONFIGS[@]}"; do
  IFS=: read -r mem kv <<<"$cfg"
  for max_running in "${MAX_RUNNING_REQUESTS_CANDIDATES[@]}"; do
    for chunked in "${CHUNKED_PREFILL_SIZE_CANDIDATES[@]}"; do
      tag="bench_ctx${BEST_CONTEXT}_mem${mem}_kv${kv}_mr${max_running}_chunk${chunked}"
      log_file="$RESULT_DIR/${tag}.server.log"
      bench_json="$RESULT_DIR/${tag}.json"
      bench_log="$RESULT_DIR/${tag}.bench.log"

      echo
      echo "== Benchmarking $tag =="
      if [[ "$kv" == "auto" ]]; then
        start_server "$log_file" \
          CONTEXT_LENGTH="$BEST_CONTEXT" \
          MEM_FRACTION_STATIC="$mem" \
          CHUNKED_PREFILL_SIZE="$chunked" \
          MAX_RUNNING_REQUESTS="$max_running"
      else
        start_server "$log_file" \
          CONTEXT_LENGTH="$BEST_CONTEXT" \
          MEM_FRACTION_STATIC="$mem" \
          CHUNKED_PREFILL_SIZE="$chunked" \
          MAX_RUNNING_REQUESTS="$max_running" \
          KV_CACHE_DTYPE="$kv"
      fi

      if ! wait_for_server; then
        echo "startup=fail $tag"
        cleanup_server
        continue
      fi

      if ! run_bench "$bench_json" "$bench_log"; then
        echo "bench=fail $tag"
      fi

      cleanup_server
    done
  done
done

python3 - "$RESULT_DIR" <<'PY'
import json
import os
import re
import sys

result_dir = sys.argv[1]
rows = []
pat = re.compile(
    r"bench_ctx(?P<context>\d+)_mem(?P<mem>[0-9.]+)_kv(?P<kv>[^_]+)_mr(?P<mr>\d+)_chunk(?P<chunk>\d+)\.json$"
)

for name in sorted(os.listdir(result_dir)):
    if not name.endswith(".json"):
        continue
    m = pat.match(name)
    if not m:
        continue
    with open(os.path.join(result_dir, name), "r", encoding="utf-8") as f:
        data = json.load(f)
    rows.append(
        {
            "file": name,
            "context": int(m.group("context")),
            "mem_fraction_static": float(m.group("mem")),
            "kv_cache_dtype": m.group("kv"),
            "max_running_requests": int(m.group("mr")),
            "chunked_prefill_size": int(m.group("chunk")),
            "output_throughput": float(data.get("output_throughput", 0.0)),
            "mean_ttft_ms": float(data.get("mean_ttft_ms", 0.0)),
            "mean_tpot_ms": float(data.get("mean_tpot_ms", 0.0)),
            "failed": int(data.get("failed", 0)),
        }
    )

if not rows:
    raise SystemExit("No successful benchmark results found")

rows.sort(key=lambda row: (-row["output_throughput"], row["mean_ttft_ms"]))
summary_path = os.path.join(result_dir, "summary.tsv")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(
        "rank\tcontext\tmem_fraction_static\tkv_cache_dtype\tmax_running_requests\tchunked_prefill_size\toutput_throughput\tmean_ttft_ms\tmean_tpot_ms\tfailed\tfile\n"
    )
    for idx, row in enumerate(rows, start=1):
        f.write(
            f"{idx}\t{row['context']}\t{row['mem_fraction_static']}\t{row['kv_cache_dtype']}\t"
            f"{row['max_running_requests']}\t{row['chunked_prefill_size']}\t"
            f"{row['output_throughput']:.4f}\t{row['mean_ttft_ms']:.2f}\t"
            f"{row['mean_tpot_ms']:.2f}\t{row['failed']}\t{row['file']}\n"
        )

print(summary_path)
PY
