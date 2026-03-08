#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="$ROOT_DIR/script/run_qwen35_27b_fp8_5090.sh"
VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"

if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Missing virtualenv: $VENV_ACTIVATE" >&2
  exit 1
fi

RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/script/autotune_results_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR"

PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
MODEL_PATH="${MODEL_PATH:-/home/looper/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B-FP8}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-FP8}"

# Phase 1: find the largest context that can start stably.
CONTEXT_CANDIDATES=(${CONTEXT_CANDIDATES:-262144 196608 131072 98304 65536})
GPU_UTIL_CANDIDATES=(${GPU_UTIL_CANDIDATES:-0.94 0.92 0.90 0.88})

# Phase 2: benchmark configs under the chosen context.
SEQ_CANDIDATES=(${SEQ_CANDIDATES:-1 2})
BATCH_CANDIDATES=(${BATCH_CANDIDATES:-2048 4096})
PREFIX_CACHING_CANDIDATES=(${PREFIX_CACHING_CANDIDATES:-0})

BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-16}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-4096}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-256}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-4}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_TIMEOUT_SEC="${BENCH_TIMEOUT_SEC:-600}"
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
  for pid in $(pgrep -f "vllm serve $MODEL_PATH" || true); do
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

extract_log_value() {
  local pattern="$1"
  local file="$2"
  python3 - "$pattern" "$file" <<'PY'
import re
import sys

pattern = re.compile(sys.argv[1])
path = sys.argv[2]
value = ""
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            value = m.group(1)
print(value)
PY
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

  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
  timeout "${BENCH_TIMEOUT_SEC}s" vllm bench serve \
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

MAX_CONTEXT=""
WORKING_GPU_UTILS=()

for context in "${CONTEXT_CANDIDATES[@]}"; do
  echo
  echo "== Probing max_model_len=$context =="
  for gpu_util in "${GPU_UTIL_CANDIDATES[@]}"; do
    tag="probe_ctx${context}_util${gpu_util}"
    log_file="$RESULT_DIR/${tag}.log"

    echo "Trying ctx=$context util=$gpu_util"
    start_server "$log_file" \
      MAX_MODEL_LEN="$context" \
      GPU_MEMORY_UTILIZATION="$gpu_util" \
      MAX_NUM_SEQS=1 \
      MAX_NUM_BATCHED_TOKENS=2048 \
      ENABLE_PREFIX_CACHING=0 \
      SPECULATIVE_CONFIG=off

    if wait_for_server; then
      echo "startup=ok ctx=$context util=$gpu_util"
      MAX_CONTEXT="$context"
      WORKING_GPU_UTILS+=("$gpu_util")
      cleanup_server
    else
      echo "startup=fail ctx=$context util=$gpu_util"
      cleanup_server
    fi
  done

  if [[ -n "$MAX_CONTEXT" ]]; then
    break
  fi
done

if [[ -z "$MAX_CONTEXT" ]]; then
  echo "No working startup config found." >&2
  exit 1
fi

echo
echo "Selected max context: $MAX_CONTEXT"
printf '%s\n' "${WORKING_GPU_UTILS[@]}" >"$RESULT_DIR/working_gpu_utils.txt"
WORKING_GPU_UTILS=("${WORKING_GPU_UTILS[@]:0:2}")

for gpu_util in "${WORKING_GPU_UTILS[@]}"; do
  for seq in "${SEQ_CANDIDATES[@]}"; do
    for batch in "${BATCH_CANDIDATES[@]}"; do
      for prefix in "${PREFIX_CACHING_CANDIDATES[@]}"; do
        tag="bench_ctx${MAX_CONTEXT}_util${gpu_util}_seq${seq}_batch${batch}_prefix${prefix}"
        log_file="$RESULT_DIR/${tag}.server.log"
        bench_json="$RESULT_DIR/${tag}.json"
        bench_log="$RESULT_DIR/${tag}.bench.log"

        echo
        echo "== Benchmarking $tag =="
        start_server "$log_file" \
          MAX_MODEL_LEN="$MAX_CONTEXT" \
          GPU_MEMORY_UTILIZATION="$gpu_util" \
          MAX_NUM_SEQS="$seq" \
          MAX_NUM_BATCHED_TOKENS="$batch" \
          ENABLE_PREFIX_CACHING="$prefix" \
          SPECULATIVE_CONFIG=off

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
done

python3 - "$RESULT_DIR" <<'PY'
import json
import os
import re
import sys

result_dir = sys.argv[1]
rows = []

for name in sorted(os.listdir(result_dir)):
    if not name.endswith(".json"):
        continue
    m = re.match(
        r"bench_ctx(?P<context>\d+)_util(?P<util>[0-9.]+)_seq(?P<seq>\d+)_batch(?P<batch>\d+)_prefix(?P<prefix>[01])\.json$",
        name,
    )
    if not m:
        continue

    path = os.path.join(result_dir, name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows.append(
        {
            "file": name,
            "context": int(m.group("context")),
            "gpu_util": float(m.group("util")),
            "seq": int(m.group("seq")),
            "batch": int(m.group("batch")),
            "prefix": int(m.group("prefix")),
            "output_throughput": float(data.get("output_throughput", 0.0)),
            "mean_ttft_ms": float(data.get("mean_ttft_ms", 0.0)),
            "p99_ttft_ms": float(data.get("p99_ttft_ms", 0.0)),
            "mean_tpot_ms": float(data.get("mean_tpot_ms", 0.0)),
            "failed": int(data.get("failed", 0)),
        }
    )

if not rows:
    raise SystemExit("No successful benchmark results found")

# Context is already fixed to the largest working value. Rank by throughput, then TTFT.
rows.sort(key=lambda row: (-row["output_throughput"], row["mean_ttft_ms"]))

summary_path = os.path.join(result_dir, "summary.tsv")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(
        "rank\tcontext\tgpu_util\tseq\tbatch\tprefix\toutput_throughput\t"
        "mean_ttft_ms\tp99_ttft_ms\tmean_tpot_ms\tfailed\tfile\n"
    )
    for idx, row in enumerate(rows, start=1):
        f.write(
            f"{idx}\t{row['context']}\t{row['gpu_util']:.2f}\t{row['seq']}\t"
            f"{row['batch']}\t{row['prefix']}\t{row['output_throughput']:.4f}\t"
            f"{row['mean_ttft_ms']:.2f}\t{row['p99_ttft_ms']:.2f}\t"
            f"{row['mean_tpot_ms']:.2f}\t{row['failed']}\t{row['file']}\n"
        )

best = rows[0]
best_path = os.path.join(result_dir, "best.env")
with open(best_path, "w", encoding="utf-8") as f:
    f.write(f"MAX_MODEL_LEN={best['context']}\n")
    f.write(f"GPU_MEMORY_UTILIZATION={best['gpu_util']:.2f}\n")
    f.write(f"MAX_NUM_SEQS={best['seq']}\n")
    f.write(f"MAX_NUM_BATCHED_TOKENS={best['batch']}\n")
    f.write(f"ENABLE_PREFIX_CACHING={best['prefix']}\n")
    f.write(f"OUTPUT_THROUGHPUT={best['output_throughput']:.4f}\n")
    f.write(f"MEAN_TTFT_MS={best['mean_ttft_ms']:.2f}\n")
    f.write(f"P99_TTFT_MS={best['p99_ttft_ms']:.2f}\n")
    f.write(f"MEAN_TPOT_MS={best['mean_tpot_ms']:.2f}\n")
    f.write(f"RESULT_FILE={best['file']}\n")

print(f"SUMMARY={summary_path}")
print(f"BEST={best_path}")
print(f"BEST_CONTEXT={best['context']}")
print(f"BEST_GPU_UTIL={best['gpu_util']:.2f}")
print(f"BEST_SEQ={best['seq']}")
print(f"BEST_BATCH={best['batch']}")
print(f"BEST_PREFIX={best['prefix']}")
print(f"BEST_OUTPUT_THROUGHPUT={best['output_throughput']:.4f}")
print(f"BEST_MEAN_TTFT_MS={best['mean_ttft_ms']:.2f}")
PY

echo
echo "Done. Results: $RESULT_DIR"
