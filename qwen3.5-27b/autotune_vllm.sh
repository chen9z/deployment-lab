#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Missing virtualenv activate script: $VENV_ACTIVATE" >&2
  exit 1
fi

MODEL_REPO="Kbenkhaled/Qwen3.5-27B-NVFP4"
SERVED_MODEL="Qwen3.5-27B"

# Benchmark workload (adjust if needed)
NUM_PROMPTS=80
RANDOM_INPUT_LEN=1024
RANDOM_OUTPUT_LEN=256
MAX_CONCURRENCY=8
REQUEST_RATE="inf"
BENCH_TIMEOUT_SEC=900

GPU_MEMORY_UTILIZATION_CANDIDATES=("0.92" "0.94")
MAX_NUM_SEQS_CANDIDATES=("3" "4")
MAX_NUM_BATCHED_TOKENS="4096"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$SCRIPT_DIR/bench_results_$RUN_ID"
mkdir -p "$RESULT_DIR"

write_compose() {
  local gpu_mem="$1"
  local max_num_seqs="$2"
  local max_num_batched_tokens="$3"

  cat > "$COMPOSE_FILE" <<EOF
services:
  vllm:
    image: vllm-openai:cu130-nightly-tfmain
    container_name: vllm-qwen35-27b
    ports:
      - "8001:8001"
    volumes:
      - \${HOME}/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HF_HOME=/root/.cache/huggingface
    ipc: host
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    command: >
      ${MODEL_REPO}
      --tensor-parallel-size 1
      --max-model-len 262144
      --kv-cache-dtype fp8
      --gpu-memory-utilization ${gpu_mem}
      --max-num-seqs ${max_num_seqs}
      --max-num-batched-tokens ${max_num_batched_tokens}
      --enable-prefix-caching
      --attention-backend FLASHINFER
      --served-model-name ${SERVED_MODEL}
      --host 0.0.0.0 --port 8001
      --reasoning-parser qwen3
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --limit-mm-per-prompt.video 0
      --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
      --async-scheduling
      --disable-log-requests
      --generation-config auto
      --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
EOF
}

restart_and_wait_ready() {
  (
    cd "$SCRIPT_DIR"
    docker compose up -d --force-recreate
  )

  local max_wait_sec=900
  local step=5
  local waited=0
  while (( waited < max_wait_sec )); do
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8001/v1/models || true)"
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    sleep "$step"
    waited=$((waited + step))
  done

  echo "Timed out waiting for vLLM readiness after ${max_wait_sec}s" >&2
  return 1
}

run_bench() {
  local out_json="$1"
  local out_log="$2"

  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
  timeout "${BENCH_TIMEOUT_SEC}s" vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8001 \
    --model "$SERVED_MODEL" \
    --tokenizer "$MODEL_REPO" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$RANDOM_INPUT_LEN" \
    --random-output-len "$RANDOM_OUTPUT_LEN" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --request-rate "$REQUEST_RATE" \
    --save-result \
    --result-filename "$out_json" \
    > "$out_log" 2>&1
}

echo "Saving benchmark artifacts to: $RESULT_DIR"
cp "$COMPOSE_FILE" "$RESULT_DIR/docker-compose.original.yml"

for gpu_mem in "${GPU_MEMORY_UTILIZATION_CANDIDATES[@]}"; do
  for max_num_seqs in "${MAX_NUM_SEQS_CANDIDATES[@]}"; do
    tag="gm${gpu_mem}_seq${max_num_seqs}_bt${MAX_NUM_BATCHED_TOKENS}"
    json_file="$RESULT_DIR/${tag}.json"
    log_file="$RESULT_DIR/${tag}.log"

    echo
    echo "=== Testing ${tag} ==="
    write_compose "$gpu_mem" "$max_num_seqs" "$MAX_NUM_BATCHED_TOKENS"
    restart_and_wait_ready
    if ! run_bench "$json_file" "$log_file"; then
      echo "Benchmark failed or timed out for ${tag}, skipping." | tee -a "$log_file"
    fi
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
    path = os.path.join(result_dir, name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = re.match(r"gm([0-9.]+)_seq([0-9]+)_bt([0-9]+)\.json$", name)
    if not m:
        continue
    gm, seq, bt = m.groups()
    rows.append({
        "file": name,
        "gpu_memory_utilization": float(gm),
        "max_num_seqs": int(seq),
        "max_num_batched_tokens": int(bt),
        "output_throughput": float(data.get("output_throughput", 0.0)),
        "mean_ttft_ms": float(data.get("mean_ttft_ms", 0.0)),
        "p99_ttft_ms": float(data.get("p99_ttft_ms", 0.0)),
        "mean_tpot_ms": float(data.get("mean_tpot_ms", 0.0)),
        "failed": int(data.get("failed", 0)),
    })

if not rows:
    raise SystemExit("No benchmark result JSON found")

# Prefer higher output throughput; tie-break with lower mean TTFT.
rows.sort(key=lambda r: (-r["output_throughput"], r["mean_ttft_ms"]))
best = rows[0]

summary_path = os.path.join(result_dir, "summary.tsv")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(
        "rank\tgpu_memory_utilization\tmax_num_seqs\tmax_num_batched_tokens\t"
        "output_throughput\tmean_ttft_ms\tp99_ttft_ms\tmean_tpot_ms\tfailed\tfile\n"
    )
    for i, r in enumerate(rows, start=1):
        f.write(
            f"{i}\t{r['gpu_memory_utilization']:.2f}\t{r['max_num_seqs']}\t"
            f"{r['max_num_batched_tokens']}\t{r['output_throughput']:.4f}\t"
            f"{r['mean_ttft_ms']:.2f}\t{r['p99_ttft_ms']:.2f}\t{r['mean_tpot_ms']:.2f}\t"
            f"{r['failed']}\t{r['file']}\n"
        )

best_path = os.path.join(result_dir, "best.env")
with open(best_path, "w", encoding="utf-8") as f:
    f.write(f"GPU_MEMORY_UTILIZATION={best['gpu_memory_utilization']:.2f}\n")
    f.write(f"MAX_NUM_SEQS={best['max_num_seqs']}\n")
    f.write(f"MAX_NUM_BATCHED_TOKENS={best['max_num_batched_tokens']}\n")
    f.write(f"OUTPUT_THROUGHPUT={best['output_throughput']:.4f}\n")
    f.write(f"MEAN_TTFT_MS={best['mean_ttft_ms']:.2f}\n")
    f.write(f"P99_TTFT_MS={best['p99_ttft_ms']:.2f}\n")
    f.write(f"MEAN_TPOT_MS={best['mean_tpot_ms']:.2f}\n")
    f.write(f"RESULT_FILE={best['file']}\n")

print(f"BEST_GPU_MEMORY_UTILIZATION={best['gpu_memory_utilization']:.2f}")
print(f"BEST_MAX_NUM_SEQS={best['max_num_seqs']}")
print(f"BEST_MAX_NUM_BATCHED_TOKENS={best['max_num_batched_tokens']}")
print(f"BEST_OUTPUT_THROUGHPUT={best['output_throughput']:.4f}")
print(f"BEST_MEAN_TTFT_MS={best['mean_ttft_ms']:.2f}")
print(f"SUMMARY_TSV={summary_path}")
print(f"BEST_ENV={best_path}")
PY

# shellcheck disable=SC1090
source "$RESULT_DIR/best.env"

echo
echo "=== Applying best config to docker-compose.yml ==="
echo "gpu-memory-utilization=${GPU_MEMORY_UTILIZATION}, max-num-seqs=${MAX_NUM_SEQS}, max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS}"
write_compose "$GPU_MEMORY_UTILIZATION" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS"
restart_and_wait_ready

echo
echo "Done. Best config is active."
echo "Results: $RESULT_DIR"
echo "Summary: $RESULT_DIR/summary.tsv"
