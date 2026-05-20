#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-$ROOT_DIR/.venv/bin/vllm}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Lorbus/Qwen3.6-27B-int4-AutoRound}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8010}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-27b-autoround}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/qwen3.5-27b/dual_3090_qwen36_bench_$(date +%Y%m%d_%H%M%S)}"
NUM_PROMPTS_SINGLE="${NUM_PROMPTS_SINGLE:-24}"
NUM_PROMPTS_MULTI="${NUM_PROMPTS_MULTI:-80}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM CLI not found: $VLLM_BIN" >&2
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "Checking server at $BASE_URL ..."
if ! curl -fsS --max-time 5 "$BASE_URL/v1/models" > "$OUT_DIR/models.json"; then
  if [[ -n "${HOST_IP:-}" ]]; then
    ALT_BASE_URL="http://${HOST_IP}:8010"
    if curl -fsS --max-time 5 "$ALT_BASE_URL/v1/models" > "$OUT_DIR/models.json"; then
      BASE_URL="$ALT_BASE_URL"
      echo "Using reachable base URL: $BASE_URL"
    else
      echo "Server is not reachable at $BASE_URL or $ALT_BASE_URL" >&2
      exit 1
    fi
  else
    echo "Server is not reachable at $BASE_URL" >&2
    exit 1
  fi
fi

run_case() {
  local tag="$1"
  local concurrency="$2"
  local prompts="$3"
  local json_file="$OUT_DIR/${tag}.json"
  local log_file="$OUT_DIR/${tag}.log"

  echo
  echo "[RUN] $tag"
  CUDA_DEVICE_ORDER=PCI_BUS_ID "$VLLM_BIN" bench serve \
    --backend openai \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --served-model-name "$MODEL_NAME" \
    --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --num-prompts "$prompts" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --max-concurrency "$concurrency" \
    --request-rate "$REQUEST_RATE" \
    --save-result \
    --result-filename "$json_file" \
    2>&1 | tee "$log_file"
}

run_case "single_c1_in${INPUT_LEN}_out${OUTPUT_LEN}" 1 "$NUM_PROMPTS_SINGLE"
run_case "throughput_c8_in${INPUT_LEN}_out${OUTPUT_LEN}" 8 "$NUM_PROMPTS_MULTI"

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import json
import os
import sys

result_dir = sys.argv[1]
rows = []
for name in sorted(os.listdir(result_dir)):
    if not name.endswith(".json") or name == "models.json":
        continue
    path = os.path.join(result_dir, name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows.append(
        {
            "file": name,
            "completed": data.get("completed"),
            "failed": data.get("failed"),
            "output_throughput": data.get("output_throughput"),
            "total_token_throughput": data.get("total_token_throughput"),
            "mean_ttft_ms": data.get("mean_ttft_ms"),
            "mean_tpot_ms": data.get("mean_tpot_ms"),
        }
    )

summary_path = os.path.join(result_dir, "summary.tsv")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(
        "file\tcompleted\tfailed\toutput_throughput\ttotal_token_throughput\t"
        "mean_ttft_ms\tmean_tpot_ms\n"
    )
    for row in rows:
        f.write(
            f"{row['file']}\t{row['completed']}\t{row['failed']}\t"
            f"{row['output_throughput']}\t{row['total_token_throughput']}\t"
            f"{row['mean_ttft_ms']}\t{row['mean_tpot_ms']}\n"
        )

print(f"SUMMARY_TSV={summary_path}")
for row in rows:
    print(
        f"{row['file']}: output_throughput={row['output_throughput']}, "
        f"total_token_throughput={row['total_token_throughput']}, "
        f"mean_ttft_ms={row['mean_ttft_ms']}, mean_tpot_ms={row['mean_tpot_ms']}, "
        f"completed={row['completed']}, failed={row['failed']}"
    )
PY
