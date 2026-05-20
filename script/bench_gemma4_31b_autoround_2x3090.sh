#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-gemma4-31b-autoround-2x3090}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8003}"
MODELS_URL="${MODELS_URL:-${BASE_URL%/}/v1/models}"
CONTAINER_BASE_URL="${CONTAINER_BASE_URL:-http://127.0.0.1:8000}"
TOKENIZER_PATH_IN_CONTAINER="${TOKENIZER_PATH_IN_CONTAINER:-/models/gemma-4-31b-autoround}"
PROMPT_LEN="${PROMPT_LEN:-64}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
PROMPTS_PER_CONCURRENCY="${PROMPTS_PER_CONCURRENCY:-4}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/gemma-4-31b/gemma4_31b_autoround_2x3090_bench_$(date +%Y%m%d_%H%M%S)}"
CONTAINER_OUT_DIR="${CONTAINER_OUT_DIR:-/tmp/$(basename "$OUT_DIR")}"

DEPTH_VALUES=(${DEPTH_VALUES:-0 65536 131072 196608})
CONCURRENCY_VALUES=(${CONCURRENCY_VALUES:-1})

mkdir -p "$OUT_DIR"

if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" != "true" ]]; then
  echo "Container is not running: $CONTAINER_NAME" >&2
  exit 1
fi

echo "Checking server at $MODELS_URL ..."
curl -fsS --max-time 10 "$MODELS_URL" > "$OUT_DIR/models.json"

MODEL_NAME="${MODEL_NAME:-$(python3 - "$OUT_DIR/models.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

models = data.get("data", [])
if not models:
    raise SystemExit("No models returned by /v1/models")

print(models[0]["id"])
PY
)}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"

echo "Using model: $MODEL_NAME"
echo "Using served model name: $SERVED_MODEL_NAME"

docker exec "$CONTAINER_NAME" sh -lc "rm -rf '$CONTAINER_OUT_DIR' && mkdir -p '$CONTAINER_OUT_DIR'"

for depth in "${DEPTH_VALUES[@]}"; do
  for concurrency in "${CONCURRENCY_VALUES[@]}"; do
    if [[ -n "${NUM_PROMPTS:-}" ]]; then
      num_prompts="$NUM_PROMPTS"
    else
      num_prompts=$((concurrency * PROMPTS_PER_CONCURRENCY))
    fi

    run_name="c${concurrency}_depth${depth}_in${PROMPT_LEN}_out${OUTPUT_LEN}"
    result_json="${run_name}.json"
    result_log="$OUT_DIR/${run_name}.log"

    echo
    echo "[RUN] $run_name num_prompts=$num_prompts"

    bench_cmd=(
      docker exec
      "$CONTAINER_NAME"
      vllm
      bench
      serve
      --backend
      openai-chat
      --base-url
      "$CONTAINER_BASE_URL"
      --endpoint
      /v1/chat/completions
      --model
      "$MODEL_NAME"
      --served-model-name
      "$SERVED_MODEL_NAME"
      --tokenizer
      "$TOKENIZER_PATH_IN_CONTAINER"
      --dataset-name
      random
      --random-input-len
      "$PROMPT_LEN"
      --random-output-len
      "$OUTPUT_LEN"
      --random-prefix-len
      "$depth"
      --num-prompts
      "$num_prompts"
      --max-concurrency
      "$concurrency"
      --num-warmups
      "$NUM_WARMUPS"
      --save-result
      --result-dir
      "$CONTAINER_OUT_DIR"
      --result-filename
      "$result_json"
      --disable-tqdm
    )

    if [[ "$REQUEST_RATE" != "inf" ]]; then
      bench_cmd+=(--request-rate "$REQUEST_RATE")
    fi

    "${bench_cmd[@]}" 2>&1 | tee "$result_log"
  done
done

docker cp "$CONTAINER_NAME:$CONTAINER_OUT_DIR/." "$OUT_DIR/"

python3 - "$OUT_DIR" "$PROMPT_LEN" "$OUTPUT_LEN" <<'PY'
import json
import os
import re
import sys
from glob import glob

result_dir = sys.argv[1]
default_prompt_len = int(sys.argv[2])
default_output_len = int(sys.argv[3])

pattern = re.compile(r"c(?P<concurrency>\d+)_depth(?P<depth>\d+)_in(?P<prompt>\d+)_out(?P<output>\d+)\.json$")

rows = []
for json_path in sorted(glob(os.path.join(result_dir, "c*_depth*_in*_out*.json"))):
    match = pattern.search(os.path.basename(json_path))
    if not match:
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows.append(
        {
            "concurrency": int(match.group("concurrency")),
            "context_depth": int(match.group("depth")),
            "prompt_size": int(match.group("prompt")) if match.group("prompt") else default_prompt_len,
            "response_size": int(match.group("output")) if match.group("output") else default_output_len,
            "total_input_tokens": data.get("total_input_tokens"),
            "num_prompts": data.get("num_prompts"),
            "completed": data.get("completed"),
            "failed": data.get("failed"),
            "duration": data.get("duration"),
            "request_throughput": data.get("request_throughput"),
            "output_throughput": data.get("output_throughput"),
            "total_token_throughput": data.get("total_token_throughput"),
            "mean_ttft_ms": data.get("mean_ttft_ms"),
            "p99_ttft_ms": data.get("p99_ttft_ms"),
            "mean_tpot_ms": data.get("mean_tpot_ms"),
            "p99_tpot_ms": data.get("p99_tpot_ms"),
            "spec_decode_acceptance_rate": data.get("spec_decode_acceptance_rate"),
            "spec_decode_acceptance_length": data.get("spec_decode_acceptance_length"),
        }
    )

rows.sort(key=lambda row: (row["concurrency"], row["context_depth"], row["prompt_size"], row["response_size"]))

summary_path = os.path.join(result_dir, "summary.tsv")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(
        "concurrency\tcontext_depth\tprompt_size\tresponse_size\tnum_prompts\tcompleted\tfailed\t"
        "duration\trequest_throughput\toutput_throughput\ttotal_token_throughput\t"
        "mean_ttft_ms\tp99_ttft_ms\tmean_tpot_ms\tp99_tpot_ms\t"
        "spec_decode_acceptance_rate\tspec_decode_acceptance_length\ttotal_input_tokens\n"
    )
    for row in rows:
        f.write(
            f"{row['concurrency']}\t{row['context_depth']}\t{row['prompt_size']}\t{row['response_size']}\t"
            f"{row['num_prompts']}\t{row['completed']}\t{row['failed']}\t{row['duration']}\t"
            f"{row['request_throughput']}\t{row['output_throughput']}\t{row['total_token_throughput']}\t"
            f"{row['mean_ttft_ms']}\t{row['p99_ttft_ms']}\t{row['mean_tpot_ms']}\t{row['p99_tpot_ms']}\t"
            f"{row['spec_decode_acceptance_rate']}\t{row['spec_decode_acceptance_length']}\t"
            f"{row['total_input_tokens']}\n"
        )

print(f"SUMMARY_TSV={summary_path}")
for row in rows:
    print(
        "concurrency={concurrency} depth={depth} prompt={prompt} output={output} "
        "req/s={req:.2f} tok/s={tok:.2f} ttft_ms={ttft:.2f} p99_ttft_ms={p99_ttft:.2f} "
        "tpot_ms={tpot:.2f} accept_rate={accept:.2f}%".format(
            concurrency=row["concurrency"],
            depth=row["context_depth"],
            prompt=row["prompt_size"],
            output=row["response_size"],
            req=row["request_throughput"] or 0.0,
            tok=row["output_throughput"] or 0.0,
            ttft=row["mean_ttft_ms"] or 0.0,
            p99_ttft=row["p99_ttft_ms"] or 0.0,
            tpot=row["mean_tpot_ms"] or 0.0,
            accept=row["spec_decode_acceptance_rate"] or 0.0,
        )
    )
PY
