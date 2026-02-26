#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
MODEL="${MODEL:-Qwen3.5-35B-A3B}"   # 你的 --served-model-name
MODEL_REPO="${MODEL_REPO:-cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit}"  # 用于 tokenizer 的 HF repo id
OUT_DIR="${OUT_DIR:-bench_results/$(date +%Y%m%d_%H%M%S)}"

# 并发与请求数（一般请求数至少是并发的 10~20 倍，统计更稳）
IFS=' ' read -r -a CONCURRENCY_LIST <<< "$(printf '%s' "${CONCURRENCY_LIST:-1 2 4 8}" | tr ',' ' ')"
NUM_REQUESTS="${NUM_REQUESTS:-80}"

# 输入/输出长度 sweep（token 数）
# 你跑 262K 上下文时建议先从 8k/32k/64k 开始，别一上来 128k+
IFS=' ' read -r -a INPUT_LEN_LIST <<< "$(printf '%s' "${INPUT_LEN_LIST:-8192 32768 65536}" | tr ',' ' ')"
IFS=' ' read -r -a OUTPUT_LEN_LIST <<< "$(printf '%s' "${OUTPUT_LEN_LIST:-256 512 1024}" | tr ',' ' ')"

# dataset 选择：random（合成 prompt），最通用
DATASET="${DATASET:-random}"

mkdir -p "$OUT_DIR"

echo "[INFO] Base URL: $BASE_URL"
echo "[INFO] Model:    $MODEL"
echo "[INFO] Repo:     $MODEL_REPO"
echo "[INFO] Out dir:  $OUT_DIR"

# =========================
# Sanity check server
# =========================
echo "[INFO] Checking /v1/models ..."
if ! curl -sS --max-time 3 "${BASE_URL}/v1/models" | head -c 400; then
  ALT_BASE_URL="${BASE_URL/127.0.0.1/127.0.1.1}"
  if [ "${ALT_BASE_URL}" != "${BASE_URL}" ] && \
     curl -sS --max-time 3 "${ALT_BASE_URL}/v1/models" >/dev/null; then
    BASE_URL="${ALT_BASE_URL}"
    echo
    echo "[WARN] Fallback to reachable base URL: ${BASE_URL}"
    curl -sS --max-time 3 "${BASE_URL}/v1/models" | head -c 400 || true
  fi
fi
echo
echo "[INFO] Starting sweeps..."

if vllm bench serve --help=all 2>/dev/null | grep -q -- '--dataset-name'; then
  DATASET_FLAG="--dataset-name"
else
  DATASET_FLAG="--dataset"
fi

if vllm bench serve --help=all 2>/dev/null | grep -q -- '--num-prompts'; then
  COUNT_FLAG="--num-prompts"
else
  COUNT_FLAG="--num-requests"
fi

# =========================
# Run sweeps
# vllm bench serve:
#   - 用于 OpenAI-compatible server 的在线服务压测
#   - 支持 --model / --input-len / --output-len / --max-concurrency 等
# =========================
for IN_LEN in "${INPUT_LEN_LIST[@]}"; do
  for OUT_LEN in "${OUTPUT_LEN_LIST[@]}"; do
    for C in "${CONCURRENCY_LIST[@]}"; do
      NAME="serve_${DATASET}_in${IN_LEN}_out${OUT_LEN}_c${C}_n${NUM_REQUESTS}"
      LOG="${OUT_DIR}/${NAME}.log"
      JSON="${OUT_DIR}/${NAME}.json"

      echo "[RUN] $NAME"
      # 说明：vllm bench serve 会输出吞吐与延迟统计（TTFT/TPOT/ITL等随版本略有差异）
      #      参数名与含义见 vLLM bench serve 文档
      vllm bench serve \
        --base-url "${BASE_URL}" \
        --model "${MODEL_REPO}" \
        --served-model-name "${MODEL}" \
        "${DATASET_FLAG}" "${DATASET}" \
        --input-len "${IN_LEN}" \
        --output-len "${OUT_LEN}" \
        --max-concurrency "${C}" \
        "${COUNT_FLAG}" "${NUM_REQUESTS}" \
        --save-result \
        --result-filename "${JSON}" \
        2>&1 | tee "${LOG}"

      echo
    done
  done
done

echo "[DONE] Results saved to: ${OUT_DIR}"
