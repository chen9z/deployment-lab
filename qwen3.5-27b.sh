#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

vllm serve "$ROOT_DIR/models/Lorbus/Qwen3.6-27B-int4-AutoRound" --served-model-name=Qwen3.6-27B \
--dtype half \
--tensor-parallel-size=1 \
--max-model-len=262144 \
--max-num-seqs=4 \
--enable-auto-tool-choice \
--kv-cache-dtype fp8 \
--tool-call-parser qwen3_xml \
--reasoning-parser qwen3 \
--enable-prefix-caching \
--attention-backend FLASHINFER \
--gpu-memory-utilization=0.92 \
--max-num-batched-tokens=4096 \
--limit-mm-per-prompt.video 0 \
--host=0.0.0.0 --port=8001 \
--compilation-config.cudagraph_mode none \
--speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
--generation-config auto \
--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
