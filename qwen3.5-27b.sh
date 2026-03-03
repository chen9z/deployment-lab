#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate 

vllm serve Kbenkhaled/Qwen3.5-27B-NVFP4 --served-model-name=qwen3.5-27b \
--tensor-parallel-size=1 \
--max-model-len=131072 \
--max-num-seqs=1 \
--enable-auto-tool-choice \
--kv-cache-dtype fp8 \
--tool-call-parser qwen3_coder \
--reasoning-parser qwen3 \
--enable-prefix-caching \
--attention-backend FLASHINFER \
--gpu-memory-utilization=0.85 \
--limit-mm-per-prompt.video 0 \
--host=0.0.0.0 --port=8001 \
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}' \
--generation-config auto \
--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
