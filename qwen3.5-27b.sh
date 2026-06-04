#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_FLOAT32_MATMUL_PRECISION="${VLLM_FLOAT32_MATMUL_PRECISION:-high}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

vllm serve "$ROOT_DIR/models/Lorbus/Qwen3.6-27B-int4-AutoRound" --served-model-name=Qwen3.6-27B \
--dtype half \
--tensor-parallel-size=1 \
--max-model-len=262144 \
--max-num-seqs=4 \
--enable-auto-tool-choice \
--quantization auto_round \
--skip-mm-profiling \
--kv-cache-dtype fp8_e4m3 \
--tool-call-parser qwen3_xml \
--reasoning-parser qwen3 \
--enable-chunked-prefill \
--enable-prefix-caching \
--attention-backend FLASHINFER \
--gpu-memory-utilization=0.92 \
--max-num-batched-tokens=4128 \
--limit-mm-per-prompt.video 0 \
--host=0.0.0.0 --port=8001 \
--chat-template "$ROOT_DIR/qwen3.5-27b/qwen3.5-enhanced.jinja" \
--compilation-config.cudagraph_mode none \
--speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
--generation-config auto \
--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
