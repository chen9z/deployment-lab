# Local Model Deployment Lab

Deployment configs for local OpenAI-compatible model servers. The repository
keeps Compose files for the current local GPU estate. Model checkpoints live
under `models/` or under absolute paths referenced by the relevant Compose
file.

### Qwen3.6-27B INT4 AutoRound on RTX 5090 (32GB, vLLM)

The retained 5090 deployment path is the Docker Compose stack under
`qwen3.5-27b/`. Historical shell launchers have been removed.

The Compose stack keeps these defaults:

- `--tensor-parallel-size 1` on a single 5090 (auto-clamped if you set a larger value)
- `--max-model-len 262144`
- `--quantization auto_round`
- `--dtype float16`
- `--gpu-memory-utilization 0.93`
- `--max-num-seqs 4`
- `--max-num-batched-tokens 4128`
- `--kv-cache-dtype fp8_e4m3`
- `--attention-backend FLASHINFER`
- `--reasoning-parser qwen3`
- `--enable-auto-tool-choice`
- `--tool-call-parser qwen3_coder`
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3,"quantization":null}'`

Use environment variables with `docker compose` for overrides.

### Qwen3.6-27B Docker Compose on RTX 5090

The Compose stack under `qwen3.5-27b/` serves `models/Lorbus/Qwen3.6-27B-int4-AutoRound` on the RTX 5090. On the current machine, `nvidia-smi -L` shows `GPU 1` is the `RTX 5090`, so the stack is pinned to that card by UUID.

By default the `vllm` service uses `${QWEN36_VLLM_IMAGE:-vllm/vllm-openai:v0.24.0-cu129-ubuntu2404}` so the stack uses the local CUDA 12.9 vLLM image while still allowing image overrides without editing the compose file.
The 5090 deployment keeps the OpenAI-compatible endpoint on `http://127.0.0.1:8001` with served model name `Qwen3.5-27B`. It uses `--max-model-len 262144`, `--dtype float16`, `--gpu-memory-utilization 0.93`, `--max-num-seqs 4`, `--max-num-batched-tokens 4128`, `fp8_e4m3` KV cache, FlashInfer attention, the enhanced Qwen chat template, and MTP speculative decoding with `num_speculative_tokens=3`. A 255K-token completion request has been smoke-tested on the 5090. `OMP_NUM_THREADS=1` follows the club-3090 vLLM profiles and reduced host-side thread fan-out without hurting decode throughput. `PYTORCH_CUDA_ALLOC_CONF` is set to `expandable_segments:False,max_split_size_mb:512`; the local CUDA 12.9 vLLM image fails during Qwen MTP drafter allocation with expandable segments enabled.
If the current stack cannot hold the full context window under load, lower `max_num_seqs` and `max_num_batched_tokens` first before reducing `max_model_len`.

Start it with:

```bash
docker compose -f qwen3.5-27b/docker-compose.yml up -d
```

### Qwen3.6-27B Evaluation

Model evaluation scripts live under `eval/` and default to the Qwen service at
`http://127.0.0.1:8001` with served model name `Qwen3.5-27B`.

```bash
eval/verify.sh
eval/verify-full.sh
eval/bench.sh
```

Use `eval/verify-stress.sh` for slower long-context and prefill boundary
checks after image or Compose changes. Override `URL`, `MODEL`, or `CONTAINER`
when pointing the same scripts at another OpenAI-compatible endpoint.

### Gemma 4 Docker on 1x RTX 3090

For containerized deployment on the second `RTX 3090` (`GPU 2`), the Gemma4
stack runs the local QAT AWQ INT4 checkpoint with
`vllm/vllm-openai:nightly-rollback-20260516`:

```bash
docker compose -f gemma-4-26b/docker-compose.gemma4.yml up -d
```

This mounts
`models/cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`, binds the
OpenAI-compatible API to `http://127.0.0.1:8006`, and serves the model as
`gemma-4-26B-A4B`. It keeps `temperature=1.0`, `top_p=0.95`, and
`top_k=64` as the default sampling config, enables
`--enable-auto-tool-choice`, `--reasoning-parser gemma4`,
`--tool-call-parser gemma4`, and `--async-scheduling`, and allows up to `2`
input images via `--limit-mm-per-prompt image=2`. Audio input is not enabled.

The `vllm/vllm-openai:gemma-cu129` image does not currently start this
checkpoint on the RTX 3090. Its CUDA 12.9 Marlin W4A16 MoE path fails in
`gptq_marlin_moe_repack` with `CUDA driver error: device not ready`.

The tuned 3090 defaults use a 131072-token context window,
`--max-num-seqs 16`, and `--max-num-batched-tokens 8192`. Override them
without editing the Compose file by setting `GEMMA_MAX_MODEL_LEN`,
`GEMMA_MAX_NUM_SEQS`, or `GEMMA_MAX_NUM_BATCHED_TOKENS` when invoking
Compose. The 128K profile retains roughly 140K tokens of GPU KV cache on the
current card, so it supports one full-length request or a larger batch of
shorter requests.

### Gemma 4 31B INT4 AutoRound on 2x RTX 3090

The `gemma-4-31b/` directory keeps the dual-3090 Gemma 31B reference stack,
including the Compose file and local vLLM patch overlays used for
`Intel/gemma-4-31B-it-int4-AutoRound`.

Models are expected under `/models` by default:

- `/models/Intel/gemma-4-31B-it-int4-AutoRound`
- `/models/google/gemma-4-31B-it-assistant`

Invoke Compose directly:

```bash
docker compose -f gemma-4-31b/docker-compose.2x3090.autoround.yml up -d
```

The standalone run, stop, and benchmark helper scripts for this stack are not
kept in the repository; the directory is retained as a reference Compose and
patch set.

### Jina Embeddings v5 Text Small on RTX 3090

The stack under `jina-v5-embedding/` serves the official multitask checkpoint
at `models/jinaai/jina-embeddings-v5-text-small` through a Jina-compatible
endpoint on `http://127.0.0.1:8016/v1/embeddings`. The served model name is
`jina-embeddings-v5-text-small`.

It uses one vLLM BF16 base model with four dynamic LoRA adapters for
`retrieval`, `text-matching`, `classification`, and `clustering`. The gateway
maps the Jina `task` field to the corresponding vLLM LoRA while retaining
continuous batching, FlashAttention 2, torch compilation, and piecewise CUDA
Graphs. The base model is forced to the raw `Qwen3Model`; otherwise vLLM's
Jina-specific loader would merge the retrieval adapter before applying the
selected dynamic LoRA.

Returned float embeddings are always L2-normalized. Supported Matryoshka
dimensions are `32`, `64`, `128`, `256`, `512`, `768`, and `1024`, and the
configured context limit is 32768 tokens. The default GPU memory utilization is
`0.26`, which allocates enough KV cache for one 32K request while using about
6.4 GiB on the current RTX 3090.

Start or rebuild it with:

```bash
docker compose -f jina-v5-embedding/docker-compose.yml up -d --build
```

The API accepts plain OpenAI-style strings and Jina-style text objects:

```bash
curl http://127.0.0.1:8016/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v5-text-small",
    "task": "retrieval.query",
    "dimensions": 256,
    "normalized": true,
    "input": [{"text": "Which planet is known as the Red Planet?"}]
  }'
```

Use `retrieval.query` or `retrieval.passage` to apply the model's `Query: ` or
`Document: ` prefix. The API also accepts `retrieval`, `text-matching`,
`classification`, and `clustering`; the corresponding adapter is activated
for each request. A mixed retrieval request can set `prompt_name` to `query`
or `document` on each input object. Authorization headers are accepted but
are not validated by this local service.

This is a text-only model. Inputs containing an `image` field return HTTP 400.
Use `jina-embeddings-v5-omni-small`, `jina-embeddings-v4`, or a Jina CLIP model
when text and images must share an embedding space.

Benchmark methodology and the deployment comparisons are recorded in
`jina-v5-embedding/BENCHMARK.md`. The model is distributed under CC BY-NC 4.0;
review the [model card](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)
before commercial use.

## Project Layout

```
deployment-lab/
├── eval/               # Model evaluation scripts
├── gemma-4-26b/        # Gemma 4 26B Compose config
├── gemma-4-31b/        # Gemma 4 31B reference config and patches
├── jina-v5-embedding/  # Jina-compatible embedding service and benchmark
├── models/             # Local checkpoint subdirectories
├── qwen3.5-27b/        # Qwen 27B Compose config and runtime cache
└── README.md
```

## Notes

- Hugging Face caches live under the default `HF_HOME`. Set it before launch if needed.
- CUDA visibility is controlled by each launcher or Compose file.
