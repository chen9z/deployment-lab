# Local Model Deployment Lab

Deployment configs for local OpenAI-compatible model servers. The repository
keeps Compose files for the current local GPU estate. Model checkpoints live
under `models/` or under absolute paths referenced by the relevant Compose
file.

### Qwen3.8-27B NVFP4 on RTX 5090 (NInfer)

The default 27B stack is `qwen3.8-27b/docker-compose.yml`. It builds the local
`ninfer/` source, loads the registered Qwen3.8-27B NVFP4 `.ninfer` artifact,
and pins execution to the RTX 5090. The public OpenAI-compatible model alias
remains `Qwen3.5-27B` so existing callers can move to the new weights without
changing request payloads.

The default profile keeps a 262,144-token per-request ceiling and one shared
262,144-token INT8 KV pool for four active request lanes. One request can use
the complete context window; concurrent shorter requests share that pool and
form compact decode batches. MTP3, the optimized proposal head, CUDA Graphs,
and prefix reuse are enabled. The sampling defaults match the repository's
embedded Qwen3.8 profiles: thinking uses `1.0/0.95/20/0/0`, while
non-thinking uses `0.7/0.8/20/0/1.5` for
temperature/top-p/top-k/min-p/presence penalty. Thinking is disabled by
default; callers can explicitly enable it with the top-level `enable_thinking`
or `reasoning_effort` request field.

Start it on the repository's standard Qwen endpoint:

```bash
docker compose -f qwen3.8-27b/docker-compose.yml up -d
```

The endpoint remains `http://127.0.0.1:8001/v1`. NInfer also exposes its
Anthropic-compatible endpoint on the same port. Override the image or artifact
with `QWEN38_NINFER_IMAGE` or `QWEN38_NINFER_ARTIFACT`. Tune the logical
context ceiling, shared KV pool, active lanes, pending queue, or prefill chunk
with `QWEN38_MAX_CONTEXT`, `QWEN38_KV_CAPACITY`,
`QWEN38_MAX_CONCURRENCY`, `QWEN38_MAX_PENDING_REQUESTS`, or
`QWEN38_PREFILL_CHUNK`. Pending requests wait up to 300 seconds by default;
override that bound with `QWEN38_PENDING_TIMEOUT_MS`.

The validated RTX 5090 profile accepted a 260,096-token prompt and returned
the expected NIAH value. Four concurrent 4,096-token structured generations
completed in 24.41 seconds with the embedded non-thinking sampler and sustained
769.9 aggregate decode tok/s over complete full-batch intervals. Four concurrent
4,096-token reasoning generations completed in 40.15 seconds and sustained
426.6 aggregate decode tok/s with the embedded thinking sampler. The C=4 layout
leaves little unused device memory; do not raise concurrency without reducing
the shared KV capacity.

### Legacy Qwen3.6-27B stack

The prior vLLM deployment remains under `qwen3.5-27b/` as a legacy reference,
but it is no longer the default 27B service. It uses the same RTX 5090 and port
8001, so stop the NInfer stack before starting that Compose file.

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
├── qwen3.5-27b/        # Legacy Qwen 27B vLLM Compose stack
├── qwen3.8-27b/        # Default Qwen 27B NInfer Compose stack
├── ninfer/             # NInfer C++/CUDA inference engine
└── README.md
```

## Notes

- Hugging Face caches live under the default `HF_HOME`. Set it before launch if needed.
- CUDA visibility is controlled by each launcher or Compose file.
