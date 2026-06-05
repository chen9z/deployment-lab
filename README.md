# Local Model Deployment Lab

Deployment and benchmark scripts for local OpenAI-compatible model servers.
The repository keeps Compose files, launch scripts, and benchmark helpers for
the current local GPU estate. Model checkpoints live under `models/` or under
absolute paths referenced by the relevant Compose file.

### Qwen3.6-27B INT4 AutoRound on RTX 5090 (32GB, vLLM + uv)

For local vLLM deployment of `Lorbus/Qwen3.6-27B-int4-AutoRound`, use the script below:

```bash
script/run_qwen35_27b_fp8_5090.sh
```

This keeps the tool-use settings from the model card and applies 5090-safe defaults:

- `--tensor-parallel-size 1` on a single 5090 (auto-clamped if you set a larger value)
- `--max-model-len 262144`
- `--dtype half`
- `--gpu-memory-utilization 0.92`
- `--max-num-seqs 1`
- `--kv-cache-dtype fp8`
- `--reasoning-parser qwen3`
- `--enable-auto-tool-choice`
- `--tool-call-parser qwen3_xml`
- `--compilation-config.cudagraph_mode none`
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`

Optional overrides:

```bash
MAX_MODEL_LEN=32768 GPU_MEMORY_UTILIZATION=0.99 PORT=8002 script/run_qwen35_27b_fp8_5090.sh
```

Speculative decode override:

```bash
# disable MTP
SPECULATIVE_CONFIG=off script/run_qwen35_27b_fp8_5090.sh

# custom MTP
SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":2}' script/run_qwen35_27b_fp8_5090.sh
```

### Qwen3.6-27B Docker Compose on RTX 5090

The Compose stack under `qwen3.5-27b/` serves `models/Lorbus/Qwen3.6-27B-int4-AutoRound` on `GPU 1`. On the current machine, `nvidia-smi -L` shows `GPU 1` is the `RTX 5090`, so the stack is pinned to that card.

By default the `vllm` service uses `${QWEN36_VLLM_IMAGE:-vllm/vllm-openai:nightly-rollback-20260516}` so the stack can be pinned or moved to a newer local image without editing the compose file.
The 5090 deployment keeps the OpenAI-compatible endpoint on `http://127.0.0.1:8001` with served model name `Qwen3.5-27B`. It uses `--max-model-len 262144`, `--gpu-memory-utilization 0.93`, `--max-num-seqs 4`, `--max-num-batched-tokens 4128`, `fp8_e4m3` KV cache, FlashInfer attention, the enhanced Qwen chat template, and MTP speculative decoding with `num_speculative_tokens=3`.
If the current stack cannot hold the full context window under load, lower `max_num_seqs` and `max_num_batched_tokens` first before reducing `max_model_len`.

Start it with:

```bash
docker compose -f qwen3.5-27b/docker-compose.yml up -d
```

### Qwen3.6-27B Stress Benchmark

To benchmark the deployed 5090 service, run:

```bash
script/bench_qwen36_27b_int4_5090.sh
```

The script checks `/v1/models`, then runs a `vllm bench serve` sweep over prompt length, context depth, and concurrency against `/v1/chat/completions` from inside the serving container. It copies the raw JSON results back to the repo and writes `summary.tsv` alongside them.

### Qwen3.6-27B on 2x RTX 3090

For the local `models/Lorbus/Qwen3.6-27B-int4-AutoRound` checkpoint, the dual-3090 Docker setup follows the reference `qwen36-dual-3090` recipe: tensor parallelism across two GPUs, `auto_round`, `fp8_e5m2` KV cache, `--disable-custom-all-reduce`, and speculative MTP.

Start the containerized server:

```bash
script/run_qwen35_27b_awq_2x3090_docker.sh
```

Or invoke Compose directly:

```bash
docker compose -f qwen3.5-27b/docker-compose.2x3090.awq.yml up -d
```

Run a benchmark against the exposed OpenAI-compatible endpoint:

```bash
script/bench_qwen35_27b_awq_2x3090.sh
```

Stop the stack:

```bash
script/stop_qwen35_27b_awq_2x3090_docker.sh
```

Defaults:

- Binds host `GPU 1,2` only, leaving the `RTX 5090` untouched
- Serves on `http://127.0.0.1:8010`
- Uses `--max-model-len 262144`
- Uses `--max-num-batched-tokens 8192`
- Uses `--kv-cache-dtype fp8_e5m2`
- Uses `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
- Benchmarks one single-stream case and one `8`-concurrency throughput case via `vllm bench serve`

If Docker GPU filtering is flaky on your machine, the existing host-side launcher still works:

```bash
script/run_qwen35_27b_awq_2x3090.sh
```

### Gemma 4 Docker on 1x RTX 3090

For containerized deployment on the first `RTX 3090` (`GPU 1`), the Gemma4 stack now builds a small local compatibility image on top of `vllm/vllm-openai:v0.19.1-cu130` and then starts the service:

```bash
docker compose -f gemma-4-26b/docker-compose.gemma4.yml up -d
```

Run a quick benchmark against the Docker service:

```bash
script/bench_gemma4_26b_awq_3090.sh
```

This binds the OpenAI-compatible API to `http://127.0.0.1:8006`, serves the model as `gemma-4-26B-A4B`, keeps `temperature=1.0`, `top_p=0.95`, `top_k=64` as the default sampling config, enables `--enable-auto-tool-choice`, `--reasoning-parser gemma4`, `--tool-call-parser gemma4`, `--async-scheduling`, and allows up to `2` input images via `--limit-mm-per-prompt image=2`. Audio input is not enabled.

The local patch only adjusts Gemma4 MoE compressed-tensors weight-name handling for this AWQ checkpoint so that `v0.19.1-cu130` can load it successfully.

## Project Layout

```
deployment-lab/
├── gemma-4-26b/        # Gemma 4 26B Compose config
├── gemma-4-31b/        # Gemma 4 31B Compose config and patches
├── jina-v5-embedding/  # Jina embedding vLLM Compose config
├── models/             # Local checkpoint subdirectories
├── qwen3.5-27b/        # Qwen 27B Compose config and runtime cache
├── script/             # Launch, stop, and benchmark scripts
└── README.md
```

## Notes

- Hugging Face caches live under the default `HF_HOME`. Set it before launch if needed.
- CUDA visibility is controlled by each launcher or Compose file.
