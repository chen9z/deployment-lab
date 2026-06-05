# Local Model Deployment Lab

Deployment configs for local OpenAI-compatible model servers. The repository
keeps Compose files for the current local GPU estate. Model checkpoints live
under `models/` or under absolute paths referenced by the relevant Compose
file.

### Qwen3.6-27B INT4 AutoRound on RTX 5090 (32GB, vLLM + uv)

The retained 5090 deployment path is the Docker Compose stack under
`qwen3.5-27b/`. Historical shell launchers have been removed.

The Compose stack keeps these defaults:

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

Use environment variables with `docker compose` for overrides.

### Qwen3.6-27B Docker Compose on RTX 5090

The Compose stack under `qwen3.5-27b/` serves `models/Lorbus/Qwen3.6-27B-int4-AutoRound` on `GPU 1`. On the current machine, `nvidia-smi -L` shows `GPU 1` is the `RTX 5090`, so the stack is pinned to that card.

By default the `vllm` service uses `${QWEN36_VLLM_IMAGE:-vllm/vllm-openai:nightly-rollback-20260516}` so the stack can be pinned or moved to a newer local image without editing the compose file.
The 5090 deployment keeps the OpenAI-compatible endpoint on `http://127.0.0.1:8001` with served model name `Qwen3.5-27B`. It uses `--max-model-len 262144`, `--gpu-memory-utilization 0.93`, `--max-num-seqs 4`, `--max-num-batched-tokens 4128`, `fp8_e4m3` KV cache, FlashInfer attention, the enhanced Qwen chat template, and MTP speculative decoding with `num_speculative_tokens=3`.
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

For containerized deployment on the first `RTX 3090` (`GPU 1`), the Gemma4 stack now builds a small local compatibility image on top of `vllm/vllm-openai:v0.19.1-cu130` and then starts the service:

```bash
docker compose -f gemma-4-26b/docker-compose.gemma4.yml up -d
```

This binds the OpenAI-compatible API to `http://127.0.0.1:8006`, serves the model as `gemma-4-26B-A4B`, keeps `temperature=1.0`, `top_p=0.95`, `top_k=64` as the default sampling config, enables `--enable-auto-tool-choice`, `--reasoning-parser gemma4`, `--tool-call-parser gemma4`, `--async-scheduling`, and allows up to `2` input images via `--limit-mm-per-prompt image=2`. Audio input is not enabled.

The local patch only adjusts Gemma4 MoE compressed-tensors weight-name handling for this AWQ checkpoint so that `v0.19.1-cu130` can load it successfully.

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

## Project Layout

```
deployment-lab/
├── eval/               # Model evaluation scripts
├── gemma-4-26b/        # Gemma 4 26B Compose config
├── gemma-4-31b/        # Gemma 4 31B reference config and patches
├── jina-v5-embedding/  # Jina embedding vLLM Compose config
├── models/             # Local checkpoint subdirectories
├── qwen3.5-27b/        # Qwen 27B Compose config and runtime cache
└── README.md
```

## Notes

- Hugging Face caches live under the default `HF_HOME`. Set it before launch if needed.
- CUDA visibility is controlled by each launcher or Compose file.
