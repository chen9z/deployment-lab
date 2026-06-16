# Jina v5 embedding benchmark

Measured on June 16, 2026 with one RTX 3090 and 1024-dimensional normalized
outputs. Each input is roughly 40 English tokens. Latency includes HTTP,
serialization, and response transfer.

| Workload | Retrieval-only vLLM | Optimized Transformers | vLLM multi-LoRA |
| --- | ---: | ---: | ---: |
| Batch 1 throughput | 78.0 texts/s | 59.1 texts/s | 79.2 texts/s |
| Batch 8 throughput | 490.8 texts/s | 318.1 texts/s | 366.1 texts/s |
| Batch 32 throughput | 955.0 texts/s | 422.7 texts/s | 697.2 texts/s |
| Batch 64 throughput | 1191.2 texts/s | 442.7 texts/s | 1197.4 texts/s |
| Concurrency 4 throughput | 294.7 texts/s | 220.7 texts/s | 195.6 texts/s |
| Concurrency 16 throughput | 586.5 texts/s | 303.0 texts/s | 461.1 texts/s |

The vLLM figures use the retrieval adapter merged into one static checkpoint.
That is substantially faster, but it cannot correctly switch to
`text-matching`, `classification`, or `clustering` at request time.

The final deployment loads the 1.11 GiB base checkpoint once and registers all
four roughly 40 MiB adapters with vLLM. The adapters therefore add about
160 MiB rather than four copies of the base model. The total observed GPU
allocation is about 6.4 GiB; most of the difference from model weight size is
the 37,168-token KV cache and CUDA Graph workspace required for the 32K context
window.

The vLLM pooling OpenAI route in the pinned image lists static LoRAs but fails
to resolve them during embedding requests. `vllm_middleware.py` applies a small
request-time compatibility patch that selects the already-registered
`LoRARequest`. The model is loaded as raw `Qwen3Model` through HF overrides so
the selected adapter is applied exactly once. Output comparisons against the
Transformers implementation have cosine similarity between 0.99987 and
0.99995 across all four tasks.

Re-run the current-container benchmark with:

```bash
python3 jina-v5-embedding/benchmark.py
```
