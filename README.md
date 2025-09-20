# Multi-Model Embedding & Reranking API Server

FastAPI service that implements OpenAI-style `/v1/embeddings` and `/v1/rerank` endpoints. Models are loaded from Hugging Face and exposed through a minimal wrapper layer.

## Supported Models

- `jinaai/jina-embeddings-v4`
- `jinaai/jina-code-embeddings-1.5b`
- `Qwen/Qwen3-Embedding-4B`
- `jinaai/jina-reranker-m0`
- `jinaai/jina-reranker-v2-base-multilingual`

All wrappers live in `models/` and lazy-load checkpoints on demand.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running the Server

Default start (preloads `jina-embeddings-v4` and `jina-reranker-m0`):

```bash
python server.py
```

Override preload list:

```bash
python server.py --model jinaai/jina-code-embeddings-1.5b --model Qwen/Qwen3-Embedding-4B
```

You can still use uvicorn directly:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Endpoints
- `GET /` – service metadata & supported models
- `GET /v1/models` – supported models in OpenAI schema
- `POST /v1/embeddings` – generate embeddings
- `POST /v1/rerank` – rerank documents for a query

## Benchmarks

Manual performance scripts live in `benchmark/`. Activate your virtualenv, then run for example:

```bash
python benchmark/embedding_latency.py --model jinaai/jina-embeddings-v4
```

## Testing

A small pytest suite validates the factory wiring:

```bash
pytest tests/test_factory.py
```

## Project Layout

```
deployment-lab/
├── api/                # FastAPI routes & schemas
├── benchmark/          # Benchmark scripts for embeddings/rerankers
├── models/             # Model wrappers and factory
├── server.py           # FastAPI application & CLI entrypoint
├── tests/              # Pytest modules & manual scripts
└── README.md
```

## Notes

- Hugging Face caches live under the default `HF_HOME`. Set it before launch if needed.
- CUDA is used automatically when available; otherwise models fall back to CPU.