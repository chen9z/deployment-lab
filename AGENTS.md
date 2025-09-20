# Repository Guidelines

## Project Structure & Module Organization
- `server.py` owns the FastAPI app, startup model preloading, and CLI entry (`python server.py --model ...`).
- `api/` houses the router (`routes.py`), request/response schemas, and the lightweight model manager used by the endpoints.
- `models/` contains the shared wrapper base plus per-model wrappers (e.g., Jina v4, Jina code, Qwen embeddings; Jina rerankers) registered by the factory.
- `benchmark/` stores manual latency scripts for embeddings and rerankers.
- `tests/` keeps minimal pytest coverage (e.g., factory wiring) plus archived manual scripts under `tests/scripts/`.
- `README.md` documents runtime instructions—update it with any new model support or CLI flags.

## Development & Runtime Commands
- Bootstrap with `python3 -m venv .venv && source .venv/bin/activate && pip install -e .[test]`.
- Start the service via `python server.py` for default preload (`jina-embeddings-v4`, `jina-reranker-m0`) or override using repeated `--model` flags.
- Use `uvicorn server:app --reload` while iterating; avoid committing reload mode defaults.
- Benchmarks run from the `benchmark/` directory, e.g. `python benchmark/embedding_latency.py --model Qwen/Qwen3-Embedding-4B`.

## Coding Style & Naming Conventions
Target Python 3.11, stick with four-space indentation, and keep module docstrings short. Use `snake_case` for functions and async helpers, `CamelCase` for wrappers, and verb-centric coroutine names (`compute_scores`). Prefer f-strings for logging and protect CUDA-only branches with `torch.cuda.is_available()`. Run `python -m black` before large refactors when possible.

## Testing Guidelines
`pytest` coverage currently focuses on the factory layer (`tests/test_factory.py`); API interfaces are validated manually through scripts in `tests/scripts/`. When adding new registry logic, extend or clone the factory tests rather than spinning up the API. Manual smoke checks should stay in the scripts directory with a module-level pytest skip guard.

## Commit & Pull Request Guidelines
Write imperative, ≤60-character commit subjects (e.g., `Add Qwen embedding wrapper`). Document the rationale, mention executed sanity checks (`pytest`, benchmark scripts, manual curl), and call out new models or CLI flags in PR descriptions. Keep changes scoped; land benchmark tooling separately from API behavior tweaks when practical.

## Security & Configuration Notes
Do not bake credentials into code or scripts. Encourage contributors to configure `HF_HOME`/`TRANSFORMERS_CACHE` for shared environments and capture new environment toggles in the README. If adding benchmarks that store outputs, ensure they do not leak proprietary data.
