# Repository Guidelines

## Project Structure & Module Organization
- `qwen3.5-27b/`, `gemma-4-26b/`, `gemma-4-31b/`, and `jina-v5-embedding/` hold model-specific Compose files, chat templates, patches, and runtime cache ignores.
- `script/` stores launch, stop, autotune, and vLLM benchmark helpers for local deployments.
- `models/` stores local checkpoint subdirectories mounted by Compose files; do not treat it as an importable Python package.
- `benchmark/` and `tests/` may contain archived manual scripts from the removed FastAPI wrapper service; prefer deployment-specific scripts under `script/`.
- `README.md` documents runtime instructions—update it with any new model support or CLI flags.

## Development & Runtime Commands
- Start Qwen 27B on the 5090 with `docker compose -f qwen3.5-27b/docker-compose.yml up -d`.
- Run the Qwen vLLM sweep with `script/bench_qwen36_27b_int4_5090.sh`.
- Start Gemma/Jina stacks with their model-specific Compose files or `script/run_*` helpers.
- Avoid committing generated benchmark result directories or runtime caches.

## Coding Style & Naming Conventions
Target Python 3.11 for helper scripts, stick with four-space indentation, and keep module docstrings short. Use `snake_case` for functions and verb-centric helper names. Prefer f-strings for logging, quote shell variables in scripts, and keep Compose overrides explicit through environment variables. Run `python -m black` before large Python refactors when possible.

## Testing Guidelines
Validate changes with the narrowest deployment-specific check available: `docker compose config`, `/v1/models` smoke checks, and the relevant `script/bench_*` helper for performance-sensitive changes. Archived pytest/API tests may reference the removed FastAPI wrapper service and should not be treated as authoritative unless they are updated first.

## Commit & Pull Request Guidelines
Write imperative, ≤60-character commit subjects (e.g., `Add Qwen embedding wrapper`). Document the rationale, mention executed sanity checks (`pytest`, benchmark scripts, manual curl), and call out new models or CLI flags in PR descriptions. Keep changes scoped; land benchmark tooling separately from API behavior tweaks when practical.

## Security & Configuration Notes
Do not bake credentials into code or scripts. Encourage contributors to configure `HF_HOME`/`TRANSFORMERS_CACHE` for shared environments and capture new environment toggles in the README. If adding benchmarks that store outputs, ensure they do not leak proprietary data.
