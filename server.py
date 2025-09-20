#!/usr/bin/env python3
"""FastAPI application exposing OpenAI-style embedding and rerank APIs."""

import argparse
import logging
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI

from api.routes import router
from api.model_manager import preload_models, unload_all_models

logger = logging.getLogger(__name__)

# Default preload models
DEFAULT_MODELS = [
    "jinaai/jina-embeddings-v4",
    "jinaai/jina-reranker-m0",
]

_preload_targets: List[str] = DEFAULT_MODELS.copy()


def configure_preload(models: List[str] | None) -> None:
    """Override which models are loaded during startup."""
    global _preload_targets
    if models:
        _preload_targets = models
    else:
        _preload_targets = DEFAULT_MODELS.copy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager that preloads configured models."""
    logger.info("Starting API server...")
    logger.info("Preloading models: %s", ", ".join(_preload_targets))
    await preload_models(_preload_targets)
    yield
    logger.info("Shutting down API server...")
    await unload_all_models()


app = FastAPI(
    title="Multi-Model Embedding & Reranking API",
    description="OpenAI-compatible endpoints for embeddings and reranking",
    version="1.1.0",
    lifespan=lifespan,
)
app.include_router(router)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding & Reranking API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Model to preload (repeatable). Overrides default preload list.",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable autoreload (development only)"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level (default: info)",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _run_server(args: argparse.Namespace) -> None:
    configure_preload(args.models)
    _configure_logging(args.log_level)
    logger.info("Launching server on %s:%s", args.host, args.port)
    if args.models:
        logger.info("Overriding preload models: %s", ", ".join(args.models))
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    _run_server(_parse_args())
