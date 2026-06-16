"""Jina-compatible gateway for the vLLM multitask LoRA backend."""

from __future__ import annotations

import base64
import math
import os
import struct
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8017")
MODEL_NAME = os.getenv("MODEL_NAME", "jina-embeddings-v5-text-small")
MODEL_DIMENSIONS = 1024
MATRYOSHKA_DIMENSIONS = {32, 64, 128, 256, 512, 768, 1024}
TASKS = {
    "retrieval": ("retrieval", "document"),
    "retrieval.query": ("retrieval", "query"),
    "retrieval.passage": ("retrieval", "document"),
    "text-matching": ("text-matching", "document"),
    "classification": ("classification", "document"),
    "clustering": ("clustering", "document"),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        base_url=BACKEND_URL,
        timeout=httpx.Timeout(300.0, connect=5.0),
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="Jina Embeddings v5 Text Small", version="3.0.0", lifespan=lifespan)


def api_error(status_code: int, message: str, param: str | None = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": status_code,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> Response:
    detail = exc.detail
    payload = (
        {"error": detail}
        if isinstance(detail, dict)
        else {"error": {"message": str(detail), "code": exc.status_code}}
    )
    return Response(
        content=orjson.dumps(payload),
        status_code=exc.status_code,
        media_type="application/json",
    )


def parse_inputs(raw_input: Any, default_prompt: str) -> list[str]:
    values = raw_input if isinstance(raw_input, list) else [raw_input]
    if not values:
        raise api_error(400, "input must contain at least one item", "input")

    texts = []
    for index, item in enumerate(values):
        prompt_name = default_prompt
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            if "image" in item:
                raise api_error(
                    400,
                    "jina-embeddings-v5-text-small is text-only; use "
                    "jina-embeddings-v5-omni-small for image inputs",
                    f"input[{index}].image",
                )
            text = item.get("text")
            if not isinstance(text, str):
                raise api_error(
                    400,
                    f"input[{index}] must contain a text string",
                    "input",
                )
            prompt_name = item.get("prompt_name", default_prompt)
        else:
            raise api_error(400, f"input[{index}] must be a string or text object")

        if prompt_name not in {"query", "document"}:
            raise api_error(
                400,
                f"input[{index}].prompt_name must be query or document",
                "input",
            )
        if not text:
            raise api_error(400, f"input[{index}] must not be empty", "input")
        prefix = "Query: " if prompt_name == "query" else "Document: "
        texts.append(f"{prefix}{text}")
    return texts


def truncate_and_normalize(payload: bytes, dimensions: int) -> bytes:
    response = orjson.loads(payload)
    if dimensions != MODEL_DIMENSIONS:
        for item in response["data"]:
            vector = item["embedding"][:dimensions]
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                raise RuntimeError("model returned a zero embedding")
            item["embedding"] = [value / norm for value in vector]
    response["model"] = MODEL_NAME
    return orjson.dumps(response)


def encode_base64(payload: bytes) -> bytes:
    response = orjson.loads(payload)
    for item in response["data"]:
        vector = item["embedding"]
        packed = struct.pack(f"<{len(vector)}f", *vector)
        item["embedding"] = base64.b64encode(packed).decode("ascii")
    return orjson.dumps(response)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    try:
        response = await request.app.state.client.get("/health")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="embedding backend is unavailable") from exc
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "backend": "vllm-multilora",
        "attention": "flash-attention-2",
        "tasks": list(TASKS),
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    try:
        body = await request.json()
    except ValueError as exc:
        raise api_error(400, "request body must be valid JSON") from exc

    if not isinstance(body, dict):
        raise api_error(400, "request body must be a JSON object")
    if body.get("model", MODEL_NAME) != MODEL_NAME:
        raise api_error(404, f"model {body.get('model')!r} was not found", "model")
    if "input" not in body:
        raise api_error(400, "input is required", "input")

    task = body.get("task", "retrieval.passage")
    if task not in TASKS:
        raise api_error(400, f"task must be one of: {', '.join(TASKS)}", "task")
    adapter, default_prompt = TASKS[task]

    dimensions = body.get("dimensions", MODEL_DIMENSIONS)
    if dimensions not in MATRYOSHKA_DIMENSIONS:
        supported = ", ".join(str(value) for value in sorted(MATRYOSHKA_DIMENSIONS))
        raise api_error(400, f"dimensions must be one of: {supported}", "dimensions")

    embedding_type: Literal["float", "base64"] = body.get(
        "embedding_type", body.get("encoding_format", "float")
    )
    if embedding_type not in ("float", "base64"):
        raise api_error(400, "only float and base64 embedding types are supported")

    backend_body = {
        "model": adapter,
        "input": parse_inputs(body["input"], default_prompt),
        "encoding_format": "float",
    }
    if body.get("truncate") is True:
        backend_body["truncate_prompt_tokens"] = 32768

    try:
        response = await request.app.state.client.post(
            "/v1/embeddings",
            content=orjson.dumps(backend_body),
            headers={"content-type": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="embedding backend is unavailable") from exc

    payload = response.content
    if response.status_code == 200:
        payload = truncate_and_normalize(payload, dimensions)
        if embedding_type == "base64":
            payload = encode_base64(payload)
    return Response(
        content=payload,
        status_code=response.status_code,
        media_type="application/json",
    )
