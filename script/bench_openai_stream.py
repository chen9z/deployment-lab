#!/usr/bin/env python3
"""Simple OpenAI-compatible streaming benchmark for local model servers."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer


@dataclass
class RequestResult:
    request_id: int
    ok: bool
    ttft_ms: float | None
    e2e_ms: float | None
    output_tokens: int
    decode_tps: float | None
    error: str | None


def build_prompt(tokenizer, target_tokens: int) -> str:
    unit = "Summarize the tradeoffs of long-context inference in one precise sentence. "
    prompt = unit
    while len(tokenizer.encode(prompt, add_special_tokens=False)) < target_tokens:
      prompt += unit
    return prompt


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((pct / 100.0) * (len(values) - 1))))
    return values[idx]


def run_request(
    client: OpenAI,
    tokenizer,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_id: int,
) -> RequestResult:
    start = time.perf_counter()
    first_token_time = None
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning is None:
                reasoning = getattr(delta, "reasoning", None)
            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(content)
            if reasoning:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                reasoning_chunks.append(reasoning)
        end = time.perf_counter()
        text = "".join(reasoning_chunks) + "".join(chunks)
        output_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        ttft_ms = (first_token_time - start) * 1000 if first_token_time else None
        e2e_ms = (end - start) * 1000
        decode_seconds = max((end - (first_token_time or end)), 1e-6)
        decode_tps = output_tokens / decode_seconds if output_tokens > 0 else 0.0
        return RequestResult(
            request_id=request_id,
            ok=True,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            output_tokens=output_tokens,
            decode_tps=decode_tps,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        return RequestResult(
            request_id=request_id,
            ok=False,
            ttft_ms=None,
            e2e_ms=(end - start) * 1000,
            output_tokens=0,
            decode_tps=None,
            error=str(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--approx-input-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    prompt = build_prompt(tokenizer, args.approx_input_tokens)
    client = OpenAI(base_url=args.base_url.rstrip("/") + "/v1", api_key="dummy")

    results: list[RequestResult] = []
    wall_start = time.perf_counter()
    lock = threading.Lock()

    def wrapped(request_id: int) -> RequestResult:
        result = run_request(
            client=client,
            tokenizer=tokenizer,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            request_id=request_id,
        )
        with lock:
            results.append(result)
        return result

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(wrapped, idx) for idx in range(args.num_prompts)]
        for future in as_completed(futures):
            future.result()

    wall_end = time.perf_counter()
    ok_results = [r for r in results if r.ok]
    ttfts = [r.ttft_ms for r in ok_results if r.ttft_ms is not None]
    e2es = [r.e2e_ms for r in ok_results if r.e2e_ms is not None]
    decode_tps_values = [r.decode_tps for r in ok_results if r.decode_tps is not None]
    total_output_tokens = sum(r.output_tokens for r in ok_results)
    wall_seconds = max(wall_end - wall_start, 1e-6)

    summary = {
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "approx_input_tokens": args.approx_input_tokens,
        "max_tokens": args.max_tokens,
        "succeeded": len(ok_results),
        "failed": len(results) - len(ok_results),
        "mean_ttft_ms": statistics.mean(ttfts) if ttfts else None,
        "p95_ttft_ms": percentile(ttfts, 95) if ttfts else None,
        "mean_e2e_ms": statistics.mean(e2es) if e2es else None,
        "p95_e2e_ms": percentile(e2es, 95) if e2es else None,
        "mean_decode_tps": statistics.mean(decode_tps_values) if decode_tps_values else None,
        "total_output_tokens": total_output_tokens,
        "global_output_throughput_tps": total_output_tokens / wall_seconds,
        "results": [asdict(r) for r in sorted(results, key=lambda x: x.request_id)],
    }

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
