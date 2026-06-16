#!/usr/bin/env python3
"""Benchmark an OpenAI/Jina-compatible embedding HTTP endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request

TEXT = (
    "Vector search systems use normalized multilingual embeddings for efficient "
    "semantic retrieval. "
) * 4


def request_once(url: str, model: str, task: str, batch_size: int) -> float:
    payload = json.dumps(
        {
            "model": model,
            "task": task,
            "input": [f"{TEXT} {index}" for index in range(batch_size)],
        },
        separators=(",", ":"),
    ).encode()
    request = urllib.request.Request(
        url,
        payload,
        {"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        response.read()
    return time.perf_counter() - start


def percentile(values: list[float], ratio: float) -> float:
    return sorted(values)[max(0, int(len(values) * ratio) - 1)]


def benchmark_batch(
    url: str, model: str, task: str, batch_size: int, repeats: int
) -> dict[str, float | int]:
    for _ in range(3):
        request_once(url, model, task, batch_size)
    durations = [
        request_once(url, model, task, batch_size) for _ in range(repeats)
    ]
    average = statistics.mean(durations)
    return {
        "runs": repeats,
        "avg_ms": average * 1000,
        "p50_ms": statistics.median(durations) * 1000,
        "p95_ms": percentile(durations, 0.95) * 1000,
        "texts_per_second": batch_size / average,
    }


def benchmark_concurrency(
    url: str, model: str, task: str, workers: int, rounds: int
) -> dict[str, float | int]:
    def run_round() -> tuple[float, list[float]]:
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            latencies = list(
                executor.map(
                    lambda _: request_once(url, model, task, 1),
                    range(workers),
                )
            )
        return time.perf_counter() - start, latencies

    for _ in range(2):
        run_round()
    samples = [run_round() for _ in range(rounds)]
    wall_times = [sample[0] for sample in samples]
    request_times = [value for sample in samples for value in sample[1]]
    average_wall = statistics.mean(wall_times)
    return {
        "requests": workers * rounds,
        "wall_avg_ms": average_wall * 1000,
        "request_p95_ms": percentile(request_times, 0.95) * 1000,
        "texts_per_second": workers / average_wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8016/v1/embeddings")
    parser.add_argument("--model", default="jina-embeddings-v5-text-small")
    parser.add_argument("--task", default="retrieval.passage")
    parser.add_argument("--output")
    args = parser.parse_args()

    results = {}
    for batch_size, repeats in ((1, 20), (8, 15), (32, 10), (64, 8)):
        results[f"batch_{batch_size}"] = benchmark_batch(
            args.url, args.model, args.task, batch_size, repeats
        )
    for workers in (4, 16):
        results[f"concurrency_{workers}"] = benchmark_concurrency(
            args.url, args.model, args.task, workers, 8
        )

    output = json.dumps(results, indent=2) + "\n"
    print(output, end="")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(output)


if __name__ == "__main__":
    main()
