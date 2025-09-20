#!/usr/bin/env python3
"""Benchmark embedding throughput for a given model."""

import argparse
import asyncio
import statistics
import time

from models.factory import ModelFactory

SAMPLE_TEXTS = [
    "Machine learning is transforming multiple industries.",
    "FastAPI provides a modern Python web framework for APIs.",
    "Vector databases rely on high-quality embeddings.",
    "Testing ensures that API changes remain backwards compatible.",
    "Transformers models benefit from GPU acceleration.",
]


async def run_benchmark(model_name: str, batch_size: int, repeats: int) -> None:
    model = ModelFactory.create_model(model_name)
    await model.load()

    texts = SAMPLE_TEXTS * batch_size
    durations = []

    for _ in range(repeats):
        start = time.perf_counter()
        await model.encode(texts)
        durations.append(time.perf_counter() - start)

    avg = statistics.mean(durations)
    p95 = statistics.quantiles(durations, n=20)[18] if repeats > 1 else avg

    print(f"Model: {model_name}")
    print(f"Batch size: {len(texts)}")
    print(f"Runs: {repeats}")
    print(f"Average latency: {avg * 1000:.2f} ms")
    print(f"p95 latency: {p95 * 1000:.2f} ms")
    print(f"Throughput: {len(texts) / avg:.2f} texts/sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding benchmark")
    parser.add_argument("--model", required=True, help="Model name to benchmark")
    parser.add_argument("--batch-size", type=int, default=4, help="Multiplier for sample texts")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(args.model, args.batch_size, args.repeats))
