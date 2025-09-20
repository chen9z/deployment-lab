#!/usr/bin/env python3
"""Benchmark rerank latency for a given model."""

import argparse
import asyncio
import statistics
import time

from models.factory import ModelFactory

QUERY = "How can we reduce cloud infrastructure costs?"
DOCUMENTS = [
    "Implement autoscaling policies to match demand peaks.",
    "Use spot instances for stateless workloads to leverage discounts.",
    "Adopt reserved instances for predictable baseline usage.",
    "Optimize container density and right-size virtual machines.",
    "Introduce CI/CD pipelines to automate deployments.",
    "Leverage managed databases to offload maintenance tasks.",
    "Enable detailed billing exports to monitor spend trends.",
    "Consider multi-cloud redundancy for critical services.",
]


async def run_benchmark(model_name: str, repeats: int) -> None:
    model = ModelFactory.create_model(model_name)
    await model.load()

    durations = []

    for _ in range(repeats):
        start = time.perf_counter()
        await model.compute_scores(QUERY, DOCUMENTS)
        durations.append(time.perf_counter() - start)

    avg = statistics.mean(durations)
    p95 = statistics.quantiles(durations, n=20)[18] if repeats > 1 else avg

    print(f"Model: {model_name}")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Runs: {repeats}")
    print(f"Average latency: {avg * 1000:.2f} ms")
    print(f"p95 latency: {p95 * 1000:.2f} ms")
    print(f"Throughput: {1 / avg:.2f} queries/sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank benchmark")
    parser.add_argument("--model", required=True, help="Model name to benchmark")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(args.model, args.repeats))
