#!/usr/bin/env python3
"""Manual smoke test for the Jina V2 multilingual reranker."""

if __name__ != "__main__":
    import pytest
    pytest.skip("Manual script; run directly with python", allow_module_level=True)

import asyncio
import logging

from models.factory import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUERY = "介绍可再生能源的优势"
DOCUMENTS = [
    "可再生能源，例如太阳能和风能，可以减少温室气体排放。",
    "Renewable energy sources like wind and solar reduce carbon emissions and reliance on fossil fuels.",
    "Le tourisme est important pour l'économie locale et crée des emplois saisonniers.",
    "Die Nutzung von Solarenergie kann den Stromverbrauch aus Kohle deutlich senken.",
]


async def main():
    model_name = "jinaai/jina-reranker-v2-base-multilingual"
    logger.info("Loading %s", model_name)
    model = ModelFactory.create_model(model_name)
    await model.load()

    logger.info("Computing scores for %d documents", len(DOCUMENTS))
    scores = await model.compute_scores(QUERY, DOCUMENTS)

    ranked = sorted(zip(scores, DOCUMENTS), reverse=True)
    for rank, (score, doc) in enumerate(ranked, start=1):
        print(f"{rank}. {score:.4f} :: {doc}")


if __name__ == "__main__":
    asyncio.run(main())
