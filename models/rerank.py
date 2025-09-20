#!/usr/bin/env python3
"""Rerank model implementations."""

import logging
from typing import List

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .base import RerankModelWrapper

logger = logging.getLogger(__name__)


class JinaRerankModel(RerankModelWrapper):
    """Wrapper for Jina reranker models using the transformers compute_score helper."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.max_length = 1024
        self.batch_size = 128

    async def load(self):
        logger.info("Loading rerank model: %s", self.model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            if torch.cuda.is_available():
                logger.info("Model loaded on GPU")
            else:
                self.model = self.model.cpu()
                logger.info("CUDA not available, running on CPU")

            self.model.eval()
        except Exception as exc:
            logger.error("Failed to load rerank model: %s", exc)
            raise

    async def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """Compute relevance scores for a query against candidate documents."""
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]

        with torch.no_grad():
            scores = self.model.compute_score(
                pairs,
                max_length=self.max_length,
                doc_type="text",
                batch_size=self.batch_size,
            )

        if isinstance(scores, torch.Tensor):
            return scores.cpu().tolist()
        if isinstance(scores, list):
            return [float(score) for score in scores]
        return [float(scores)]


class JinaV2MultilingualRerankModel(RerankModelWrapper):
    """Wrapper for the multilingual Jina V2 reranker."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.max_length = 1024

    async def load(self):
        logger.info("Loading multilingual rerank model: %s", self.model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            if torch.cuda.is_available():
                logger.info("Model loaded on GPU")
            else:
                self.model = self.model.cpu()
                logger.info("CUDA not available, running on CPU")

            self.model.eval()
        except Exception as exc:
            logger.error("Failed to load multilingual rerank model: %s", exc)
            raise

    async def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """Compute relevance scores with the multilingual reranker."""
        if not documents:
            return []

        sentence_pairs = [[query, doc] for doc in documents]

        with torch.no_grad():
            scores = self.model.compute_score(
                sentence_pairs,
                max_length=self.max_length,
            )

        if isinstance(scores, torch.Tensor):
            return scores.cpu().tolist()
        if isinstance(scores, list):
            return [float(score) for score in scores]
        return [float(scores)]
