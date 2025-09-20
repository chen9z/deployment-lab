#!/usr/bin/env python3
"""Embedding model implementations."""

import logging
import os
from typing import List, Optional

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

from .base import EmbeddingModelWrapper

logger = logging.getLogger(__name__)

# Disable tqdm progress bars in embedding helpers to avoid noisy logs
os.environ.setdefault("TQDM_DISABLE", "1")


def _stack_embeddings(embeddings: List[torch.Tensor]) -> torch.Tensor:
    """Convert a list (or single tensor) of embeddings to a normalized float32 tensor."""
    if isinstance(embeddings, list):
        tensor = torch.stack([emb.to(torch.float32) for emb in embeddings], dim=0)
    else:
        tensor = embeddings.to(torch.float32).unsqueeze(0)
    norms = torch.linalg.vector_norm(tensor, dim=1, keepdim=True).clamp(min=1e-12)
    return tensor / norms


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


class JinaEmbeddingModel(EmbeddingModelWrapper):
    """Wrapper for legacy Jina embedding models exposing `.encode`."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.max_length = 8192
        self.normalize_embeddings = True

    async def load(self):
        logger.info("Loading Jina embedding model: %s", self.model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model moved to GPU")
            else:
                logger.info("CUDA not available, using CPU")
            self.model.eval()
            logger.info("Successfully loaded Jina embedding model")
        except Exception as exc:
            logger.error("Failed to load Jina embedding model: %s", exc)
            raise

    async def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        try:
            with torch.no_grad():
                embeddings = self.model.encode(texts)

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            else:
                embeddings = np.array(embeddings)

            if self.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
                embeddings = embeddings / norms

            return embeddings
        except Exception as exc:
            logger.error("Error encoding texts: %s", exc)
            raise

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        if self.normalize_embeddings:
            return float(np.dot(embedding1, embedding2))
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def get_embedding_dimension(self) -> Optional[int]:
        if self.model is not None and hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        return None


class JinaV4EmbeddingModel(JinaEmbeddingModel):
    """Adapter-aware wrapper around jinaai/jina-embeddings-v4."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.default_task = "retrieval"
        self.batch_size = 8

    async def load(self):
        await super().load()
        if hasattr(self.model, "task"):
            self.model.task = self.default_task

    async def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        if isinstance(texts, str):
            texts = [texts]
        try:
            with torch.no_grad():
                embeddings = self.model.encode_text(
                    texts,
                    task=self.default_task,
                    batch_size=min(self.batch_size, max(len(texts), 1)),
                )
            tensor = _stack_embeddings(embeddings)
            return tensor.cpu().numpy()
        except Exception as exc:
            logger.error("Error encoding texts with Jina V4 model: %s", exc)
            raise


class JinaCodeEmbeddingModel(JinaEmbeddingModel):
    """Wrapper for jinaai/jina-code-embeddings-1.5b using mean pooling."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.max_length = 4096
        self.batch_size = 4

    async def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        if isinstance(texts, str):
            texts = [texts]

        tokenizer = self.tokenizer
        device = self.model.device if hasattr(self.model, "device") else torch.device(self.device)
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(inputs["input_ids"].shape, device=device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            embeddings = _mean_pool(hidden, attention_mask)
            embeddings = torch.nn.functional.normalize(
                embeddings.to(torch.float32), p=2, dim=1
            )

        return embeddings.cpu().numpy()


class TransformerEmbeddingModel(EmbeddingModelWrapper):
    """Generic transformer encoder with mean-pooling embeddings."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.max_length = 4096
        self.normalize_embeddings = True

    async def load(self):
        logger.info("Loading transformer embedding model: %s", self.model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            self.model = AutoModel.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            if not torch.cuda.is_available():
                self.model = self.model.cpu()
            self.model.eval()
            logger.info("Successfully loaded transformer embedding model")
        except Exception as exc:
            logger.error("Failed to load transformer embedding model: %s", exc)
            raise

    async def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            if not hasattr(outputs, "last_hidden_state"):
                raise RuntimeError("Model output does not contain last_hidden_state")
            hidden = outputs.last_hidden_state
            mask = inputs.get("attention_mask")
            if mask is None:
                mask = torch.ones(hidden.shape[:2], device=device)
            embeddings = _mean_pool(hidden, mask)

        embeddings = embeddings.to(torch.float32)
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def get_embedding_dimension(self) -> Optional[int]:
        if self.model is not None and hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        return None


class QwenEmbeddingModel(TransformerEmbeddingModel):
    """Thin wrapper for Qwen embedding checkpoints using mean pooling."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.max_length = 4096
