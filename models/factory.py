#!/usr/bin/env python3
"""Factory for creating embedding and rerank model wrappers."""

from typing import Dict, Type

from .base import BaseModelWrapper
from .embedding import (
    JinaV4EmbeddingModel,
    JinaCodeEmbeddingModel,
    QwenEmbeddingModel,
)
from .rerank import JinaRerankModel, JinaV2MultilingualRerankModel


class ModelFactory:
    """Central registry for supported embedding and rerank models."""

    _embedding_models: Dict[str, Type[BaseModelWrapper]] = {
        "jinaai/jina-embeddings-v4": JinaV4EmbeddingModel,
        "jinaai/jina-code-embeddings-1.5b": JinaCodeEmbeddingModel,
        "Qwen/Qwen3-Embedding-4B": QwenEmbeddingModel,
    }

    _rerank_models: Dict[str, Type[BaseModelWrapper]] = {
        "jinaai/jina-reranker-m0": JinaRerankModel,
        "jinaai/jina-reranker-v2-base-multilingual": JinaV2MultilingualRerankModel,
    }

    @classmethod
    def create_model(cls, model_name: str) -> BaseModelWrapper:
        if model_name in cls._embedding_models:
            return cls._embedding_models[model_name](model_name)
        if model_name in cls._rerank_models:
            return cls._rerank_models[model_name](model_name)
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models: {list(cls.get_supported_models().keys())}"
        )

    @classmethod
    def get_supported_models(cls) -> Dict[str, str]:
        models: Dict[str, str] = {}
        models.update({name: "embedding" for name in cls._embedding_models})
        models.update({name: "rerank" for name in cls._rerank_models})
        return models

    @classmethod
    def get_model_type(cls, model_name: str) -> str:
        if model_name in cls._embedding_models:
            return "embedding"
        if model_name in cls._rerank_models:
            return "rerank"
        raise ValueError(f"Model not registered: {model_name}")

    @classmethod
    def get_models_by_type(cls, model_type: str) -> Dict[str, Type[BaseModelWrapper]]:
        if model_type == "embedding":
            return cls._embedding_models.copy()
        if model_type == "rerank":
            return cls._rerank_models.copy()
        raise ValueError(f"Unknown model type: {model_type}")

    @classmethod
    def list_all_models(cls) -> Dict[str, Dict[str, str]]:
        models = {name: {"type": "embedding"} for name in cls._embedding_models}
        models.update({name: {"type": "rerank"} for name in cls._rerank_models})
        return models

    @classmethod
    def get_model_metadata(cls, model_name: str) -> Dict[str, str]:
        return {"type": cls.get_model_type(model_name)}
