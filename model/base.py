#!/usr/bin/env python3
"""
Base model wrapper classes
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    async def load(self):
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """Return model type (embedding/rerank)"""
        pass


class EmbeddingModelWrapper(BaseModelWrapper):
    """Base class for embedding models"""
    
    def get_type(self) -> str:
        return "embedding"
    
    @abstractmethod
    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        pass


class RerankModelWrapper(BaseModelWrapper):
    """Base class for rerank models"""
    
    def get_type(self) -> str:
        return "rerank"
    
    @abstractmethod
    async def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """Compute relevance scores for query-document pairs"""
        pass
