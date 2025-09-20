#!/usr/bin/env python3
"""
API request/response schemas
"""

from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request schema for embeddings endpoint"""
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    model: str = Field(..., description="Model name to use for embedding")
    encoding_format: Optional[str] = Field("float", description="Encoding format")
    dimensions: Optional[int] = Field(None, description="Number of dimensions (if supported)")


class EmbeddingData(BaseModel):
    """Single embedding data item"""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Response schema for embeddings endpoint"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]


class RerankRequest(BaseModel):
    """Request schema for rerank endpoint"""
    model: str = Field(..., description="Model name to use for reranking")
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., description="List of documents to rerank")
    top_k: Optional[int] = Field(None, description="Number of top results to return")
    return_documents: Optional[bool] = Field(True, description="Whether to return document text")


class RerankResult(BaseModel):
    """Single rerank result item"""
    index: int
    relevance_score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    """Response schema for rerank endpoint"""
    object: str = "list"
    results: List[RerankResult]
    model: str
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    """Model information schema"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"
    type: str


class ModelsResponse(BaseModel):
    """Response schema for models endpoint"""
    object: str = "list"
    data: List[ModelInfo]


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: Dict[str, Union[str, int]]


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    version: str
    models_loaded: int
    uptime: float
