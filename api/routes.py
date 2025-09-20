#!/usr/bin/env python3
"""
API route handlers
"""

import time
import logging
from typing import Dict
from fastapi import APIRouter, HTTPException

from models.base import EmbeddingModelWrapper, RerankModelWrapper
from models.factory import ModelFactory
from .schemas import (
    EmbeddingRequest, EmbeddingResponse, EmbeddingData,
    RerankRequest, RerankResponse, RerankResult,
    ModelsResponse, ModelInfo, HealthResponse
)
from .model_manager import get_or_load_model

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Server start time for uptime calculation
_start_time = time.time()


@router.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "message": "Multi-Model Embedding & Reranking API",
        "version": "1.1.0",
        "supported_models": ModelFactory.get_supported_models()
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from .model_manager import model_registry
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=len(model_registry),
        uptime=time.time() - _start_time
    )


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    models = []
    for model_id, model_type in ModelFactory.get_supported_models().items():
        models.append(ModelInfo(
            id=model_id,
            created=int(time.time()),
            type=model_type
        ))

    return ModelsResponse(
        object="list",
        data=models
    )


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for input text(s)"""
    try:
        # Get model
        model_wrapper = await get_or_load_model(request.model)
        
        if not isinstance(model_wrapper, EmbeddingModelWrapper):
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} is not an embedding model"
            )
        
        # Prepare input
        texts = request.input if isinstance(request.input, list) else [request.input]
        
        # Generate embeddings
        embeddings = await model_wrapper.encode(texts)
        
        # Create response
        embedding_data = []
        for i, embedding in enumerate(embeddings):
            embedding_data.append(EmbeddingData(
                embedding=embedding.tolist(),
                index=i
            ))
        
        # Calculate usage
        total_tokens = sum(len(text.split()) for text in texts)
        
        return EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
        )
        
    except Exception as e:
        logger.error("Error in create_embeddings: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Rerank documents based on query relevance"""
    try:
        # Get model
        model_wrapper = await get_or_load_model(request.model)
        
        if not isinstance(model_wrapper, RerankModelWrapper):
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} is not a rerank model"
            )
        
        # Generate scores
        scores = await model_wrapper.compute_scores(request.query, request.documents)
        
        # Create results with scores and indices
        results_with_scores = [
            (i, score, doc) for i, (score, doc) in enumerate(zip(scores, request.documents))
        ]
        
        # Sort by score (descending)
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k if specified
        if request.top_k:
            results_with_scores = results_with_scores[:request.top_k]
        
        # Create response
        results = []
        for original_index, score, doc in results_with_scores:
            result = RerankResult(
                index=original_index,
                relevance_score=float(score)
            )
            if request.return_documents:
                result.document = doc
            results.append(result)
        
        # Calculate usage
        query_tokens = len(request.query.split())
        doc_tokens = sum(len(doc.split()) for doc in request.documents)
        total_tokens = query_tokens + doc_tokens
        
        return RerankResponse(
            results=results,
            model=request.model,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
        )
        
    except Exception as e:
        logger.error("Error in rerank_documents: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
