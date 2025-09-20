#!/usr/bin/env python3
"""Model loading and management."""

import logging
from typing import Dict, Iterable

from models.base import BaseModelWrapper
from models.factory import ModelFactory

logger = logging.getLogger(__name__)

# Global model registry
model_registry: Dict[str, BaseModelWrapper] = {}


def _already_loaded(model_name: str) -> bool:
    return model_name in model_registry


async def get_or_load_model(model_name: str) -> BaseModelWrapper:
    """Return a model wrapper, loading it on first use."""
    if _already_loaded(model_name):
        return model_registry[model_name]

    logger.info("Loading model: %s", model_name)
    model_wrapper = ModelFactory.create_model(model_name)
    await model_wrapper.load()
    model_registry[model_name] = model_wrapper
    logger.info("Loaded model: %s", model_name)
    return model_wrapper


async def preload_models(models: Iterable[str]) -> None:
    """Eagerly load the provided models."""
    for model_name in models:
        try:
            await get_or_load_model(model_name)
        except Exception as exc:
            logger.error("Failed to preload %s: %s", model_name, exc)
            raise


async def unload_model(model_name: str) -> bool:
    """Unload a model from memory."""
    if _already_loaded(model_name):
        del model_registry[model_name]
        logger.info("Unloaded model: %s", model_name)
        return True
    return False


async def unload_all_models() -> None:
    """Clear the registry."""
    count = len(model_registry)
    model_registry.clear()
    logger.info("Unloaded %d models", count)


def get_loaded_models() -> Dict[str, str]:
    """Return a map of loaded model names to their type."""
    return {name: wrapper.get_type() for name, wrapper in model_registry.items()}


def get_model_count() -> int:
    return len(model_registry)
