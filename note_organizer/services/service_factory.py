"""
Service initialization module for Note Organizer.

This module provides functions for creating service instances,
but without using a complex factory pattern - just simple functions.
"""

from typing import Optional

from note_organizer.core.config import settings
from note_organizer.services.embedding import EmbeddingService
from note_organizer.services.tagging import TaggingService, TaggingConfig, LLMProvider


# Simple caching of service instances
_embedding_service = None
_tagging_service = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the embedding service instance.
    
    Returns:
        An EmbeddingService instance.
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService(
            model_name=settings.embedding.model_name,
            use_cce=settings.embedding.use_cce,
            cache_dir=settings.embedding.cache_dir
        )
    
    return _embedding_service


def get_tagging_service() -> TaggingService:
    """
    Get the tagging service instance.
    
    Returns:
        A TaggingService instance.
    """
    global _tagging_service
    
    if _tagging_service is None:
        # Create config from settings
        config = TaggingConfig(
            llm_provider=getattr(settings, "llm_provider", LLMProvider.NONE),
            confidence_threshold=settings.tagging.confidence_threshold,
            max_tags_per_note=settings.tagging.max_tags_per_note,
            default_tags=settings.tagging.default_tags,
            use_dspy=settings.tagging.use_dspy
        )
        
        # Create service with dependencies
        _tagging_service = TaggingService(
            embedding_service=get_embedding_service(),
            config=config
        )
    
    return _tagging_service


def reset_services():
    """Reset all service instances."""
    global _embedding_service, _tagging_service
    _embedding_service = None
    _tagging_service = None


# Create instances for import by other modules
embedding_service = get_embedding_service()
tagging_service = get_tagging_service() 