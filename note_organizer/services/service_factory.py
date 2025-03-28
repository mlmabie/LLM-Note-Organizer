"""
Service functions for Note Organizer.

This module provides simple functions to create service instances.
No singletons or global state are maintained.
"""

from typing import Dict, Any, Optional

from note_organizer.core.config import settings
from note_organizer.services.embedding import EmbeddingService
from note_organizer.services.tagging import TaggingService, TaggingConfig, LLMProvider
from note_organizer.services.processor import ProcessorService
from note_organizer.services.search import SearchService


def create_embedding_service() -> EmbeddingService:
    """
    Create a new embedding service instance.
    
    Returns:
        A new EmbeddingService instance.
    """
    return EmbeddingService(
        model_name=settings.embedding.model_name,
        use_cce=settings.embedding.use_cce,
        cache_dir=settings.embedding.cache_dir
    )


def create_tagging_service(embedding_service: Optional[EmbeddingService] = None) -> TaggingService:
    """
    Create a new tagging service instance.
    
    Args:
        embedding_service: Optional EmbeddingService to use. If None, a new one will be created.
    
    Returns:
        A new TaggingService instance.
    """
    # Create config from settings
    config = TaggingConfig(
        llm_provider=getattr(settings, "llm_provider", LLMProvider.NONE),
        confidence_threshold=settings.tagging.confidence_threshold,
        max_tags_per_note=settings.tagging.max_tags_per_note,
        default_tags=settings.tagging.default_tags,
        use_dspy=settings.tagging.use_dspy
    )
    
    # Create service with dependencies
    return TaggingService(
        embedding_service=embedding_service or create_embedding_service(),
        config=config
    )


def create_processor_service() -> ProcessorService:
    """
    Create a new processor service instance.
    
    Returns:
        A new ProcessorService instance.
    """
    return ProcessorService()


def create_search_service(embedding_service: Optional[EmbeddingService] = None) -> SearchService:
    """
    Create a new search service instance.
    
    Args:
        embedding_service: Optional EmbeddingService to use. If None, a new one will be created.
    
    Returns:
        A new SearchService instance.
    """
    return SearchService()


# For backward compatibility - these functions maintain the old interface
# but can be removed in future versions
def get_embedding_service() -> EmbeddingService:
    """Get an embedding service instance (compatibility function)."""
    return create_embedding_service()


def get_tagging_service() -> TaggingService:
    """Get a tagging service instance (compatibility function)."""
    return create_tagging_service()


def get_processor_service() -> ProcessorService:
    """Get a processor service instance (compatibility function)."""
    return create_processor_service()


def get_search_service() -> SearchService:
    """Get a search service instance (compatibility function)."""
    return create_search_service()


def reset_services() -> None:
    """
    No-op compatibility function.
    
    With the functional approach, there's no need to reset services
    as new instances can be created each time.
    """
    pass 