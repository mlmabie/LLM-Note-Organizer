"""
Initialize service instances.
"""

# Import service instances to make them available globally
from note_organizer.services.embedding import embedding_service
from note_organizer.services.tagging import tagging_service

# Export service instances
__all__ = ["embedding_service", "tagging_service"]
