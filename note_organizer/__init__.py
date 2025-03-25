"""
Note Organizer: A powerful note organization system for markdown-based notes.

This package provides tools for organizing markdown notes using semantic search,
automatic tagging, and efficient embedding techniques.
"""

__version__ = "0.1.0"
__author__ = "Note Organizer Team"
__email__ = "example@example.com"

# Import key components to make them available at the package level
from note_organizer.core.config import settings, load_config
from note_organizer.services.embedding import embedding_service
from note_organizer.services.tagging import tagging_service

__all__ = [
    "settings",
    "load_config",
    "embedding_service",
    "tagging_service",
]

from loguru import logger
import os
import sys

# Configure logger
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Set required environment variables if not already set
if "CCE_CACHE_DIR" not in os.environ:
    cache_dir = os.path.join(os.path.expanduser("~"), ".note_organizer", "cce_cache")
    os.environ["CCE_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True) 