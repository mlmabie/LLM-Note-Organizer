"""
Database connection management for Note Organizer.
"""

import contextlib
from typing import Iterator, AsyncIterator

from loguru import logger
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.pool import QueuePool, StaticPool
from sqlalchemy.ext.asyncio import create_async_engine

from note_organizer.core.config import settings

# Create URL for both sync and async engines
sync_url = settings.db.url
async_url = sync_url.replace("sqlite:///", "sqlite+aiosqlite:///")

# Create engines
if sync_url.startswith("sqlite"):
    # For SQLite, use a in-memory mode if it's a memory database, otherwise use file
    is_memory_db = sync_url == "sqlite:///:memory:"
    connect_args = {"check_same_thread": False}
    poolclass = StaticPool if is_memory_db else QueuePool
    
    engine = create_engine(
        sync_url, 
        connect_args=connect_args,
        poolclass=poolclass,
        pool_size=settings.db.min_connections,
        max_overflow=settings.db.max_connections - settings.db.min_connections,
        pool_recycle=3600,
        echo=settings.debug
    )
    
    async_engine = create_async_engine(
        async_url,
        connect_args=connect_args,
        poolclass=poolclass,
        pool_size=settings.db.min_connections,
        max_overflow=settings.db.max_connections - settings.db.min_connections,
        pool_recycle=3600,
        echo=settings.debug
    )
else:
    # For other databases like PostgreSQL, MySQL, etc.
    engine = create_engine(
        sync_url,
        pool_size=settings.db.min_connections,
        max_overflow=settings.db.max_connections - settings.db.min_connections,
        pool_recycle=3600,
        echo=settings.debug
    )
    
    async_engine = create_async_engine(
        async_url,
        pool_size=settings.db.min_connections,
        max_overflow=settings.db.max_connections - settings.db.min_connections,
        pool_recycle=3600,
        echo=settings.debug
    )


def create_db_and_tables() -> None:
    """Create database tables if they don't exist."""
    logger.info("Creating database tables...")
    # Import models to register them with SQLModel
    from note_organizer.db.models import (
        Tag, Note, NoteSection, TagNoteLink, ProcessingTask, EmbeddingCache
    )
    SQLModel.metadata.create_all(engine)
    logger.info("Database tables created!")


@contextlib.contextmanager
def get_session() -> Iterator[Session]:
    """Get a database session for synchronous operations."""
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()


@contextlib.asynccontextmanager
async def get_async_session() -> AsyncIterator[AsyncSession]:
    """Get a database session for asynchronous operations."""
    session = AsyncSession(async_engine)
    try:
        yield session
    finally:
        await session.close()


# In-memory cache for frequent queries
_cache = {}

def get_from_cache(key: str):
    """Get a value from the in-memory cache."""
    if not settings.db.use_cache:
        return None
    return _cache.get(key)

def set_in_cache(key: str, value, ttl: int = None):
    """Set a value in the in-memory cache."""
    if not settings.db.use_cache:
        return
    
    ttl = ttl or settings.db.cache_ttl
    # In a real app, you'd want to handle TTL expiry
    # For this simple implementation, we'll just store the value
    _cache[key] = value

def clear_cache():
    """Clear the in-memory cache."""
    _cache.clear()

def invalidate_cache_key(key: str):
    """Remove a specific key from the cache."""
    if key in _cache:
        del _cache[key] 