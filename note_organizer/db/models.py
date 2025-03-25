"""
Database models for Note Organizer.
"""

from datetime import datetime
from typing import List, Optional, Set

from sqlmodel import Field, Relationship, SQLModel


class TagNoteLink(SQLModel, table=True):
    """Link table between Tag and Note."""

    tag_id: Optional[int] = Field(default=None, foreign_key="tag.id", primary_key=True)
    note_id: Optional[int] = Field(default=None, foreign_key="note.id", primary_key=True)
    confidence: float = Field(default=1.0)  # Confidence score (0-1) for auto-tagged items


class Tag(SQLModel, table=True):
    """Tag model."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    category: Optional[str] = Field(default=None, index=True)
    description: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    notes: List["Note"] = Relationship(back_populates="tags", link_model=TagNoteLink)


class Note(SQLModel, table=True):
    """Note model."""

    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    title: str = Field(index=True)
    content_hash: str = Field(unique=True, index=True)  # Hash of note content for change detection
    path: str = Field()
    word_count: int = Field(default=0)
    has_front_matter: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_processed_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    tags: List[Tag] = Relationship(back_populates="notes", link_model=TagNoteLink)
    sections: List["NoteSection"] = Relationship(back_populates="note")
    
    # If this note was split from another note
    parent_id: Optional[int] = Field(default=None, foreign_key="note.id")
    part_number: Optional[int] = Field(default=None)
    total_parts: Optional[int] = Field(default=None)


class NoteSection(SQLModel, table=True):
    """Note section model."""

    id: Optional[int] = Field(default=None, primary_key=True)
    note_id: int = Field(foreign_key="note.id")
    content: str = Field()
    order: int = Field()  # Order within the note
    heading: Optional[str] = Field(default=None)
    embedding: Optional[bytes] = Field(default=None)  # Serialized embedding vector
    word_count: int = Field(default=0)
    char_count: int = Field(default=0)
    
    # Relationships
    note: Note = Relationship(back_populates="sections")


class ProcessingTask(SQLModel, table=True):
    """Background processing task model."""

    id: Optional[int] = Field(default=None, primary_key=True)
    task_type: str = Field(index=True)  # e.g., "tag_generation", "note_splitting", etc.
    note_id: Optional[int] = Field(default=None, foreign_key="note.id", index=True)
    status: str = Field(default="pending", index=True)  # pending, processing, completed, failed
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    result: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)


class EmbeddingCache(SQLModel, table=True):
    """Cache for embeddings to avoid recomputing them."""

    id: Optional[int] = Field(default=None, primary_key=True)
    text_hash: str = Field(unique=True, index=True)  # Hash of the input text
    embedding: bytes = Field()  # Serialized embedding vector
    model_name: str = Field(index=True)  # Name of the embedding model used
    created_at: datetime = Field(default_factory=datetime.utcnow) 