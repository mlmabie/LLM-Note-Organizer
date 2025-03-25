"""
Pydantic models for the API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, validator, HttpUrl


class TagCreate(BaseModel):
    """Data model for creating a tag."""
    
    name: str = Field(..., min_length=1, max_length=50)
    category: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = Field(None, max_length=200)
    
    @validator('name')
    def clean_tag_name(cls, v):
        """Clean tag name: lowercase, remove special chars, etc."""
        import re
        # Convert to lowercase
        v = v.lower()
        # Remove leading hashtags
        v = re.sub(r'^#+', '', v)
        # Remove special characters
        v = re.sub(r'[^\w\-]', '', v)
        return v


class TagResponse(BaseModel):
    """Data model for tag response."""
    
    id: int
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    
    class Config:
        orm_mode = True


class NoteCreate(BaseModel):
    """Data model for creating a note."""
    
    title: str = Field(..., min_length=1, max_length=200)
    content: str
    filename: Optional[str] = Field(None, max_length=255)
    path: Optional[str] = None
    tags: Optional[List[str]] = None
    
    @validator('filename')
    def validate_filename(cls, v, values):
        """Validate and clean filename."""
        import re
        import os
        if v is None:
            # Generate filename from title if not provided
            title = values.get('title', '')
            if title:
                # Convert to lowercase, replace spaces with hyphens
                v = title.lower().replace(' ', '-')
                # Remove special characters
                v = re.sub(r'[^\w\-]', '', v)
                # Add .md extension if not present
                if not v.endswith('.md'):
                    v = f"{v}.md"
            else:
                v = "untitled.md"
        elif not v.endswith(('.md', '.markdown')):
            # Add .md extension if not present
            v = f"{v}.md"
        return v


class NoteUpdate(BaseModel):
    """Data model for updating a note."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = None
    tags: Optional[List[str]] = None


class NoteMetadata(BaseModel):
    """Data model for note metadata."""
    
    id: int
    title: str
    filename: str
    path: str
    word_count: int
    has_front_matter: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class NoteResponse(BaseModel):
    """Data model for note response."""
    
    id: int
    title: str
    filename: str
    path: str
    word_count: int
    has_front_matter: bool
    created_at: datetime
    updated_at: datetime
    tags: List[TagResponse]
    
    class Config:
        orm_mode = True


class TagSuggestion(BaseModel):
    """Data model for tag suggestion."""
    
    tag: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: Optional[str] = None


class SearchQuery(BaseModel):
    """Data model for search query."""
    
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=100)
    include_content: bool = False


class ErrorResponse(BaseModel):
    """Data model for error response."""
    
    detail: str
    status_code: int = 400
    path: Optional[str] = None


class SuccessResponse(BaseModel):
    """Data model for success response."""
    
    detail: str
    status_code: int = 200


class TaskStatus(BaseModel):
    """Data model for task status."""
    
    task_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class TagStats(BaseModel):
    """Data model for tag statistics."""
    
    name: str
    count: int
    category: Optional[str] = None


class NoteSummary(BaseModel):
    """Summary of a note for list views."""
    
    id: int
    title: str
    word_count: int
    tag_count: int
    updated_at: datetime
    
    class Config:
        orm_mode = True 