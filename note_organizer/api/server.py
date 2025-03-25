"""
FastAPI server for Note Organizer.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from note_organizer.core.config import settings
from note_organizer.db.database import create_db_and_tables, get_session
from note_organizer.db.models import Note, Tag, TagNoteLink, NoteSection, ProcessingTask
from note_organizer.services.embedding import embedding_service
from note_organizer.services.tagging import tagging_service


# API models
class TagCreate(BaseModel):
    """Data model for creating a tag."""
    
    name: str
    category: Optional[str] = None
    description: Optional[str] = None


class TagResponse(BaseModel):
    """Data model for tag response."""
    
    id: int
    name: str
    category: Optional[str] = None
    description: Optional[str] = None


class NoteCreate(BaseModel):
    """Data model for creating a note."""
    
    title: str
    content: str
    filename: Optional[str] = None
    path: Optional[str] = None
    tags: Optional[List[str]] = None


class NoteUpdate(BaseModel):
    """Data model for updating a note."""
    
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None


class NoteResponse(BaseModel):
    """Data model for note response."""
    
    id: int
    title: str
    filename: str
    path: str
    word_count: int
    has_front_matter: bool
    created_at: str
    updated_at: str
    tags: List[TagResponse]


class TagSuggestion(BaseModel):
    """Data model for tag suggestion."""
    
    tag: str
    confidence: float


class ErrorResponse(BaseModel):
    """Data model for error response."""
    
    detail: str


# Create FastAPI app
app = FastAPI(
    title="Note Organizer API",
    description="API for organizing and managing markdown notes",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.api.cors_origins] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event
@app.on_event("startup")
def on_startup():
    """Create database tables on startup."""
    create_db_and_tables()
    logger.info("Note Organizer API started")


# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Note Organizer API!"}


@app.get("/tags", response_model=List[TagResponse])
async def get_tags(category: Optional[str] = None):
    """Get all tags, optionally filtered by category."""
    with get_session() as session:
        query = session.query(Tag)
        if category:
            query = query.filter(Tag.category == category)
        tags = query.all()
        
        # Convert to response model
        return [
            TagResponse(
                id=tag.id,
                name=tag.name,
                category=tag.category,
                description=tag.description
            )
            for tag in tags
        ]


@app.post("/tags", response_model=TagResponse)
async def create_tag(tag: TagCreate):
    """Create a new tag."""
    with get_session() as session:
        # Check if tag already exists
        existing = session.query(Tag).filter(Tag.name == tag.name).first()
        if existing:
            return TagResponse(
                id=existing.id,
                name=existing.name,
                category=existing.category,
                description=existing.description
            )
        
        # Create new tag
        db_tag = Tag(
            name=tag.name,
            category=tag.category,
            description=tag.description
        )
        session.add(db_tag)
        session.commit()
        session.refresh(db_tag)
        
        return TagResponse(
            id=db_tag.id,
            name=db_tag.name,
            category=db_tag.category,
            description=db_tag.description
        )


@app.get("/notes", response_model=List[NoteResponse])
async def get_notes(tag: Optional[str] = None, limit: int = 100, offset: int = 0):
    """Get all notes, optionally filtered by tag."""
    with get_session() as session:
        if tag:
            # Filter by tag
            notes = session.query(Note).join(
                TagNoteLink
            ).join(
                Tag
            ).filter(
                Tag.name == tag
            ).offset(offset).limit(limit).all()
        else:
            # Get all notes
            notes = session.query(Note).offset(offset).limit(limit).all()
        
        # Convert to response model
        return [
            NoteResponse(
                id=note.id,
                title=note.title,
                filename=note.filename,
                path=note.path,
                word_count=note.word_count,
                has_front_matter=note.has_front_matter,
                created_at=note.created_at.isoformat(),
                updated_at=note.updated_at.isoformat(),
                tags=[
                    TagResponse(
                        id=tag.id,
                        name=tag.name,
                        category=tag.category,
                        description=tag.description
                    )
                    for tag in note.tags
                ]
            )
            for note in notes
        ]


@app.post("/notes", response_model=NoteResponse)
async def create_note(note: NoteCreate, background_tasks: BackgroundTasks):
    """Create a new note."""
    with get_session() as session:
        # Generate hash for content
        content_hash = hashlib.md5(note.content.encode()).hexdigest()
        
        # Check if note already exists with this hash
        existing = session.query(Note).filter(Note.content_hash == content_hash).first()
        if existing:
            raise HTTPException(status_code=400, detail="Note with this content already exists")
        
        # Count words
        word_count = len(note.content.split())
        
        # Check for front matter
        has_front_matter = note.content.startswith("---")
        
        # Set filename if not provided
        filename = note.filename or f"{note.title.lower().replace(' ', '-')}.md"
        
        # Set path if not provided
        path = note.path or os.path.join(settings.notes_dir, filename)
        
        # Create note
        db_note = Note(
            title=note.title,
            filename=filename,
            path=path,
            content_hash=content_hash,
            word_count=word_count,
            has_front_matter=has_front_matter
        )
        session.add(db_note)
        session.flush()
        
        # Add tags if provided
        if note.tags:
            for tag_name in note.tags:
                # Get or create tag
                tag = session.query(Tag).filter(Tag.name == tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    session.add(tag)
                    session.flush()
                
                # Create link
                link = TagNoteLink(tag_id=tag.id, note_id=db_note.id)
                session.add(link)
        
        session.commit()
        session.refresh(db_note)
        
        # Schedule background task to process note sections
        task = ProcessingTask(
            task_type="note_processing",
            note_id=db_note.id,
            status="pending"
        )
        session.add(task)
        session.commit()
        
        # Convert to response model
        return NoteResponse(
            id=db_note.id,
            title=db_note.title,
            filename=db_note.filename,
            path=db_note.path,
            word_count=db_note.word_count,
            has_front_matter=db_note.has_front_matter,
            created_at=db_note.created_at.isoformat(),
            updated_at=db_note.updated_at.isoformat(),
            tags=[
                TagResponse(
                    id=tag.id,
                    name=tag.name,
                    category=tag.category,
                    description=tag.description
                )
                for tag in db_note.tags
            ]
        )


@app.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(note_id: int):
    """Get a specific note by ID."""
    with get_session() as session:
        note = session.query(Note).filter(Note.id == note_id).first()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        
        # Convert to response model
        return NoteResponse(
            id=note.id,
            title=note.title,
            filename=note.filename,
            path=note.path,
            word_count=note.word_count,
            has_front_matter=note.has_front_matter,
            created_at=note.created_at.isoformat(),
            updated_at=note.updated_at.isoformat(),
            tags=[
                TagResponse(
                    id=tag.id,
                    name=tag.name,
                    category=tag.category,
                    description=tag.description
                )
                for tag in note.tags
            ]
        )


@app.put("/notes/{note_id}", response_model=NoteResponse)
async def update_note(note_id: int, note_update: NoteUpdate):
    """Update a note."""
    with get_session() as session:
        # Get the note
        db_note = session.query(Note).filter(Note.id == note_id).first()
        if not db_note:
            raise HTTPException(status_code=404, detail="Note not found")
        
        # Update fields
        if note_update.title:
            db_note.title = note_update.title
        
        if note_update.content:
            # Update content hash
            db_note.content_hash = hashlib.md5(note_update.content.encode()).hexdigest()
            # Update word count
            db_note.word_count = len(note_update.content.split())
            # Update front matter flag
            db_note.has_front_matter = note_update.content.startswith("---")
        
        # Update tags if provided
        if note_update.tags is not None:
            # Remove existing tag links
            session.query(TagNoteLink).filter(TagNoteLink.note_id == note_id).delete()
            
            # Add new tags
            for tag_name in note_update.tags:
                # Get or create tag
                tag = session.query(Tag).filter(Tag.name == tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    session.add(tag)
                    session.flush()
                
                # Create link
                link = TagNoteLink(tag_id=tag.id, note_id=note_id)
                session.add(link)
        
        session.commit()
        session.refresh(db_note)
        
        # Convert to response model
        return NoteResponse(
            id=db_note.id,
            title=db_note.title,
            filename=db_note.filename,
            path=db_note.path,
            word_count=db_note.word_count,
            has_front_matter=db_note.has_front_matter,
            created_at=db_note.created_at.isoformat(),
            updated_at=db_note.updated_at.isoformat(),
            tags=[
                TagResponse(
                    id=tag.id,
                    name=tag.name,
                    category=tag.category,
                    description=tag.description
                )
                for tag in db_note.tags
            ]
        )


@app.delete("/notes/{note_id}", response_model=Dict[str, str])
async def delete_note(note_id: int):
    """Delete a note."""
    with get_session() as session:
        # Get the note
        db_note = session.query(Note).filter(Note.id == note_id).first()
        if not db_note:
            raise HTTPException(status_code=404, detail="Note not found")
        
        # Delete note
        session.delete(db_note)
        session.commit()
        
        return {"detail": "Note deleted successfully"}


@app.post("/notes/{note_id}/suggest-tags", response_model=List[TagSuggestion])
async def suggest_tags(note_id: int):
    """Suggest tags for a note."""
    with get_session() as session:
        # Get the note
        db_note = session.query(Note).filter(Note.id == note_id).first()
        if not db_note:
            raise HTTPException(status_code=404, detail="Note not found")
        
        # Get content from note sections
        sections = session.query(NoteSection).filter(NoteSection.note_id == note_id).order_by(NoteSection.order).all()
        content = "\n\n".join(section.content for section in sections)
        
        if not content:
            # Use actual file if no sections are available
            try:
                with open(db_note.path, "r") as f:
                    content = f.read()
            except Exception:
                raise HTTPException(status_code=500, detail="Failed to read note content")
        
        # Get existing tags
        existing_tags = [tag.name for tag in db_note.tags]
        
        # Generate suggestions
        suggestions = tagging_service.generate_tags_for_note(
            content, db_note.title, existing_tags
        )
        
        # Convert to response model
        return [
            TagSuggestion(tag=tag, confidence=confidence)
            for tag, confidence in suggestions
        ]


@app.post("/upload", response_model=NoteResponse)
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    title: Optional[str] = None,
    auto_tag: bool = True
):
    """Upload a markdown file."""
    # Read file content
    content = await file.read()
    content_str = content.decode("utf-8")
    
    # Generate title if not provided
    if not title:
        # Use filename without extension
        title = os.path.splitext(file.filename)[0]
        # Replace underscores and hyphens with spaces
        title = title.replace("_", " ").replace("-", " ")
        # Capitalize words
        title = " ".join(word.capitalize() for word in title.split())
    
    # Create note
    note_create = NoteCreate(
        title=title,
        content=content_str,
        filename=file.filename
    )
    
    # Create note
    with get_session() as session:
        # Generate hash for content
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        
        # Check if note already exists with this hash
        existing = session.query(Note).filter(Note.content_hash == content_hash).first()
        if existing:
            raise HTTPException(status_code=400, detail="Note with this content already exists")
        
        # Count words
        word_count = len(content_str.split())
        
        # Check for front matter
        has_front_matter = content_str.startswith("---")
        
        # Set path
        path = os.path.join(settings.notes_dir, file.filename)
        
        # Create note
        db_note = Note(
            title=title,
            filename=file.filename,
            path=path,
            content_hash=content_hash,
            word_count=word_count,
            has_front_matter=has_front_matter
        )
        session.add(db_note)
        session.flush()
        
        # Auto-tag if requested
        if auto_tag:
            suggestions = tagging_service.generate_tags_for_note(content_str, title)
            for tag_name, confidence in suggestions:
                # Get or create tag
                tag = session.query(Tag).filter(Tag.name == tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    session.add(tag)
                    session.flush()
                
                # Create link
                link = TagNoteLink(
                    tag_id=tag.id,
                    note_id=db_note.id,
                    confidence=confidence
                )
                session.add(link)
        
        session.commit()
        session.refresh(db_note)
        
        # Schedule background task to process note sections
        task = ProcessingTask(
            task_type="note_processing",
            note_id=db_note.id,
            status="pending"
        )
        session.add(task)
        session.commit()
        
        # Save the file to disk
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        
        # Convert to response model
        return NoteResponse(
            id=db_note.id,
            title=db_note.title,
            filename=db_note.filename,
            path=db_note.path,
            word_count=db_note.word_count,
            has_front_matter=db_note.has_front_matter,
            created_at=db_note.created_at.isoformat(),
            updated_at=db_note.updated_at.isoformat(),
            tags=[
                TagResponse(
                    id=tag.id,
                    name=tag.name,
                    category=tag.category,
                    description=tag.description
                )
                for tag in db_note.tags
            ]
        )


@app.get("/search", response_model=List[NoteResponse])
async def search_notes(query: str, limit: int = 10):
    """Search notes using semantic search."""
    with get_session() as session:
        # Get all notes
        notes = session.query(Note).all()
        
        if not notes:
            return []
        
        # Get note titles and IDs
        note_titles = [f"{note.id}:{note.title}" for note in notes]
        note_ids = [note.id for note in notes]
        
        # Perform semantic search using embeddings
        query_embedding = embedding_service.get_embedding(query, compress=True)
        
        # Get note embeddings
        note_embeddings = []
        for note in notes:
            # Get all sections for this note
            sections = session.query(NoteSection).filter(NoteSection.note_id == note.id).all()
            
            if sections:
                # Use average of section embeddings
                section_embeddings = []
                for section in sections:
                    if section.embedding:
                        try:
                            embedding = embedding_service.decompress_embedding(
                                np.frombuffer(section.embedding, dtype=np.float32)
                            )
                            section_embeddings.append(embedding)
                        except Exception:
                            pass
                
                if section_embeddings:
                    note_embedding = np.mean(section_embeddings, axis=0)
                else:
                    # Generate embedding for title if no sections available
                    note_embedding = embedding_service.get_embedding(note.title, compress=False)
            else:
                # Generate embedding for title
                note_embedding = embedding_service.get_embedding(note.title, compress=False)
                
            note_embeddings.append(note_embedding)
        
        # Find most similar notes
        most_similar = embedding_service.find_most_similar(
            query_embedding, note_embeddings, compressed=True, top_k=limit
        )
        
        # Get result note IDs
        result_ids = [note_ids[idx] for idx, _ in most_similar]
        
        # Get notes in order
        result_notes = []
        for note_id in result_ids:
            note = session.query(Note).filter(Note.id == note_id).first()
            if note:
                result_notes.append(note)
        
        # Convert to response model
        return [
            NoteResponse(
                id=note.id,
                title=note.title,
                filename=note.filename,
                path=note.path,
                word_count=note.word_count,
                has_front_matter=note.has_front_matter,
                created_at=note.created_at.isoformat(),
                updated_at=note.updated_at.isoformat(),
                tags=[
                    TagResponse(
                        id=tag.id,
                        name=tag.name,
                        category=tag.category,
                        description=tag.description
                    )
                    for tag in note.tags
                ]
            )
            for note in result_notes
        ]


def start():
    """Start the FastAPI server."""
    # Create database tables
    create_db_and_tables()
    
    # Start server
    uvicorn.run(
        "note_organizer.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        log_level=settings.log.level.lower(),
    ) 