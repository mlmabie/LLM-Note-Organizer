"""
Nushell integration for Note Organizer.

This module provides commands and utilities for interacting with Note Organizer
from Nushell (https://www.nushell.sh/), a modern shell designed for the GitHub era.
"""

import os
import sys
import json
import shutil
import subprocess
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

from loguru import logger
from sqlalchemy import func

from note_organizer.core.config import settings
from note_organizer.db.database import get_session
from note_organizer.db.models import Note, Tag, TagNoteLink, NoteSection
from note_organizer.services.service_factory import (
    create_embedding_service, create_tagging_service,
    # Keep old names for backward compatibility
    get_embedding_service, get_tagging_service
)


def is_nushell_available() -> bool:
    """Check if Nushell is available on the system."""
    return shutil.which("nu") is not None


def generate_nu_header() -> str:
    """Generate the header for Nushell commands."""
    return f"""# Note Organizer Nushell Integration
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# This file contains Nushell commands for interacting with Note Organizer.
# To use, source this file in your Nushell config: source path/to/note_organizer.nu

use std log
"""


def generate_notes_command() -> str:
    """Generate the notes command for Nushell."""
    return """
# List all notes
export def "notes list" [
    --tag: string    # Filter by tag
    --limit: int = 100    # Number of notes to return
    --offset: int = 0    # Offset for pagination
] {
    let cmd = ["notes", "list", "--json"]
    if $tag != null { $cmd = ($cmd | append ["--tag", $tag]) }
    if $limit != 100 { $cmd = ($cmd | append ["--limit", $limit]) }
    if $offset != 0 { $cmd = ($cmd | append ["--offset", $offset]) }
    
    note-organizer $cmd | from json
}

# Get a specific note
export def "notes get" [
    id: int     # Note ID
    --content   # Include note content
] {
    mut cmd = ["notes", "get", "--id", $id, "--json"]
    if $content { $cmd = ($cmd | append ["--content"]) }
    
    note-organizer $cmd | from json
}

# Search for notes
export def "notes search" [
    query: string    # Search query
    --limit: int = 10    # Number of results
    --full-text    # Use full-text search instead of semantic
] {
    mut cmd = ["notes", "search", "--query", $query, "--json"]
    if $limit != 10 { $cmd = ($cmd | append ["--limit", $limit]) }
    if $full_text { $cmd = ($cmd | append ["--full-text"]) }
    
    note-organizer $cmd | from json
}

# Suggest tags for a note
export def "notes suggest-tags" [
    id: int    # Note ID
] {
    note-organizer ["notes", "suggest-tags", "--id", $id, "--json"] | from json
}

# Apply tags to a note
export def "notes apply-tags" [
    id: int    # Note ID
    tags: string    # Comma-separated list of tags
] {
    note-organizer ["notes", "apply-tags", "--id", $id, "--tags", $tags]
}
"""


def generate_tags_command() -> str:
    """Generate the tags command for Nushell."""
    return """
# List all tags
export def "tags list" [
    --category: string    # Filter by category
] {
    mut cmd = ["tags", "list", "--json"]
    if $category != null { $cmd = ($cmd | append ["--category", $category]) }
    
    note-organizer $cmd | from json
}

# Create a new tag
export def "tags create" [
    name: string    # Tag name
    --category: string    # Tag category
    --description: string    # Tag description
] {
    mut cmd = ["tags", "create", "--name", $name]
    if $category != null { $cmd = ($cmd | append ["--category", $category]) }
    if $description != null { $cmd = ($cmd | append ["--description", $description]) }
    
    note-organizer $cmd
}
"""


def generate_process_command() -> str:
    """Generate the process command for Nushell."""
    return """
# Process notes
export def "process" [
    --path: string    # Path to notes directory
    --force    # Force reprocessing of all notes
    --recursive    # Process subdirectories recursively
] {
    mut cmd = ["process"]
    if $path != null { $cmd = ($cmd | append ["--path", $path]) }
    if $force { $cmd = ($cmd | append ["--force"]) }
    if $recursive { $cmd = ($cmd | append ["--recursive"]) }
    
    note-organizer $cmd
}
"""


def generate_config_command() -> str:
    """Generate the config command for Nushell."""
    return """
# Show or update configuration
export def "config" [
    --show    # Show current configuration
    --llm: string    # Set LLM provider (none, openai, claude, google)
    --openai-key: string    # Set OpenAI API key
    --claude-key: string    # Set Claude API key
    --google-key: string    # Set Google AI API key
    --notes-dir: string    # Set notes directory
] {
    mut cmd = ["config"]
    if $show { 
        $cmd = ($cmd | append ["--show", "--json"])
        note-organizer $cmd | from json
    } else {
        if $llm != null { $cmd = ($cmd | append ["--llm", $llm]) }
        if $openai_key != null { $cmd = ($cmd | append ["--openai-key", $openai_key]) }
        if $claude_key != null { $cmd = ($cmd | append ["--claude-key", $claude_key]) }
        if $google_key != null { $cmd = ($cmd | append ["--google-key", $google_key]) }
        if $notes_dir != null { $cmd = ($cmd | append ["--notes-dir", $notes_dir]) }
        
        note-organizer $cmd
    }
}
"""


def generate_stats_command() -> str:
    """Generate the stats command for Nushell."""
    return """
# Show statistics
export def "stats" [] {
    note-organizer ["stats", "--json"] | from json
}
"""


def generate_api_command() -> str:
    """Generate the API command for Nushell."""
    return """
# Start API server
export def "api" [
    --host: string    # Host to bind to
    --port: int    # Port to listen on
] {
    mut cmd = ["api"]
    if $host != null { $cmd = ($cmd | append ["--host", $host]) }
    if $port != null { $cmd = ($cmd | append ["--port", $port]) }
    
    note-organizer $cmd
}
"""


def generate_nu_commands() -> str:
    """Generate all Nushell commands."""
    return (
        generate_nu_header() + 
        generate_notes_command() + 
        generate_tags_command() + 
        generate_process_command() + 
        generate_config_command() + 
        generate_stats_command() + 
        generate_api_command()
    )


def export_nu_commands(output_path: str) -> None:
    """Export all Nushell commands to a file."""
    commands = generate_nu_commands()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write commands to file
    with open(output_path, "w") as f:
        f.write(commands)


def generate_notes_json() -> str:
    """Generate JSON representation of all notes."""
    with get_session() as session:
        notes = session.query(Note).all()
        
        result = []
        for note in notes:
            result.append({
                "id": note.id,
                "title": note.title,
                "filename": note.filename,
                "path": note.path,
                "word_count": note.word_count,
                "has_front_matter": note.has_front_matter,
                "created_at": note.created_at.isoformat(),
                "updated_at": note.updated_at.isoformat(),
                "tags": [{"name": tag.name, "category": tag.category} for tag in note.tags]
            })
        
        return json.dumps(result)


def generate_tags_json() -> str:
    """Generate JSON representation of all tags."""
    with get_session() as session:
        tags = session.query(Tag).all()
        
        result = []
        for tag in tags:
            result.append({
                "id": tag.id,
                "name": tag.name,
                "category": tag.category,
                "description": tag.description,
                "note_count": len(tag.notes)
            })
        
        return json.dumps(result)


def generate_search_json(query: str, limit: int = 10) -> str:
    """Generate JSON representation of search results."""
    # Create services
    embedding_service = create_embedding_service()
    
    with get_session() as session:
        notes = session.query(Note).all()
        
        if not notes:
            return json.dumps([])
        
        # Get notes with embeddings
        note_ids = [note.id for note in notes]
        note_embeddings = []
        
        for note in notes:
            sections = session.query(NoteSection).filter(NoteSection.note_id == note.id).all()
            
            if sections:
                section_embeddings = []
                for section in sections:
                    if section.embedding:
                        try:
                            embedding = embedding_service.decompress_embedding(
                                section.embedding
                            )
                            section_embeddings.append(embedding)
                        except Exception:
                            pass
                
                if section_embeddings:
                    note_embedding = sum(section_embeddings) / len(section_embeddings)
                else:
                    note_embedding = embedding_service.get_embedding(note.title, compress=False)
            else:
                note_embedding = embedding_service.get_embedding(note.title, compress=False)
                
            note_embeddings.append(note_embedding)
        
        # Get query embedding
        query_embedding = embedding_service.get_embedding(query, compress=True)
        
        # Find most similar notes
        most_similar = embedding_service.find_most_similar(
            query_embedding, note_embeddings, compressed=True, top_k=limit
        )
        
        # Get result note IDs and scores
        result_ids = [note_ids[idx] for idx, _ in most_similar]
        similarities = [score for _, score in most_similar]
        
        # Get notes
        result = []
        for i, note_id in enumerate(result_ids):
            note = session.query(Note).filter(Note.id == note_id).first()
            if note:
                result.append({
                    "id": note.id,
                    "title": note.title,
                    "filename": note.filename,
                    "path": note.path,
                    "similarity": similarities[i],
                    "similarity_pct": int(similarities[i] * 100),
                    "tags": [{"name": tag.name, "category": tag.category} for tag in note.tags]
                })
        
        return json.dumps(result)


def generate_note_stats() -> str:
    """Generate JSON representation of note statistics."""
    with get_session() as session:
        # Count notes and tags
        note_count = session.query(Note).count()
        tag_count = session.query(Tag).count()
        
        # Get total word count
        total_words = session.query(func.sum(Note.word_count)).scalar() or 0
        
        # Get notes with front matter
        front_matter_count = session.query(Note).filter(Note.has_front_matter).count()
        
        # Get top tags
        tag_counts = session.query(
            Tag.name, func.count(TagNoteLink.note_id)
        ).join(
            TagNoteLink
        ).group_by(
            Tag.id
        ).order_by(
            func.count(TagNoteLink.note_id).desc()
        ).limit(10).all()
        
        # Build result
        result = {
            "note_count": note_count,
            "tag_count": tag_count,
            "total_words": total_words,
            "avg_words": round(total_words / note_count, 1) if note_count > 0 else 0,
            "front_matter_count": front_matter_count,
            "front_matter_pct": round((front_matter_count / note_count) * 100, 1) if note_count > 0 else 0,
            "top_tags": [{"name": name, "count": count} for name, count in tag_counts]
        }
        
        return json.dumps(result)


def suggest_tags_json(note_id: int) -> str:
    """Generate JSON representation of tag suggestions for a note."""
    # Create services
    embedding_service = create_embedding_service()
    tagging_service = create_tagging_service(embedding_service=embedding_service)
    
    with get_session() as session:
        note = session.query(Note).filter(Note.id == note_id).first()
        
        if not note:
            return json.dumps({"error": f"Note with ID {note_id} not found"})
        
        # Get content from note sections
        sections = session.query(NoteSection).filter(NoteSection.note_id == note_id).order_by(NoteSection.order).all()
        content = "\n\n".join(section.content for section in sections)
        
        if not content:
            # Use actual file if no sections are available
            try:
                with open(note.path, "r") as f:
                    content = f.read()
            except Exception as e:
                return json.dumps({"error": f"Failed to read note content: {str(e)}"})
        
        # Get existing tags
        existing_tags = [tag.name for tag in note.tags]
        
        # Generate suggestions
        suggestions = tagging_service.generate_tags_for_note(
            content, note.title, existing_tags
        )
        
        # Build result
        result = {
            "note_id": note_id,
            "title": note.title,
            "existing_tags": existing_tags,
            "suggestions": [
                {"tag": tag, "confidence": confidence, "confidence_pct": int(confidence * 100)}
                for tag, confidence in suggestions
            ]
        }
        
        return json.dumps(result) 