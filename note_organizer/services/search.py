"""
Search service for Note Organizer.

This module provides search capabilities including both semantic search
and full-text search for notes. It leverages embeddings for semantic similarity
and simple text matching for full-text search.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from loguru import logger
from sqlalchemy import func, or_

from note_organizer.core.config import settings
from note_organizer.db.database import get_session
from note_organizer.db.models import Note, Tag, TagNoteLink, NoteSection
from note_organizer.services.service_factory import create_embedding_service
from note_organizer.services.embedding import EmbeddingService


class SearchService:
    """Service for searching notes."""
    
    def __init__(self, embedding_service=None):
        """
        Initialize SearchService.
        
        Args:
            embedding_service: An optional EmbeddingService instance. If None, a new one will be created.
        """
        self.embedding_service = embedding_service or create_embedding_service()
    
    def semantic_search(
        self, 
        query: str, 
        limit: int = 10, 
        min_similarity: float = 0.0
    ) -> List[Tuple[Note, float]]:
        """
        Perform semantic search on notes.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            min_similarity: Minimum similarity score (0-1)
            
        Returns:
            List of (note, similarity_score) tuples
        """
        with get_session() as session:
            notes = session.query(Note).all()
            
            if not notes:
                return []
            
            # Get all note IDs
            note_ids = [note.id for note in notes]
            
            # Get all note embeddings
            note_embeddings = []
            for note in notes:
                # Get sections for this note
                sections = session.query(NoteSection).filter(NoteSection.note_id == note.id).all()
                
                # If note has sections with embeddings, use them
                if sections:
                    section_embeddings = []
                    for section in sections:
                        if section.embedding:
                            try:
                                embedding = self.embedding_service.decompress_embedding(
                                    section.embedding
                                )
                                section_embeddings.append(embedding)
                            except Exception as e:
                                logger.warning(f"Error decompressing embedding: {e}")
                                continue
                    
                    # Use average of section embeddings
                    if section_embeddings:
                        note_embedding = sum(section_embeddings) / len(section_embeddings)
                    else:
                        # Fallback to title embedding if no section embeddings
                        note_embedding = self.embedding_service.get_embedding(note.title, compress=False)
                else:
                    # Use title embedding if no sections
                    note_embedding = self.embedding_service.get_embedding(note.title, compress=False)
                
                note_embeddings.append(note_embedding)
            
            # Get query embedding
            query_embedding = self.embedding_service.get_embedding(query, compress=False)
            
            # Get most similar notes
            most_similar = self.embedding_service.find_most_similar(
                query_embedding, 
                note_embeddings, 
                compressed=False, 
                top_k=limit
            )
            
            # Filter by minimum similarity
            filtered_similar = [(idx, score) for idx, score in most_similar if score >= min_similarity]
            
            # Get notes and sort by similarity
            results = []
            for idx, score in filtered_similar:
                note_id = note_ids[idx]
                note = session.query(Note).filter(Note.id == note_id).first()
                if note:
                    results.append((note, score))
            
            return results
    
    def full_text_search(
        self, 
        query: str, 
        limit: int = 10,
        search_content: bool = True,
        search_title: bool = True,
        search_tags: bool = True,
        case_sensitive: bool = False
    ) -> List[Tuple[Note, float]]:
        """
        Perform full-text search on notes.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            search_content: Whether to search in note content
            search_title: Whether to search in note titles
            search_tags: Whether to search in note tags
            case_sensitive: Whether the search is case sensitive
            
        Returns:
            List of (note, match_score) tuples, where match_score is always 1.0
            for full-text search
        """
        with get_session() as session:
            # Build base query
            note_query = session.query(Note)
            
            # For database search on title and tags
            if search_title or search_tags:
                filters = []
                
                # Handle case sensitivity
                if case_sensitive:
                    if search_title:
                        filters.append(Note.title.contains(query))
                    
                    if search_tags:
                        # Join with Tags
                        note_query = note_query.join(
                            TagNoteLink
                        ).join(
                            Tag
                        )
                        filters.append(Tag.name.contains(query))
                else:
                    # Case insensitive
                    query_lower = query.lower()
                    
                    if search_title:
                        filters.append(func.lower(Note.title).contains(query_lower))
                    
                    if search_tags:
                        # Join with Tags
                        note_query = note_query.join(
                            TagNoteLink
                        ).join(
                            Tag
                        )
                        filters.append(func.lower(Tag.name).contains(query_lower))
                
                # Apply filters if any
                if filters:
                    note_query = note_query.filter(or_(*filters))
                    
                # Get matches and add to results
                db_matches = note_query.distinct().all()
                
            else:
                # If no database search, get all notes
                db_matches = []
            
            # For content search (requires reading files)
            content_matches = []
            if search_content:
                # Get all notes not already matched
                already_matched_ids = [note.id for note in db_matches]
                all_notes = session.query(Note).filter(Note.id.notin_(already_matched_ids)).all()
                
                for note in all_notes:
                    try:
                        with open(note.path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        # Handle case sensitivity
                        if case_sensitive:
                            if query in content:
                                content_matches.append(note)
                        else:
                            content_lower = content.lower()
                            query_lower = query.lower()
                            if query_lower in content_lower:
                                content_matches.append(note)
                    except Exception as e:
                        logger.warning(f"Error reading file {note.path}: {e}")
            
            # Combine results
            all_matches = list(set(db_matches + content_matches))
            
            # Sort by title for consistent results
            all_matches.sort(key=lambda note: note.title)
            
            # Limit results
            limited_matches = all_matches[:limit]
            
            # Convert to result format
            results = [(note, 1.0) for note in limited_matches]
            
            return results
    
    def search(
        self, 
        query: str, 
        semantic: bool = True,
        limit: int = 10,
        **kwargs
    ) -> List[Tuple[Note, float]]:
        """
        Search notes using either semantic or full-text search.
        
        Args:
            query: The search query
            semantic: Whether to use semantic search (True) or full-text search (False)
            limit: Maximum number of results to return
            **kwargs: Additional arguments for the specific search method
            
        Returns:
            List of (note, score) tuples
        """
        if semantic:
            return self.semantic_search(query, limit, **kwargs)
        else:
            return self.full_text_search(query, limit, **kwargs)
    
    def get_note_context(
        self, 
        note: Note, 
        query: str, 
        context_size: int = 3,
        max_contexts: int = 3,
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get text context around search matches in a note.
        
        Args:
            note: The note to get context from
            query: The search query
            context_size: Number of lines of context on each side
            max_contexts: Maximum number of contexts to return
            case_sensitive: Whether the search is case sensitive
            
        Returns:
            List of context dictionaries with the following keys:
                - line_number: The line number of the match
                - context: The context text
                - snippet: A small snippet with the match highlighted
        """
        try:
            with open(note.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Find matches
            matches = []
            for i, line in enumerate(lines):
                if case_sensitive:
                    if query in line:
                        matches.append(i)
                else:
                    if query.lower() in line.lower():
                        matches.append(i)
            
            # Get context for each match
            contexts = []
            for line_num in matches[:max_contexts]:
                # Get context lines
                start = max(0, line_num - context_size)
                end = min(len(lines), line_num + context_size + 1)
                
                context_lines = lines[start:end]
                context_text = "".join(context_lines).strip()
                
                # Get a small snippet
                match_line = lines[line_num].strip()
                if not case_sensitive:
                    # Find the match position
                    query_lower = query.lower()
                    match_line_lower = match_line.lower()
                    start_pos = match_line_lower.find(query_lower)
                else:
                    start_pos = match_line.find(query)
                
                if start_pos >= 0:
                    # Create snippet with ellipsis if needed
                    snippet_size = 50
                    
                    # Before match
                    if start_pos > snippet_size:
                        before = "..." + match_line[start_pos - snippet_size:start_pos]
                    else:
                        before = match_line[:start_pos]
                    
                    # Match itself
                    if case_sensitive:
                        match_text = match_line[start_pos:start_pos + len(query)]
                    else:
                        match_text = match_line[start_pos:start_pos + len(query)]
                    
                    # After match
                    after_start = start_pos + len(query)
                    if len(match_line) - after_start > snippet_size:
                        after = match_line[after_start:after_start + snippet_size] + "..."
                    else:
                        after = match_line[after_start:]
                    
                    snippet = before + "[" + match_text + "]" + after
                else:
                    # Fallback if match not found (shouldn't happen)
                    snippet = match_line
                
                contexts.append({
                    "line_number": line_num + 1,  # 1-indexed line numbers
                    "context": context_text,
                    "snippet": snippet
                })
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error getting context for note {note.id}: {e}")
            return [] 