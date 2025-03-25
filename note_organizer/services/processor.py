"""
Note processor service for processing notes in the background.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import re
import hashlib
import time
from datetime import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger
import yaml
from sqlalchemy.orm import Session

from note_organizer.core.config import settings
from note_organizer.db.database import get_session
from note_organizer.db.models import Note, NoteSection, ProcessingTask, Tag, TagNoteLink
from note_organizer.services.embedding import embedding_service
from note_organizer.services.tagging import tagging_service


def extract_sections(content: str) -> List[Tuple[str, str]]:
    """
    Extract sections from markdown content.
    
    Returns a list of tuples (heading, content).
    """
    # Split content by headings
    pattern = r'^(#{1,6})\s+(.+?)$'
    
    # Find all headings and their positions
    headings = []
    for match in re.finditer(pattern, content, re.MULTILINE):
        level = len(match.group(1))
        text = match.group(2).strip()
        position = match.start()
        headings.append((level, text, position))
    
    # Handle case where there are no headings
    if not headings:
        return [("", content)]
    
    # Extract sections
    sections = []
    for i, (level, text, position) in enumerate(headings):
        next_position = headings[i+1][2] if i < len(headings) - 1 else len(content)
        section_content = content[position:next_position].strip()
        sections.append((text, section_content))
    
    return sections


def process_note(note_id: int, force: bool = False) -> bool:
    """
    Process a single note.
    
    Args:
        note_id: ID of the note to process
        force: Force reprocessing even if note has been processed before
        
    Returns:
        True if processing was successful, False otherwise
    """
    with get_session() as session:
        # Get the note
        note = session.query(Note).filter(Note.id == note_id).first()
        if not note:
            logger.error(f"Note with ID {note_id} not found")
            return False
        
        # Check if we need to reprocess
        if not force and note.last_processed:
            # Check if file has been modified since last processing
            file_path = Path(note.path)
            if file_path.exists():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime <= note.last_processed:
                    logger.debug(f"Note {note.title} already processed and not modified since")
                    return True
            else:
                logger.warning(f"Note file not found at {note.path}")
        
        # Read note content
        try:
            with open(note.path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading note file: {e}")
            return False
        
        # Generate hash for content
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Update note info
        note.content_hash = content_hash
        note.word_count = len(content.split())
        note.has_front_matter = content.startswith("---")
        note.last_processed = datetime.now()
        
        # Extract front matter if present
        front_matter = {}
        if note.has_front_matter:
            try:
                # Extract YAML front matter
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    try:
                        front_matter = yaml.safe_load(parts[1])
                        if front_matter and isinstance(front_matter, dict):
                            # Process tags from front matter
                            if "tags" in front_matter and front_matter["tags"]:
                                tags = front_matter["tags"]
                                if isinstance(tags, str):
                                    tags = [tag.strip() for tag in tags.split(",")]
                                elif isinstance(tags, list):
                                    tags = [str(tag).strip() for tag in tags]
                                
                                for tag_name in tags:
                                    if tag_name:
                                        tag = session.query(Tag).filter(Tag.name == tag_name).first()
                                        if not tag:
                                            tag = Tag(name=tag_name, category="frontmatter")
                                            session.add(tag)
                                            session.flush()
                                        
                                        # Check if link already exists
                                        link = session.query(TagNoteLink).filter(
                                            TagNoteLink.tag_id == tag.id,
                                            TagNoteLink.note_id == note.id
                                        ).first()
                                        
                                        if not link:
                                            link = TagNoteLink(
                                                tag_id=tag.id,
                                                note_id=note.id,
                                                source="frontmatter",
                                                confidence=1.0
                                            )
                                            session.add(link)
                    except Exception as e:
                        logger.error(f"Error parsing front matter: {e}")
            except Exception as e:
                logger.error(f"Error processing front matter: {e}")
        
        # Extract and process sections
        sections = extract_sections(content)
        
        # Delete existing sections
        session.query(NoteSection).filter(NoteSection.note_id == note.id).delete()
        
        # Add new sections
        for order, (heading, section_content) in enumerate(sections):
            # Skip very short sections (just the heading)
            if len(section_content.split()) < 5:
                continue
                
            # Generate embedding
            embedding = embedding_service.get_embedding(section_content, compress=True)
            
            # Convert to bytes for storage
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            
            # Create section
            section = NoteSection(
                note_id=note.id,
                heading=heading,
                content=section_content,
                word_count=len(section_content.split()),
                order=order,
                embedding=embedding_bytes
            )
            session.add(section)
        
        # Generate tags if note has few or no tags
        existing_tags = [tag.name for tag in note.tags]
        if len(existing_tags) < 3:
            suggestions = tagging_service.generate_tags_for_note(content, note.title, existing_tags)
            for tag_name, confidence in suggestions:
                if confidence > 0.7:  # Only add high confidence tags automatically
                    tag = session.query(Tag).filter(Tag.name == tag_name).first()
                    if not tag:
                        tag = Tag(name=tag_name, category="auto")
                        session.add(tag)
                        session.flush()
                    
                    # Check if link already exists
                    link = session.query(TagNoteLink).filter(
                        TagNoteLink.tag_id == tag.id,
                        TagNoteLink.note_id == note.id
                    ).first()
                    
                    if not link:
                        link = TagNoteLink(
                            tag_id=tag.id,
                            note_id=note.id,
                            source="auto",
                            confidence=confidence
                        )
                        session.add(link)
        
        # Commit changes
        session.commit()
        logger.info(f"Successfully processed note: {note.title}")
        return True


def _worker_process_note(note_id: int, force: bool = False) -> bool:
    """Worker function for processing notes in parallel."""
    try:
        return process_note(note_id, force)
    except Exception as e:
        logger.error(f"Error in worker thread processing note {note_id}: {e}")
        return False


def process_pending_tasks(max_tasks: int = 50) -> int:
    """
    Process pending tasks from the database.
    
    Args:
        max_tasks: Maximum number of tasks to process
        
    Returns:
        Number of tasks processed
    """
    processed = 0
    with get_session() as session:
        # Get pending tasks
        tasks = session.query(ProcessingTask).filter(
            ProcessingTask.status == "pending"
        ).order_by(
            ProcessingTask.priority.desc(),
            ProcessingTask.created_at
        ).limit(max_tasks).all()
        
        if not tasks:
            return 0
        
        # Create pool with number of workers based on CPU count
        num_workers = min(len(tasks), max(1, multiprocessing.cpu_count() - 1))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Process tasks
            for task in tasks:
                if task.task_type == "note_processing" and task.note_id:
                    # Mark task as in progress
                    task.status = "in_progress"
                    task.started_at = datetime.now()
                    session.commit()
                    
                    # Process note
                    future = executor.submit(_worker_process_note, task.note_id, task.force)
                    
                    # Update task status
                    task.status = "completed" if future.result() else "failed"
                    task.completed_at = datetime.now()
                    session.commit()
                    processed += 1
    
    return processed


def process_notes(directory: str, force: bool = False) -> int:
    """
    Process all markdown files in a directory.
    
    Args:
        directory: Directory to process
        force: Force reprocessing of all notes
        
    Returns:
        Number of notes processed
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        logger.error(f"Directory {directory} does not exist")
        return 0
    
    # Get all markdown files
    markdown_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".md", ".markdown")):
                markdown_files.append(os.path.join(root, file))
    
    if not markdown_files:
        logger.warning(f"No markdown files found in {directory}")
        return 0
    
    logger.info(f"Found {len(markdown_files)} markdown files in {directory}")
    
    # Process files
    processed = 0
    with get_session() as session:
        for file_path in markdown_files:
            # Get relative path from directory
            rel_path = os.path.relpath(file_path, directory)
            
            # Check if note already exists
            note = session.query(Note).filter(Note.path == file_path).first()
            
            if not note:
                # Read file content to create new note
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    continue
                
                # Generate title from filename
                filename = os.path.basename(file_path)
                title = os.path.splitext(filename)[0]
                # Convert kebab-case or snake_case to title case
                title = title.replace("-", " ").replace("_", " ")
                title = " ".join(word.capitalize() for word in title.split())
                
                # Generate hash for content
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                # Create note
                note = Note(
                    title=title,
                    filename=filename,
                    path=file_path,
                    content_hash=content_hash,
                    word_count=len(content.split()),
                    has_front_matter=content.startswith("---")
                )
                session.add(note)
                session.flush()
                
                logger.info(f"Created new note: {title}")
            
            # Create processing task for this note
            task = ProcessingTask(
                task_type="note_processing",
                note_id=note.id,
                status="pending",
                force=force
            )
            session.add(task)
            processed += 1
        
        session.commit()
    
    # Process tasks
    total_processed = process_pending_tasks()
    logger.info(f"Processed {total_processed} notes")
    
    return total_processed


def run_processor(interval: int = 300):
    """
    Run the processor in a loop, checking for new tasks at regular intervals.
    
    Args:
        interval: Interval in seconds between checks for new tasks
    """
    logger.info(f"Starting note processor, checking for tasks every {interval} seconds")
    
    while True:
        try:
            num_processed = process_pending_tasks()
            if num_processed > 0:
                logger.info(f"Processed {num_processed} tasks")
        except Exception as e:
            logger.error(f"Error processing tasks: {e}")
        
        # Sleep until next check
        time.sleep(interval) 