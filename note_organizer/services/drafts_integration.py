"""
Drafts app integration for Note Organizer.

This module provides utilities and functions to integrate with the Drafts app,
supporting import/export functionality, AI processing, and structured decision making.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from enum import Enum
from datetime import datetime

from loguru import logger
from pydantic import BaseModel, Field, validator

from note_organizer.core.config import settings
from note_organizer.services.tagging import tagging_service, TagWithConfidence
from note_organizer.services.embedding import embedding_service


class NoteAction(str, Enum):
    """Actions that can be performed on a note."""
    TAG = "tag"
    SPLIT = "split"
    MERGE = "merge"
    REFINE = "refine"
    EXPAND = "expand"
    EXTRACT = "extract"


class NotePart(str, Enum):
    """Parts of a note that can be processed."""
    ENTIRE = "entire"
    SELECTION = "selection"
    SECTION = "section"
    FRONTMATTER = "frontmatter"


class DraftsExportFormat(str, Enum):
    """Export formats supported by Drafts."""
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    JSON = "json"


class ImportRequest(BaseModel):
    """Request model for importing a note from Drafts."""
    content: str
    title: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    uuid: Optional[str] = None


class ExportFormat(BaseModel):
    """Export format configuration."""
    format: DraftsExportFormat = DraftsExportFormat.MARKDOWN
    include_frontmatter: bool = True
    include_tags: bool = True
    template: Optional[str] = None


class ProcessingResult(BaseModel):
    """Result of a note processing operation."""
    success: bool
    content: Optional[str] = None
    suggested_tags: List[TagWithConfidence] = Field(default_factory=list)
    message: Optional[str] = None
    actions: List[Dict[str, Any]] = Field(default_factory=list)


class ProcessingRequest(BaseModel):
    """Request model for processing a note with AI."""
    content: str
    action: NoteAction
    part: NotePart = NotePart.ENTIRE
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None
    options: Dict[str, Any] = Field(default_factory=dict)


def _sanitize_filename(title: str) -> str:
    """Sanitizes a string to be safe for use as a filename."""
    # Remove characters not suitable for filenames
    safe_title = re.sub(r'[^\w\\s-]', '', title).strip().lower()
    # Replace whitespace/hyphens with a single hyphen
    safe_title = re.sub(r'[-\\s]+', '-', safe_title)
    # Limit length to avoid issues with long filenames
    return safe_title[:100]


def _generate_frontmatter(title: str, tags: List[str]) -> str:
    """Generates YAML frontmatter string for a note."""
    frontmatter = {
        'title': title,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    if tags:
        frontmatter['tags'] = tags

    # Convert to YAML format
    frontmatter_str = '---\n'
    for key, value in frontmatter.items():
        if isinstance(value, list):
            frontmatter_str += f'{key}:\n'
            for item in value:
                frontmatter_str += f'  - {item}\n'
        else:
            frontmatter_str += f'{key}: {value}\n'
    frontmatter_str += '---\n\n'
    return frontmatter_str


def import_from_drafts(request: ImportRequest) -> Dict[str, Any]:
    """
    Import a note from Drafts into the Note Organizer system.
    
    Args:
        request: Import request containing note data
        
    Returns:
        Dictionary with result information
    """
    try:
        # Generate a filename from title or first line
        title = request.title
        if not title:
            # Extract title from first line or generate one
            first_line = request.content.split('\n', 1)[0].strip()
            title = first_line[:50] if first_line else f"Imported Note {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Sanitize filename using helper
        safe_title = _sanitize_filename(title)
        
        # Prepare file path
        notes_dir = Path(settings.notes_dir)
        file_path = notes_dir / f"{safe_title}.md"
        
        # Ensure directory exists
        notes_dir.mkdir(parents=True, exist_ok=True)
        
        # If file exists, add a timestamp to make it unique
        if file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_path = notes_dir / f"{safe_title}-{timestamp}.md"
        
        # Add frontmatter if not already present
        content = request.content
        if not content.startswith('---'):
            # Generate frontmatter using helper
            frontmatter_str = _generate_frontmatter(title, request.tags)
            # Add frontmatter to content
            content = frontmatter_str + content
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Log the import
        logger.info(f"Imported note from Drafts: {file_path}")
        
        return {
            "success": True,
            "path": str(file_path),
            "title": title,
            "message": f"Note imported successfully as {file_path.name}"
        }
    
    except Exception as e:
        logger.error(f"Error importing note from Drafts: {e}")
        return {
            "success": False,
            "message": f"Failed to import note: {str(e)}"
        }


def process_with_ai(request: ProcessingRequest) -> ProcessingResult:
    """
    Process a note with AI according to the requested action.
    
    Args:
        request: Processing request with action and options
        
    Returns:
        ProcessingResult with AI suggestions and processed content
    """
    try:
        content = request.content
        
        # Extract the part of the content to process
        if request.part == NotePart.SELECTION and request.selection_start is not None and request.selection_end is not None:
            text_to_process = content[request.selection_start:request.selection_end]
        else:
            text_to_process = content
        
        result = ProcessingResult(success=True)
        
        # Process according to the requested action
        if request.action == NoteAction.TAG:
            # Get tag suggestions from tagging service
            suggested_tags = tagging_service.generate_tags_for_content(text_to_process)
            
            # Convert to TagWithConfidence objects
            tags_with_confidence = []
            for tag_name, confidence in suggested_tags:
                tags_with_confidence.append(
                    TagWithConfidence(name=tag_name, confidence=confidence, source="ai")
                )
            
            result.suggested_tags = tags_with_confidence
            result.message = f"Generated {len(tags_with_confidence)} tag suggestions"
            
        elif request.action == NoteAction.SPLIT:
            # Analyze for potential split points
            sections = _analyze_for_split(text_to_process)
            
            # Create suggested actions
            for i, (heading, content) in enumerate(sections):
                if i > 0:  # Skip the first section which is the original
                    result.actions.append({
                        "type": "create_note",
                        "title": heading,
                        "content": content,
                        "tags": ["auto-split"]
                    })
            
            result.message = f"Suggested splitting into {len(sections)} notes"
            
        elif request.action == NoteAction.REFINE:
            # Refine the content with AI
            refined_content = _refine_content(text_to_process, request.options)
            
            result.content = refined_content
            result.message = "Content refined"
            
        elif request.action == NoteAction.EXPAND:
            # Expand the content with AI
            expanded_content = _expand_content(text_to_process, request.options)
            
            result.content = expanded_content
            result.message = "Content expanded"
            
        elif request.action == NoteAction.EXTRACT:
            # Extract entities or information from the content
            extracted_info = _extract_information(text_to_process, request.options)
            
            result.content = extracted_info
            result.message = "Information extracted"
            
        elif request.action == NoteAction.MERGE:
            # This would require additional context like multiple notes
            result.success = False
            result.message = "Merge action requires multiple notes and is not supported through this interface"
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing note with AI: {e}")
        return ProcessingResult(
            success=False,
            message=f"Failed to process note: {str(e)}"
        )


def _analyze_for_split(content: str) -> List[Tuple[str, str]]:
    """
    Analyze content to find natural split points.
    
    Args:
        content: The content to analyze
        
    Returns:
        List of tuples with (heading, content) for potential new notes
    """
    # Placeholder Implementation: Splits based on markdown headers.
    # A more sophisticated implementation could use NLP to find semantic boundaries.
    sections = []
    current_heading = None
    current_content = []
    
    for line in content.split('\n'):
        if line.startswith('#'):
            # If we have accumulated content, add it to sections
            if current_heading is not None:
                sections.append((current_heading, '\n'.join(current_content)))
                
            # Start a new section
            current_heading = line.lstrip('#').strip()
            current_content = [line]
        else:
            # If no heading yet, create a generic one from first line
            if current_heading is None and line.strip():
                current_heading = line.strip()[:50]
                current_content = []
            
            current_content.append(line)
    
    # Add the last section
    if current_heading is not None:
        sections.append((current_heading, '\n'.join(current_content)))
    
    # If no sections were found, return the entire content as one section
    if not sections:
        sections = [("Untitled Note", content)]
    
    return sections


def _refine_content(content: str, options: Dict[str, Any]) -> str:
    """
    Refine content using AI.
    
    Args:
        content: Content to refine
        options: Refinement options
        
    Returns:
        Refined content
    """
    # Placeholder Implementation: In a real system, this would involve
    # calling an external LLM API (e.g., OpenAI, Anthropic) with specific prompts
    # based on the provided 'options'. This introduces external dependencies and side effects.
    logger.info(f"Placeholder: Refining content with options: {options}")
    return f"## Refined Content (AI Placeholder)\\n\\n{content}\\n\\n_Refinement options used: {json.dumps(options)}_\""


def _expand_content(content: str, options: Dict[str, Any]) -> str:
    """
    Expand content using AI.
    
    Args:
        content: Content to expand
        options: Expansion options
        
    Returns:
        Expanded content
    """
    # Placeholder Implementation: Similar to _refine_content, this would call an LLM.
    logger.info(f"Placeholder: Expanding content with options: {options}")
    return f"## Expanded Content (AI Placeholder)\\n\\n{content}\\n\\n_Expansion options used: {json.dumps(options)}_\""


def _extract_information(content: str, options: Dict[str, Any]) -> str:
    """
    Extract information from content using AI.
    
    Args:
        content: Content to extract information from
        options: Extraction options
        
    Returns:
        Extracted information (e.g., summary, key points, entities)
    """
    # Placeholder Implementation: Calls an LLM for extraction tasks.
    logger.info(f"Placeholder: Extracting information with options: {options}")
    # Example extraction type (can be configured via options)
    extraction_type = options.get("type", "summary")
    
    # Construct the response string clearly
    response_string = f"""## Extracted Information (AI Placeholder - Type: {extraction_type})

- Extracted item 1 based on '{extraction_type}'
- Extracted item 2 based on '{extraction_type}'

_Extraction options used: {json.dumps(options)}_"""
    return response_string


def suggest_folder_for_content(content: str, tags: Optional[List[str]] = None, title: Optional[str] = None) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Suggest a folder for organizing a note based on its content and tags.

    Args:
        content: The note content.
        tags: Optional list of tags associated with the note.
        title: Optional title of the note (currently unused).

    Returns:
        A tuple containing:
            - The suggested folder name (str).
            - The confidence score for the suggestion (float).
            - A list of alternative folder suggestions (List[Dict[str, Any]]).
    """
    content = content.lower()
    tags = tags or []

    # Simple keyword-based folder suggestion (Placeholder - replace with actual AI/ML model)
    folder_mapping = {
        "work": ["work", "job", "project", "meeting", "client", "deadline"],
        "personal": ["personal", "family", "home", "life", "shopping"],
        "health": ["health", "fitness", "exercise", "doctor", "medication"],
        "finance": ["money", "finance", "budget", "expense", "investment"],
        "ideas": ["idea", "concept", "brainstorm", "thought"],
        "reference": ["reference", "guide", "manual", "documentation"],
        "journal": ["journal", "diary", "today", "yesterday", "reflection"],
        "projects": ["project", "task", "milestone", "progress"],
    }

    # Count matches for each folder
    folder_scores = {}
    for folder, keywords in folder_mapping.items():
        score = 0
        for keyword in keywords:
            if keyword in content:
                score += 1

        # Also check tags
        for tag in tags:
            if tag.lower() in keywords:
                score += 2
            elif tag.lower() == folder:
                score += 3

        if score > 0:
            folder_scores[folder] = score

    # If no matches, use default
    if not folder_scores:
        return "Inbox", 1.0, []

    # Get the best match and alternatives
    sorted_folders = sorted(folder_scores.items(), key=lambda x: x[1], reverse=True)
    best_folder, best_score = sorted_folders[0]

    # Calculate confidence (simplified)
    total_score = sum(score for _, score in sorted_folders)
    confidence = best_score / total_score if total_score > 0 else 0.5

    # Format alternatives
    alternatives = [
        {"folder": folder, "confidence": score / total_score}
        for folder, score in sorted_folders[1:4]  # Top 3 alternatives
    ]

    return best_folder, confidence, alternatives