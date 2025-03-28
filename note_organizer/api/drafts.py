"""
API endpoints for Drafts app integration.

This module provides API endpoints for integrating with the Drafts app,
including note processing, tagging, and folder suggestions.
"""

from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status, Body
from pydantic import BaseModel

from note_organizer.core.config import settings
from note_organizer.services.drafts_integration import (
    ImportRequest, ProcessingRequest, ProcessingResult, 
    import_from_drafts, process_with_ai
)
from note_organizer.api.auth import get_api_key


router = APIRouter(prefix="/api/v1", tags=["drafts"])


class SuggestFolderRequest(BaseModel):
    """Request model for folder suggestion."""
    content: str
    tags: Optional[List[str]] = None
    title: Optional[str] = None


class SuggestFolderResponse(BaseModel):
    """Response model for folder suggestion."""
    folder: str
    confidence: float
    alternatives: Optional[List[Dict[str, Any]]] = None


@router.post("/import", response_model=Dict[str, Any])
async def import_note(
    request: ImportRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Import a note from Drafts into the Note Organizer system.
    
    Args:
        request: Import request with note content and metadata
        
    Returns:
        Result of the import operation
    """
    result = import_from_drafts(request)
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Failed to import note")
        )
    
    return result


@router.post("/process", response_model=ProcessingResult)
async def process_note(
    request: ProcessingRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Process a note with AI according to the requested action.
    
    Args:
        request: Processing request with note content and action
        
    Returns:
        Processing result with suggestions and processed content
    """
    result = process_with_ai(request)
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.message or "Failed to process note"
        )
    
    return result


@router.post("/suggest_folder", response_model=SuggestFolderResponse)
async def suggest_folder(
    request: SuggestFolderRequest = Body(...),
    api_key: str = Depends(get_api_key)
):
    """
    Suggest a folder for organizing a note based on its content.
    
    Args:
        request: Request with note content and optional tags/title
        
    Returns:
        Suggested folder and alternatives
    """
    # In a real implementation, this would use ML/AI to analyze the content
    # and suggest an appropriate folder.
    # Here's a simplified implementation:
    
    content = request.content.lower()
    tags = request.tags or []
    
    # Simple keyword-based folder suggestion
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
        return SuggestFolderResponse(
            folder="Inbox",
            confidence=1.0,
            alternatives=[]
        )
    
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
    
    return SuggestFolderResponse(
        folder=best_folder,
        confidence=confidence,
        alternatives=alternatives
    ) 