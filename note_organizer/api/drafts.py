"""
API endpoints for Drafts app integration.

This module provides API endpoints for integrating with the Drafts app,
including note processing, tagging, and folder suggestions.
"""
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, status, Body
from pydantic import BaseModel

from note_organizer.core.config import settings
from note_organizer.services.drafts_integration import (
    ImportRequest, ProcessingRequest, ProcessingResult, 
    import_from_drafts, process_with_ai,
    suggest_folder_for_content
)
from loguru import logger

load_dotenv()
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
    api_key = os.getenv("ANTHROPIC_API_KEY")
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
    api_key = os.getenv("ANTHROPIC_API_KEY")
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
    api_key = os.getenv("ANTHROPIC_API_KEY")
):
    """
    Suggest a folder for organizing a note based on its content.
    
    Args:
        request: Request with note content and optional tags/title
        
    Returns:
        Suggested folder and alternatives
    """
    # Delegate the core logic to the service layer
    try:
        folder, confidence, alternatives = suggest_folder_for_content(
            content=request.content,
            tags=request.tags,
            title=request.title
        )

        return SuggestFolderResponse(
            folder=folder,
            confidence=confidence,
            alternatives=alternatives
        )
    except Exception as e:
        # Log the error
        logger.error(f"Error suggesting folder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while suggesting the folder."
        ) 