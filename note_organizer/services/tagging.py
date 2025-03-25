"""
Tagging service for note content.

This module provides functions for generating tags for notes,
using a combination of embedding similarity, DSPy, and optional LLM integration
with support for OpenAI, Anthropic Claude, and Google AI APIs.
"""

import hashlib
import json
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Literal
from enum import Enum

import dspy
import numpy as np
import yaml
from loguru import logger
from pydantic import BaseModel, Field, validator

from note_organizer.core.config import settings
from note_organizer.db.database import get_session
from note_organizer.db.models import Tag, Note, TagNoteLink
from note_organizer.services.embedding import embedding_service


# Define Pydantic models for more type safety
class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    GOOGLE = "google"
    NONE = "none"


class TagWithConfidence(BaseModel):
    """A tag with confidence score."""
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "auto"

    @validator('name')
    def clean_tag_name(cls, v):
        """Clean tag name: lowercase, remove special chars, etc."""
        # Convert to lowercase
        v = v.lower()
        # Remove leading hashtags
        v = re.sub(r'^#+', '', v)
        # Remove special characters
        v = re.sub(r'[^\w\-]', '', v)
        return v


class TaggingResult(BaseModel):
    """Result of a tagging operation."""
    tags: List[TagWithConfidence]
    explanation: Optional[str] = None


# Initialize DSPy if configured
if settings.tagging.use_dspy:
    # Setup DSPy with the appropriate LM
    llm_provider = getattr(settings, "llm_provider", LLMProvider.NONE)
    try:
        if llm_provider == LLMProvider.OPENAI and settings.openai_api_key:
            import openai
            openai.api_key = settings.openai_api_key
            dspy_lm = dspy.OpenAI(model=settings.openai.model)
            dspy.settings.configure(lm=dspy_lm)
            DSPY_AVAILABLE = True
        elif llm_provider == LLMProvider.CLAUDE and settings.claude_api_key:
            # Configure Anthropic Claude
            import anthropic
            claude_client = anthropic.Anthropic(api_key=settings.claude_api_key)
            
            # Adapt Claude to DSPy interface
            class ClaudeLM(dspy.LM):
                def __call__(self, prompt, **kwargs):
                    response = claude_client.completions.create(
                        prompt=prompt,
                        model=settings.claude.model,
                        max_tokens_to_sample=kwargs.get("max_tokens", 1000),
                        temperature=kwargs.get("temperature", 0.7)
                    )
                    return [{"text": response.completion}]
            
            dspy_lm = ClaudeLM()
            dspy.settings.configure(lm=dspy_lm)
            DSPY_AVAILABLE = True
        elif llm_provider == LLMProvider.GOOGLE and settings.google_api_key:
            # Configure Google AI
            import google.generativeai as genai
            genai.configure(api_key=settings.google_api_key)
            
            # Adapt Google AI to DSPy interface
            class GoogleLM(dspy.LM):
                def __call__(self, prompt, **kwargs):
                    model = genai.GenerativeModel(settings.google.model)
                    response = model.generate_content(prompt)
                    return [{"text": response.text}]
            
            dspy_lm = GoogleLM()
            dspy.settings.configure(lm=dspy_lm)
            DSPY_AVAILABLE = True
        else:
            # Use a local model if no LLM is configured
            class MockLM(dspy.LM):
                def __call__(self, prompt, **kwargs):
                    # Just return a simple response for demonstration
                    return [{"text": "Sorry, local model not implemented yet"}]
            
            dspy_lm = MockLM()
            dspy.settings.configure(lm=dspy_lm)
            DSPY_AVAILABLE = False
            logger.warning("DSPy initialized with mock LM (no real functionality)")
    except Exception as e:
        logger.warning(f"Failed to initialize DSPy: {e}")
        DSPY_AVAILABLE = False
else:
    DSPY_AVAILABLE = False


# Define DSPy module for tagging
class NoteTagger(dspy.Module):
    """DSPy module for generating tags for notes."""
    
    def __init__(self):
        super().__init__()
        self.generate_tags = dspy.TypedPredictor(
            instruction="Generate relevant tags for a markdown note based on its content.",
            input_types={"title": str, "content": str},
            output_types={"tags": List[str], "explanation": str}
        )
    
    def forward(self, title: str, content: str) -> Dict[str, Any]:
        """Generate tags for a note.
        
        Args:
            title: Note title
            content: Note content
            
        Returns:
            Dictionary with tags and explanation
        """
        return self.generate_tags(title=title, content=content)


class TaggingConfig(BaseModel):
    """Configuration for the tagging service."""
    
    llm_provider: LLMProvider = LLMProvider.NONE
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tags_per_note: int = Field(default=10, ge=1)
    default_tags: List[str] = Field(default_factory=lambda: ["inbox", "needs_processing"])
    use_dspy: bool = True
    
    class Config:
        use_enum_values = True


class TaggingService:
    """Service for generating and managing tags."""
    
    def __init__(
        self,
        embedding_service=None,
        config: Optional[TaggingConfig] = None
    ):
        """Initialize the tagging service."""
        # Use provided config or create from settings
        self.config = config or TaggingConfig(
            llm_provider=getattr(settings, "llm_provider", LLMProvider.NONE),
            confidence_threshold=settings.tagging.confidence_threshold,
            max_tags_per_note=settings.tagging.max_tags_per_note,
            default_tags=settings.tagging.default_tags,
            use_dspy=settings.tagging.use_dspy
        )
        
        # Initialize DSPy module if available
        self.dspy_tagger = NoteTagger() if DSPY_AVAILABLE and self.config.use_dspy else None
        
        # Store embedding service
        self.embedding_service = embedding_service or embedding_service
        
        # Load tag categories from config
        self.tag_categories = settings.tagging.tag_categories
    
    def get_all_tags(self) -> List[Tag]:
        """Get all tags from the database.
        
        Returns:
            List of Tag objects
        """
        with get_session() as session:
            return session.query(Tag).all()
    
    def get_tags_by_category(self, category: str) -> List[Tag]:
        """Get all tags in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of Tag objects
        """
        with get_session() as session:
            return session.query(Tag).filter(Tag.category == category).all()
    
    def create_tag(self, name: str, category: Optional[str] = None, description: Optional[str] = None) -> Tag:
        """Create a new tag.
        
        Args:
            name: Tag name
            category: Optional category
            description: Optional description
            
        Returns:
            Created Tag object
        """
        with get_session() as session:
            # Check if tag already exists
            existing = session.query(Tag).filter(Tag.name == name).first()
            if existing:
                return existing
                
            # Create new tag
            tag = Tag(name=name, category=category, description=description)
            session.add(tag)
            session.commit()
            session.refresh(tag)
            return tag
    
    def generate_tags_for_note(
        self, 
        note_content: str, 
        note_title: str = "",
        existing_tags: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Generate tags for a note.
        
        This method uses multiple strategies:
        1. Rule-based matching with configured categories
        2. DSPy-based tagging if available
        3. LLM tagging if configured
        
        Args:
            note_content: The content of the note
            note_title: The title of the note
            existing_tags: List of existing tags (to avoid duplicates)
            
        Returns:
            List of (tag, confidence) tuples
        """
        # Initialize
        tags_with_confidence: Dict[str, float] = {}
        existing_tags = set(existing_tags or [])
        
        # 1. Apply rule-based tagging from categories
        category_tags = self._rule_based_tagging(note_content)
        for tag in category_tags:
            if tag.name not in existing_tags:
                tags_with_confidence[tag.name] = tag.confidence
        
        # 2. Use DSPy-based tagging if available
        if DSPY_AVAILABLE and self.dspy_tagger:
            dspy_tags = self._dspy_tagging(note_title, note_content)
            for tag in dspy_tags:
                if tag.name not in existing_tags and tag.name not in tags_with_confidence:
                    tags_with_confidence[tag.name] = tag.confidence
        
        # 3. Use LLM tagging if configured and we don't have enough tags yet
        if (self.config.llm_provider != LLMProvider.NONE and 
                len(tags_with_confidence) < self.config.max_tags_per_note):
            llm_tags = self._llm_tagging(note_content)
            for tag in llm_tags:
                if tag.name not in existing_tags and tag.name not in tags_with_confidence:
                    tags_with_confidence[tag.name] = tag.confidence
        
        # 4. Add default tags if we have no tags
        if not tags_with_confidence and not existing_tags:
            for tag_name in self.config.default_tags:
                tags_with_confidence[tag_name] = 1.0
        
        # Convert to list, sort by confidence, and filter by threshold
        result = [
            (tag, conf) for tag, conf in tags_with_confidence.items() 
            if conf >= self.config.confidence_threshold
        ]
        result.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max tags
        return result[:self.config.max_tags_per_note]
    
    def _rule_based_tagging(self, content: str) -> List[TagWithConfidence]:
        """Generate tags based on rule-based matching with configured categories.
        
        Args:
            content: Note content
            
        Returns:
            List of TagWithConfidence objects
        """
        results = []
        
        # Convert to lowercase for case-insensitive matching
        content_lower = content.lower()
        
        # Check each category
        for category, tags in self.tag_categories.items():
            for tag in tags:
                # Simple word boundary matching
                pattern = r'\b' + re.escape(tag.lower()) + r'\b'
                matches = re.findall(pattern, content_lower)
                
                if matches:
                    # Confidence based on number of matches and word count
                    word_count = len(content_lower.split())
                    confidence = min(1.0, (len(matches) / max(1, word_count // 100)) * 2)
                    results.append(TagWithConfidence(
                        name=tag,
                        confidence=confidence,
                        source=f"rule:{category}"
                    ))
        
        return results
    
    def _dspy_tagging(self, title: str, content: str) -> List[TagWithConfidence]:
        """Generate tags using DSPy.
        
        Args:
            title: Note title
            content: Note content
            
        Returns:
            List of TagWithConfidence objects
        """
        if not DSPY_AVAILABLE or not self.dspy_tagger:
            return []
            
        try:
            # Truncate content if too long (DSPy models have context limits)
            max_content_length = 4000
            truncated_content = content[:max_content_length]
            if len(content) > max_content_length:
                truncated_content += "..."
            
            # Generate tags with DSPy
            result = self.dspy_tagger(title=title, content=truncated_content)
            
            # Extract tags and assign confidence based on position
            tags = result.tags
            tag_objects = []
            
            for i, tag in enumerate(tags):
                # Confidence decreases with position (first tags are more relevant)
                confidence = 1.0 - (i * 0.05)
                tag_objects.append(TagWithConfidence(
                    name=tag,
                    confidence=max(0.7, confidence),
                    source="dspy"
                ))
            
            return tag_objects
        except Exception as e:
            logger.error(f"Error generating tags with DSPy: {e}")
            return []
    
    def _llm_tagging(self, content: str) -> List[TagWithConfidence]:
        """Generate tags using configured LLM (OpenAI, Claude, or Google).
        
        Args:
            content: Note content
            
        Returns:
            List of TagWithConfidence objects
        """
        if self.config.llm_provider == LLMProvider.NONE:
            return []
        
        # Truncate content if too long
        max_content_length = 4000
        truncated_content = content[:max_content_length]
        if len(content) > max_content_length:
            truncated_content += "..."
        
        # Prepare prompt
        prompt = f"""
        Please analyze this markdown note content and suggest appropriate tags.
        Extract 3-7 tags that accurately represent the main topics and themes.
        Format your response as a YAML list with tag names in lowercase, without any explanation.
        
        Content:
        {truncated_content}
        """
        
        try:
            if self.config.llm_provider == LLMProvider.OPENAI:
                return self._openai_tagging(prompt)
            elif self.config.llm_provider == LLMProvider.CLAUDE:
                return self._claude_tagging(prompt)
            elif self.config.llm_provider == LLMProvider.GOOGLE:
                return self._google_tagging(prompt)
            else:
                return []
        except Exception as e:
            logger.error(f"Error generating tags with {self.config.llm_provider}: {e}")
            return []
    
    def _openai_tagging(self, prompt: str) -> List[TagWithConfidence]:
        """Generate tags using OpenAI API.
        
        Args:
            prompt: The prompt to send to OpenAI
            
        Returns:
            List of TagWithConfidence objects
        """
        try:
            import openai
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=settings.openai.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts relevant tags from markdown notes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            tags_text = response.choices[0].message.content.strip()
            return self._parse_llm_response(tags_text, "openai")
        except Exception as e:
            logger.error(f"Error generating tags with OpenAI: {e}")
            return []
    
    def _claude_tagging(self, prompt: str) -> List[TagWithConfidence]:
        """Generate tags using Anthropic Claude API.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            List of TagWithConfidence objects
        """
        try:
            import anthropic
            
            claude_client = anthropic.Anthropic(api_key=settings.claude_api_key)
            
            # Add Claude prompt format
            formatted_prompt = f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"
            
            # Call Claude API
            response = claude_client.completions.create(
                prompt=formatted_prompt,
                model=settings.claude.model,
                max_tokens_to_sample=150,
                temperature=0.3
            )
            
            tags_text = response.completion.strip()
            return self._parse_llm_response(tags_text, "claude")
        except Exception as e:
            logger.error(f"Error generating tags with Claude: {e}")
            return []
    
    def _google_tagging(self, prompt: str) -> List[TagWithConfidence]:
        """Generate tags using Google PaLM API.
        
        Args:
            prompt: The prompt to send to Google
            
        Returns:
            List of TagWithConfidence objects
        """
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=settings.google_api_key)
            
            # Call Google API
            model = genai.GenerativeModel(settings.google.model)
            response = model.generate_content(prompt)
            
            tags_text = response.text.strip()
            return self._parse_llm_response(tags_text, "google")
        except Exception as e:
            logger.error(f"Error generating tags with Google: {e}")
            return []
    
    def _parse_llm_response(self, tags_text: str, source: str) -> List[TagWithConfidence]:
        """Parse the response from an LLM into tag objects.
        
        Args:
            tags_text: Text response from the LLM
            source: Source of the tags (e.g., "openai", "claude", "google")
            
        Returns:
            List of TagWithConfidence objects
        """
        try:
            # Try parsing as YAML
            tags = yaml.safe_load(tags_text)
            if isinstance(tags, list):
                # Assign confidence of 0.9 to all tags from LLM
                return [
                    TagWithConfidence(name=tag, confidence=0.9, source=source)
                    for tag in tags if isinstance(tag, str)
                ]
        except Exception:
            pass
        
        # If YAML parsing fails, try to extract tags with regex
        tag_matches = []
        
        # Try finding list items
        list_items = re.findall(r'[-*]\s*"?([a-z0-9_-]+)"?', tags_text)
        if list_items:
            tag_matches.extend(list_items)
        
        # Try finding hashtags
        hashtags = re.findall(r'#([a-z0-9_-]+)', tags_text)
        if hashtags:
            tag_matches.extend(hashtags)
        
        # Try finding quoted words or plain words
        if not tag_matches:
            quoted = re.findall(r'"([a-z0-9_-]+)"', tags_text)
            if quoted:
                tag_matches.extend(quoted)
            else:
                # Just try to find reasonable word-like strings
                words = re.findall(r'\b([a-z0-9_-]{3,})\b', tags_text.lower())
                # Filter out common words that are unlikely to be tags
                stop_words = {"the", "and", "for", "with", "this", "that", "from", "tags", "are"}
                tag_matches.extend([w for w in words if w not in stop_words])
        
        return [
            TagWithConfidence(name=tag, confidence=0.85, source=source)
            for tag in tag_matches
        ]
    
    def apply_tags_to_note(self, note_id: int, tags: List[Union[str, Tuple[str, float], TagWithConfidence]]) -> None:
        """Apply tags to a note in the database.
        
        Args:
            note_id: ID of the note
            tags: List of tag names, (tag_name, confidence) tuples, or TagWithConfidence objects
        """
        with get_session() as session:
            # Get the note
            note = session.query(Note).filter(Note.id == note_id).first()
            if not note:
                logger.error(f"Note with ID {note_id} not found")
                return
            
            # Process tags
            for tag_item in tags:
                if isinstance(tag_item, TagWithConfidence):
                    tag_name = tag_item.name
                    confidence = tag_item.confidence
                    source = tag_item.source
                elif isinstance(tag_item, tuple):
                    tag_name, confidence = tag_item
                    source = "manual"
                else:
                    tag_name, confidence = tag_item, 1.0
                    source = "manual"
                
                # Get or create tag
                tag = session.query(Tag).filter(Tag.name == tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    session.add(tag)
                    session.flush()
                
                # Check if link already exists
                link = session.query(TagNoteLink).filter(
                    TagNoteLink.tag_id == tag.id,
                    TagNoteLink.note_id == note_id
                ).first()
                
                if not link:
                    # Create link
                    link = TagNoteLink(
                        tag_id=tag.id,
                        note_id=note_id,
                        confidence=confidence,
                        source=source
                    )
                    session.add(link)
            
            session.commit()
    
    def suggest_tags_for_notes_without_tags(self, limit: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """Suggest tags for notes that don't have any.
        
        Args:
            limit: Maximum number of notes to process
            
        Returns:
            Dictionary mapping note IDs to suggested tags with confidence
        """
        with get_session() as session:
            # Find notes without tags
            notes_without_tags = session.query(Note).outerjoin(
                TagNoteLink
            ).filter(
                TagNoteLink.note_id == None
            ).limit(limit).all()
            
            # Generate suggestions
            suggestions = {}
            for note in notes_without_tags:
                tags = self.generate_tags_for_note(note.content, note.title)
                suggestions[note.id] = tags
                
            return suggestions


# Singleton instance for reuse
tagging_service = TaggingService() 