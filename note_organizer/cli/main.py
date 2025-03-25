"""
Command-line interface for Note Organizer.

This module provides a comprehensive CLI for interacting with the Note Organizer system.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from loguru import logger
from sqlalchemy import func

from note_organizer.core.config import settings, load_config, update_settings
from note_organizer.db.database import get_session, create_db_and_tables
from note_organizer.db.models import Note, Tag, TagNoteLink, NoteSection
from note_organizer.services.service_factory import (
    create_embedding_service, create_tagging_service, create_search_service, 
    # Keep the old names for backward compatibility
    get_embedding_service, get_tagging_service, get_search_service, 
    reset_services
)
from note_organizer.services.processor import process_notes, process_note
from note_organizer.api.server import start as start_api
from note_organizer.cli.nushell import (
    is_nushell_available, export_nu_commands, 
    generate_notes_json, generate_tags_json, generate_search_json,
    generate_note_stats, suggest_tags_json
)


def setup_cli():
    """Set up command-line interface."""
    parser = argparse.ArgumentParser(
        description="Note Organizer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  note-organizer init                              # Initialize the system
  note-organizer process                           # Process all notes in configured directory
  note-organizer notes list                        # List all notes
  note-organizer tags list                         # List all tags
  note-organizer notes search --query "python"     # Search for notes about Python
  note-organizer api                               # Start the API server
  note-organizer stats                             # Show statistics about your notes
"""
    )
    
    # Global arguments
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add Nushell commands
    nushell_parser = subparsers.add_parser("nushell", help="Nushell integration")
    nushell_parser.add_argument("--install", action="store_true", help="Install Nushell integration")
    nushell_parser.add_argument("--path", type=str, help="Path to save Nushell commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize note organizer")
    init_parser.add_argument("--llm", choices=["none", "openai", "claude", "google"], 
                             default="none", help="Choose LLM provider")
    init_parser.add_argument("--notes-dir", type=str, help="Set notes directory")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configure settings")
    config_parser.add_argument("--llm", choices=["none", "openai", "claude", "google"], 
                             help="Set LLM provider")
    config_parser.add_argument("--openai-key", help="Set OpenAI API key")
    config_parser.add_argument("--claude-key", help="Set Claude API key")
    config_parser.add_argument("--google-key", help="Set Google AI API key")
    config_parser.add_argument("--notes-dir", help="Set notes directory")
    config_parser.add_argument("--show", action="store_true", help="Show current config")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process notes")
    process_parser.add_argument("--path", type=str, help="Path to notes directory")
    process_parser.add_argument("--force", action="store_true", help="Force reprocessing of all notes")
    process_parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument("--host", type=str, help="Host to bind to")
    api_parser.add_argument("--port", type=int, help="Port to listen on")
    
    # Notes command group
    notes_parser = subparsers.add_parser("notes", help="Note operations")
    notes_subparsers = notes_parser.add_subparsers(dest="notes_command", help="Note command")
    
    # Notes list
    list_parser = notes_subparsers.add_parser("list", help="List notes")
    list_parser.add_argument("--tag", type=str, help="Filter by tag")
    list_parser.add_argument("--limit", type=int, default=100, help="Number of notes to return")
    list_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    
    # Notes get
    get_parser = notes_subparsers.add_parser("get", help="Get note details")
    get_parser.add_argument("--id", type=int, required=True, help="Note ID")
    get_parser.add_argument("--content", action="store_true", help="Include note content")
    
    # Notes search
    search_parser = notes_subparsers.add_parser("search", help="Search notes")
    search_parser.add_argument("--query", type=str, required=True, help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Number of results")
    search_parser.add_argument("--full-text", action="store_true", help="Use full-text search instead of semantic")
    
    # Notes suggest-tags
    suggest_parser = notes_subparsers.add_parser("suggest-tags", help="Suggest tags for a note")
    suggest_parser.add_argument("--id", type=int, required=True, help="Note ID")
    
    # Notes apply-tags
    apply_parser = notes_subparsers.add_parser("apply-tags", help="Apply tags to a note")
    apply_parser.add_argument("--id", type=int, required=True, help="Note ID")
    apply_parser.add_argument("--tags", type=str, required=True, help="Comma-separated list of tags")
    
    # Tags command group
    tags_parser = subparsers.add_parser("tags", help="Tag operations")
    tags_subparsers = tags_parser.add_subparsers(dest="tags_command", help="Tag command")
    
    # Tags list
    tags_list_parser = tags_subparsers.add_parser("list", help="List tags")
    tags_list_parser.add_argument("--category", type=str, help="Filter by category")
    
    # Tags create
    tags_create_parser = tags_subparsers.add_parser("create", help="Create a tag")
    tags_create_parser.add_argument("--name", type=str, required=True, help="Tag name")
    tags_create_parser.add_argument("--category", type=str, help="Tag category")
    tags_create_parser.add_argument("--description", type=str, help="Tag description")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    
    return parser


def handle_nushell_command(args):
    """Handle Nushell integration command."""
    if not is_nushell_available():
        print("Nushell is not installed or not available in PATH.")
        print("Please install Nushell from https://www.nushell.sh/")
        return
    
    if args.install:
        path = args.path or os.path.expanduser("~/.config/nushell/note_organizer.nu")
        export_nu_commands(path)
        print(f"Nushell commands installed to {path}")
        print(f"Add 'source {path}' to your Nushell config to enable them.")
    else:
        print("To install Nushell integration, use --install.")


def handle_init_command(args, config_path):
    """Handle init command."""
    logger.info("Initializing Note Organizer")
    
    # Load config
    config = load_config(config_path)
    
    # Update settings based on args
    if args.llm:
        config.llm_provider = args.llm
    
    if args.notes_dir:
        config.notes_dir = args.notes_dir
    
    # Save updated config
    config_dict = config.dict()
    
    # Handle SecretStr values
    for provider in ["openai", "claude", "google"]:
        if config_dict[provider]["api_key"] is not None:
            config_dict[provider]["api_key"] = config_dict[provider]["api_key"].get_secret_value()
    
    with open(config_path, "w") as f:
        import yaml
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Update global settings
    update_settings(config)
    
    # Create database tables
    create_db_and_tables()
    
    # Create notes directory if it doesn't exist
    os.makedirs(config.notes_dir, exist_ok=True)
    
    print(f"Note Organizer initialized")
    print(f"Notes directory: {config.notes_dir}")
    print(f"LLM provider: {config.llm_provider}")
    print(f"Database: {config.db.url}")


def handle_config_command(args, config_path):
    """Handle config command."""
    # Load config
    config = load_config(config_path)
    
    if args.show:
        # Show current config
        config_dict = config.dict()
        # Hide API keys
        for provider in ["openai", "claude", "google"]:
            if config_dict[provider]["api_key"] is not None:
                config_dict[provider]["api_key"] = "********"
        
        if args.json:
            print(json.dumps(config_dict, indent=2))
        else:
            import yaml
            print(yaml.dump(config_dict, default_flow_style=False))
        return
    
    # Update configuration based on args
    changed = False
    
    if args.llm:
        config.llm_provider = args.llm
        changed = True
        
    if args.openai_key:
        from pydantic import SecretStr
        config.openai.api_key = SecretStr(args.openai_key)
        changed = True
        
    if args.claude_key:
        from pydantic import SecretStr
        config.claude.api_key = SecretStr(args.claude_key)
        changed = True
        
    if args.google_key:
        from pydantic import SecretStr
        config.google.api_key = SecretStr(args.google_key)
        changed = True
        
    if args.notes_dir:
        config.notes_dir = args.notes_dir
        changed = True
        
    if changed:
        # Save updated config
        config_dict = config.dict()
        
        # Handle SecretStr values
        for provider in ["openai", "claude", "google"]:
            if config_dict[provider]["api_key"] is not None:
                config_dict[provider]["api_key"] = config_dict[provider]["api_key"].get_secret_value()
        
        with open(config_path, "w") as f:
            import yaml
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # Update global settings
        update_settings(config)
        
        # Reset service singletons to use new settings
        reset_services()
        
        print(f"Configuration updated in {config_path}")
    else:
        print(f"No configuration changes specified")


def handle_process_command(args):
    """Handle process command."""
    path = args.path or settings.notes_dir
    
    if args.recursive:
        # Already recursive by default
        pass
    
    logger.info(f"Processing notes in {path}")
    processed = process_notes(path, args.force)
    
    print(f"Processed {processed} notes")


def handle_api_command(args):
    """Handle API command."""
    # Override settings if provided
    if args.host:
        settings.api.host = args.host
    
    if args.port:
        settings.api.port = args.port
    
    logger.info(f"Starting API server on {settings.api.host}:{settings.api.port}")
    start_api()


def handle_notes_list_command(args):
    """Handle notes list command."""
    if args.json:
        print(generate_notes_json())
        return
    
    with get_session() as session:
        if args.tag:
            # Filter by tag
            query = session.query(Note).join(
                TagNoteLink
            ).join(
                Tag
            ).filter(
                Tag.name == args.tag
            ).offset(args.offset).limit(args.limit)
        else:
            # Get all notes
            query = session.query(Note).offset(args.offset).limit(args.limit)
        
        notes = query.all()
        
        print(f"Found {len(notes)} notes:")
        for note in notes:
            tags = [tag.name for tag in note.tags]
            print(f"ID: {note.id} | {note.title} | Tags: {', '.join(tags) if tags else 'None'}")


def handle_notes_get_command(args):
    """Handle notes get command."""
    with get_session() as session:
        note = session.query(Note).filter(Note.id == args.id).first()
        
        if not note:
            print(f"Note with ID {args.id} not found")
            return
        
        if args.json:
            result = {
                "id": note.id,
                "title": note.title,
                "filename": note.filename,
                "path": note.path,
                "word_count": note.word_count,
                "has_front_matter": note.has_front_matter,
                "created_at": note.created_at.isoformat(),
                "updated_at": note.updated_at.isoformat(),
                "tags": [{"name": tag.name, "category": tag.category} for tag in note.tags]
            }
            
            if args.content:
                try:
                    with open(note.path, "r") as f:
                        result["content"] = f.read()
                except Exception as e:
                    result["error"] = f"Failed to read note content: {e}"
            
            print(json.dumps(result, indent=2))
            return
        
        print(f"ID: {note.id}")
        print(f"Title: {note.title}")
        print(f"File: {note.filename}")
        print(f"Path: {note.path}")
        print(f"Word count: {note.word_count}")
        print(f"Front matter: {'Yes' if note.has_front_matter else 'No'}")
        print(f"Created: {note.created_at}")
        print(f"Updated: {note.updated_at}")
        print(f"Tags: {', '.join(tag.name for tag in note.tags) if note.tags else 'None'}")
        
        if args.content:
            print("\nContent:")
            try:
                with open(note.path, "r") as f:
                    print(f.read())
            except Exception as e:
                print(f"Failed to read note content: {e}")


def handle_notes_search_command(args):
    """Handle notes search command."""
    # Create services once
    embedding_service = create_embedding_service()
    search_service = create_search_service(embedding_service=embedding_service)
    
    if args.json:
        if args.full_text:
            # Generate JSON for full-text search using the search service
            results = search_service.full_text_search(
                args.query, args.limit
            )
            
            # Convert to JSON
            result_data = []
            for note, score in results:
                result_data.append({
                    "id": note.id,
                    "title": note.title,
                    "filename": note.filename,
                    "path": note.path,
                    "tags": [{"name": tag.name, "category": tag.category} for tag in note.tags]
                })
            
            print(json.dumps(result_data))
        else:
            # Use the existing function for semantic search
            print(generate_search_json(args.query, args.limit))
        return
    
    # Use the search service created above
    if args.full_text:
        # Full-text search
        results = search_service.full_text_search(
            args.query, args.limit
        )
        
        print(f"Found {len(results)} notes matching '{args.query}':")
        for note, _ in results:
            tags = [tag.name for tag in note.tags]
            print(f"ID: {note.id} | {note.title} | Tags: {', '.join(tags) if tags else 'None'}")
            
            # Get context for the first few matches
            contexts = search_service.get_note_context(
                note, args.query, context_size=1, max_contexts=1
            )
            
            # Show a snippet from each match
            for context in contexts:
                print(f"    {context['snippet']}")
    else:
        # Semantic search
        results = search_service.semantic_search(
            args.query, args.limit
        )
        
        print(f"Found {len(results)} notes semantically similar to '{args.query}':")
        
        for note, score in results:
            tags = [tag.name for tag in note.tags]
            similarity_pct = int(score * 100)
            print(f"ID: {note.id} | {note.title} | Similarity: {similarity_pct}% | Tags: {', '.join(tags) if tags else 'None'}")


def handle_notes_suggest_tags_command(args):
    """Handle notes suggest-tags command."""
    # Create services
    embedding_service = create_embedding_service()
    tagging_service = create_tagging_service(embedding_service=embedding_service)
    
    if args.json:
        print(suggest_tags_json(args.id))
        return
    
    with get_session() as session:
        note = session.query(Note).filter(Note.id == args.id).first()
        
        if not note:
            print(f"Note with ID {args.id} not found")
            return
        
        # Get content from note sections
        sections = session.query(NoteSection).filter(NoteSection.note_id == args.id).order_by(NoteSection.order).all()
        content = "\n\n".join(section.content for section in sections)
        
        if not content:
            # Use actual file if no sections are available
            try:
                with open(note.path, "r") as f:
                    content = f.read()
            except Exception:
                print("Failed to read note content")
                return
        
        # Get existing tags
        existing_tags = [tag.name for tag in note.tags]
        
        # Generate suggestions
        suggestions = tagging_service.generate_tags_for_note(
            content, note.title, existing_tags
        )
        
        print(f"Tag suggestions for '{note.title}':")
        print(f"Existing tags: {', '.join(existing_tags) if existing_tags else 'None'}")
        print("\nSuggested tags:")
        
        for tag, confidence in suggestions:
            confidence_pct = int(confidence * 100)
            print(f"{tag} ({confidence_pct}%)")


def handle_notes_apply_tags_command(args):
    """Handle notes apply-tags command."""
    # Create tagging service
    tagging_service = create_tagging_service()
    
    with get_session() as session:
        note = session.query(Note).filter(Note.id == args.id).first()
        
        if not note:
            print(f"Note with ID {args.id} not found")
            return
        
        # Parse tags
        tags = [tag.strip() for tag in args.tags.split(",")]
        
        # Apply tags
        tagging_service.apply_tags_to_note(args.id, tags)
        
        print(f"Applied tags to '{note.title}': {', '.join(tags)}")


def handle_tags_list_command(args):
    """Handle tags list command."""
    if args.json:
        print(generate_tags_json())
        return
    
    with get_session() as session:
        query = session.query(Tag)
        
        if args.category:
            query = query.filter(Tag.category == args.category)
            
        tags = query.all()
        
        print(f"Found {len(tags)} tags:")
        
        # Group by category
        by_category = {}
        for tag in tags:
            category = tag.category or "Uncategorized"
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(tag)
        
        # Print by category
        for category, tags in sorted(by_category.items()):
            print(f"\n{category}:")
            for tag in sorted(tags, key=lambda t: t.name):
                note_count = len(tag.notes)
                print(f"  {tag.name} ({note_count} notes)")


def handle_tags_create_command(args):
    """Handle tags create command."""
    with get_session() as session:
        # Check if tag already exists
        existing = session.query(Tag).filter(Tag.name == args.name).first()
        
        if existing:
            print(f"Tag '{args.name}' already exists")
            return
        
        # Create tag
        tag = Tag(
            name=args.name,
            category=args.category,
            description=args.description
        )
        
        session.add(tag)
        session.commit()
        
        print(f"Created tag '{args.name}'")
        if args.category:
            print(f"Category: {args.category}")
        if args.description:
            print(f"Description: {args.description}")


def handle_stats_command(args):
    """Handle stats command."""
    if args.json:
        print(generate_note_stats())
        return
    
    with get_session() as session:
        # Count notes and tags
        note_count = session.query(Note).count()
        tag_count = session.query(Tag).count()
        
        # Get total word count
        total_words = session.query(func.sum(Note.word_count)).scalar() or 0
        
        print(f"Statistics:")
        print(f"Total notes: {note_count}")
        print(f"Total tags: {tag_count}")
        print(f"Total words: {total_words}")
        
        if note_count > 0:
            # Calculate average word count
            avg_words = total_words / note_count
            print(f"Average words per note: {avg_words:.1f}")
            
            # Get notes with front matter
            front_matter_count = session.query(Note).filter(Note.has_front_matter).count()
            front_matter_pct = (front_matter_count / note_count) * 100
            print(f"Notes with front matter: {front_matter_count} ({front_matter_pct:.1f}%)")
        
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
        
        if tag_counts:
            print("\nTop tags:")
            for name, count in tag_counts:
                print(f"  {name}: {count} notes")


def main(args=None):
    """Main entry point."""
    # Parse arguments
    parser = setup_cli()
    args = parser.parse_args(args)
    
    # Handle debug mode
    if args.debug:
        import logging
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("Debug mode enabled")
    
    # Load config
    config_path = args.config or os.path.expanduser("~/.config/note_organizer/config.yaml")
    
    # Create config if it doesn't exist
    if not os.path.exists(config_path) and args.command != "init":
        # Create default config
        config_dir = os.path.dirname(config_path)
        os.makedirs(config_dir, exist_ok=True)
        
        # Import here to avoid circular imports
        from note_organizer.__main__ import create_default_config
        create_default_config(config_path)
    
    # Handle commands
    if args.command == "nushell":
        handle_nushell_command(args)
    elif args.command == "init":
        handle_init_command(args, config_path)
    elif args.command == "config":
        handle_config_command(args, config_path)
    elif args.command == "process":
        handle_process_command(args)
    elif args.command == "api":
        handle_api_command(args)
    elif args.command == "notes":
        if args.notes_command == "list":
            handle_notes_list_command(args)
        elif args.notes_command == "get":
            handle_notes_get_command(args)
        elif args.notes_command == "search":
            handle_notes_search_command(args)
        elif args.notes_command == "suggest-tags":
            handle_notes_suggest_tags_command(args)
        elif args.notes_command == "apply-tags":
            handle_notes_apply_tags_command(args)
        else:
            parser.error("Please specify a notes command")
    elif args.command == "tags":
        if args.tags_command == "list":
            handle_tags_list_command(args)
        elif args.tags_command == "create":
            handle_tags_create_command(args)
        else:
            parser.error("Please specify a tags command")
    elif args.command == "stats":
        handle_stats_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 