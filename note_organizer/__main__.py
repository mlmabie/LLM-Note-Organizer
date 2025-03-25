"""
Main entry point for Note Organizer application.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from loguru import logger

from note_organizer.core.config import settings, load_config, update_settings
from note_organizer.api.server import start as start_api
from note_organizer.db.database import create_db_and_tables
from note_organizer.services.processor import process_notes
from note_organizer.services.service_factory import reset_services
from note_organizer.services.tagging import LLMProvider


def create_default_config(config_path):
    """Create a default configuration file if none exists."""
    default_config = {
        "debug": True,
        "notes_dir": str(Path.home() / "Documents/Notes"),
        "llm_provider": "none",  # none, openai, claude, or google
        "database": {
            "url": "sqlite:///notes.db",
            "echo": False
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
            "cors_origins": ["http://localhost:3000"],
            "enable_docs": True
        },
        "embedding": {
            "model_name": "all-MiniLM-L6-v2",
            "use_cce": True,
            "cce_centroids": 1024,
            "cce_dim": 64,
            "cache_dir": ".cache"
        },
        "openai": {
            "api_key": "",
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 150
        },
        "claude": {
            "api_key": "",
            "model": "claude-instant-1",
            "temperature": 0.3,
            "max_tokens": 150
        },
        "google": {
            "api_key": "",
            "model": "gemini-pro",
            "temperature": 0.3,
            "max_tokens": 150
        },
        "tagging": {
            "use_dspy": True,
            "confidence_threshold": 0.7,
            "max_tags_per_note": 10,
            "default_tags": ["inbox", "unprocessed"]
        },
        "log": {
            "level": "INFO",
            "path": "logs"
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Write config file
    with open(config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    logger.info(f"Created default configuration at {config_path}")


def setup_logging():
    """Set up logging configuration."""
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        level=settings.log.level,
        format=settings.log.format
    )
    
    # Add file logger
    log_path = Path(settings.log.path)
    log_path.mkdir(exist_ok=True)
    logger.add(
        log_path / "note_organizer.log",
        rotation=settings.log.rotation,
        retention=settings.log.retention,
        level=settings.log.level,
        format=settings.log.format.replace("<green>", "").replace("</green>", "")
                                  .replace("<level>", "").replace("</level>", "")
                                  .replace("<cyan>", "").replace("</cyan>", "")
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Note Organizer CLI")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Start API server")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process notes")
    process_parser.add_argument("--path", type=str, help="Path to notes directory")
    process_parser.add_argument("--force", action="store_true", help="Force reprocessing of all notes")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize note organizer")
    init_parser.add_argument("--llm", choices=["none", "openai", "claude", "google"], 
                            default="none", help="Choose LLM provider")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Update configuration")
    config_parser.add_argument("--llm", choices=["none", "openai", "claude", "google"], 
                             help="Set LLM provider")
    config_parser.add_argument("--openai-key", help="Set OpenAI API key")
    config_parser.add_argument("--claude-key", help="Set Claude API key")
    config_parser.add_argument("--google-key", help="Set Google AI API key")
    config_parser.add_argument("--notes-dir", help="Set notes directory")
    
    args = parser.parse_args()
    
    # Load config
    config_path = args.config or os.path.expanduser("~/.config/note_organizer/config.yaml")
    
    # Create config if it doesn't exist
    if not os.path.exists(config_path):
        create_default_config(config_path)
    
    # Load configuration
    config = load_config(config_path)
    update_settings(config)
    
    # Handle config command early
    if args.command == "config":
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
                yaml.dump(config_dict, f, default_flow_style=False)
                
            print(f"Configuration updated in {config_path}")
            return
        else:
            print(f"No configuration changes specified")
            return
    
    # Setup logging
    setup_logging()
    
    logger.info(f"Note Organizer starting up in {config.environment} mode")
    
    # Create database tables
    create_db_and_tables()
    
    # Handle commands
    if args.command == "api":
        logger.info("Starting API server")
        start_api()
    elif args.command == "process":
        path = args.path or settings.notes_dir
        logger.info(f"Processing notes in {path}")
        process_notes(path, args.force)
    elif args.command == "init":
        logger.info("Initializing Note Organizer")
        
        # Update LLM provider if specified
        if args.llm and args.llm != "none":
            # Update config
            config.llm_provider = args.llm
            
            # Save updated config
            config_dict = config.dict()
            # Handle SecretStr values
            for provider in ["openai", "claude", "google"]:
                if config_dict[provider]["api_key"] is not None:
                    config_dict[provider]["api_key"] = config_dict[provider]["api_key"].get_secret_value()
            
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            # Update settings
            update_settings(config)
            # Reset service singletons to use new settings
            reset_services()
            
            logger.info(f"Set LLM provider to {args.llm}")
        
        # Create database tables
        create_db_and_tables()
        
        # Create notes directory if it doesn't exist
        os.makedirs(settings.notes_dir, exist_ok=True)
        logger.info(f"Created notes directory at {settings.notes_dir}")
        
        logger.info("Initialization complete")
    else:
        # Default to showing help
        parser.print_help()


if __name__ == "__main__":
    main() 