"""
Main entry point for the Note Organizer package.

This module is executed when the package is run directly with `python -m note_organizer`
or when installed as a command-line tool.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from loguru import logger

from note_organizer.core.config import settings, load_config, update_settings, Settings
from note_organizer.api.server import start as start_api
from note_organizer.db.database import create_db_and_tables
from note_organizer.services.processor import process_notes
from note_organizer.services.service_factory import reset_services
from note_organizer.services.tagging import LLMProvider
from note_organizer.cli.main import main


def create_default_config(config_path):
    """Create default configuration file."""
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Default configuration
    config = Settings(
        notes_dir=os.path.expanduser("~/Documents/Notes"),
        llm_provider="none",
        # Other defaults are set by the Settings class
    )
    
    # Convert to dict
    config_dict = config.dict()
    
    # Write to file
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"Created default configuration at {config_path}")
    return config


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


if __name__ == "__main__":
    sys.exit(main()) 