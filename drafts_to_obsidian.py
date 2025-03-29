#!/usr/bin/env python3
"""
Drafts to Obsidian export tool with AI-powered organization.

This script provides a simple command-line interface for exporting notes from
Drafts to Obsidian with intelligent categorization and organization.
Supports both individual Markdown files and Drafts JSON exports.
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import re
import shutil
import time

# Configure basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("drafts_to_obsidian")

# Load the category schema
def load_schema(schema_path: str = "category_schema.yaml") -> Dict[str, Any]:
    """Load the category schema."""
    try:
        with open(schema_path, 'r') as f:
            schema = yaml.safe_load(f)
        return schema
    except Exception as e:
        logger.error(f"Error loading schema: {e}")
        sys.exit(1)

# Import necessary functions and classes
try:
    # Import from process_notes.py
    from process_notes import KeywordTagger, LLMTagger, EmbeddingTagger, NoteProcessor
    logger.info("Successfully imported from process_notes.py")
except ImportError as e:
    logger.error(f"Error importing from process_notes.py: {e}")
    logger.error("Please ensure process_notes.py is in the same directory")
    sys.exit(1)

# Path management utility functions
def get_category_path(category: str, obsidian_root: str) -> Path:
    """Convert a category to a directory path."""
    # Split the category into parts (e.g., "knowledge.technical.ai_ml" -> ["knowledge", "technical", "ai_ml"])
    parts = category.split('.')
    
    # Convert to path
    category_path = Path(obsidian_root)
    for part in parts:
        # Convert from snake_case to Title Case for better readability
        readable_name = part.replace('_', ' ').title()
        category_path = category_path / readable_name
    
    return category_path

def ensure_directory(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def sanitize_filename(title: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    # Remove characters not suitable for filenames
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().lower()
    # Replace whitespace/hyphens with a single hyphen
    safe_title = re.sub(r'[-\s]+', '-', safe_title)
    # Limit length to avoid issues with long filenames
    return safe_title[:100]

def generate_frontmatter(title: str, categories: List[str], tags: Optional[List[str]] = None) -> str:
    """Generate YAML frontmatter for a note."""
    frontmatter = {
        'title': title,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'categories': categories
    }
    
    if tags:
        frontmatter['tags'] = tags
    
    # Convert to YAML format
    frontmatter_str = '---\n'
    for key, value in frontmatter.items():
        if isinstance(value, list):
            frontmatter_str += f'{key}:\n'
            for item in value:
                frontmatter_str += f'  - "{item}"\n'
        else:
            frontmatter_str += f'{key}: "{value}"\n'
    frontmatter_str += '---\n\n'
    return frontmatter_str

def extract_title_from_content(content: str) -> str:
    """Extract a title from the content."""
    # Try to get title from first line
    lines = content.strip().split('\n')
    
    # Check for YAML front matter
    if lines and lines[0].strip() == '---':
        front_matter_end = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                front_matter_end = i
                break
        
        if front_matter_end > 0:
            # Extract front matter
            front_matter = '\n'.join(lines[1:front_matter_end])
            try:
                # Parse front matter as YAML
                metadata = yaml.safe_load(front_matter)
                if metadata and isinstance(metadata, dict) and 'title' in metadata:
                    return metadata['title']
            except Exception:
                pass
            
            # If no title in front matter or parsing failed, use first non-empty line after front matter
            for line in lines[front_matter_end+1:]:
                if line.strip():
                    # Remove markdown heading symbols
                    return re.sub(r'^#+\s*', '', line.strip())
    
    # No front matter or title not found in front matter
    # Use first non-empty line as title
    for line in lines:
        if line.strip():
            # Remove markdown heading symbols
            return re.sub(r'^#+\s*', '', line.strip())
    
    # Fallback if no non-empty lines
    return "Untitled Note"

def suggest_categories(content: str, title: str, schema: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Suggest categories for a note based on content and title."""
    # Initialize necessary taggers
    processor = NoteProcessor()
    
    # Process the note (this doesn't require a file, just the content)
    result = {
        "content_preview": content,
        "title": title,
        "categories": {}
    }
    
    # Get categories from different methods
    keyword_categories = processor.keyword_tagger.tag_note(content, title)
    result["categories"]["keyword"] = keyword_categories
    
    llm_categories = []
    if processor.llm_tagger:
        llm_categories = processor.llm_tagger.tag_note(content, title)
    result["categories"]["llm"] = llm_categories
    
    embedding_categories = []
    if processor.embedding_tagger:
        embedding_categories = processor.embedding_tagger.tag_note(content, title)
    result["categories"]["embedding"] = embedding_categories
    
    # Combine and weight results from different methods
    combined_scores = {}
    
    # Define weights for each method
    weights = {"keyword": 0.7, "llm": 1.5, "embedding": 1.0}
    
    for method, categories in result["categories"].items():
        weight = weights.get(method, 1.0)
        for category, score in categories:
            if category not in combined_scores:
                combined_scores[category] = 0
            combined_scores[category] += score * weight
    
    # Normalize scores
    max_score = max(combined_scores.values()) if combined_scores else 1.0
    normalized_scores = {category: score / max_score for category, score in combined_scores.items()}
    
    # Sort and return
    sorted_categories = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top 3 categories with score >= 0.5
    return [(category, score) for category, score in sorted_categories if score >= 0.5][:3]

def process_drafts_file(
    file_path: str, 
    obsidian_root: str,
    schema: Dict[str, Any],
    force: bool = False,
    interactive: bool = True
) -> Dict[str, Any]:
    """Process a Drafts file and export it to Obsidian."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title
        title = extract_title_from_content(content)
        
        # Extract tags from front matter or content
        tags = []
        try:
            if content.startswith('---'):
                # Find end of front matter
                end_marker = content.find('\n---', 3)
                if end_marker > 0:
                    front_matter = content[3:end_marker]
                    metadata = yaml.safe_load(front_matter)
                    if metadata and isinstance(metadata, dict) and 'tags' in metadata:
                        tags = metadata['tags']
        except Exception as e:
            logger.warning(f"Error extracting tags from front matter: {e}")
        
        # Suggest categories
        suggested_categories = suggest_categories(content, title, schema)
        
        # Get user input if interactive
        selected_categories = []
        if interactive:
            print(f"\nProcessing: {title}")
            print(f"File: {file_path}")
            print("\nSuggested categories:")
            
            for i, (category, score) in enumerate(suggested_categories):
                print(f"{i+1}. {category} ({score:.2f})")
            
            # Ask user to select categories
            selection = input("\nSelect categories (comma-separated numbers, Enter for all): ")
            if selection.strip():
                # Parse selected indices
                try:
                    indices = [int(i.strip()) - 1 for i in selection.split(',')]
                    selected_categories = [suggested_categories[i][0] for i in indices if 0 <= i < len(suggested_categories)]
                except ValueError:
                    print("Invalid selection, using all suggested categories.")
                    selected_categories = [category for category, _ in suggested_categories]
            else:
                selected_categories = [category for category, _ in suggested_categories]
            
            # Allow adding custom categories
            custom = input("Add custom category (or Enter to skip): ")
            if custom.strip():
                selected_categories.append(custom.strip())
        else:
            # Use top suggested categories
            selected_categories = [category for category, _ in suggested_categories]
        
        # Add a default category if none selected
        if not selected_categories:
            selected_categories.append("inbox")
        
        # Choose primary category (first in list) for file placement
        primary_category = selected_categories[0]
        
        # Generate safe filename
        safe_title = sanitize_filename(title)
        file_name = f"{safe_title}.md"
        
        # Get category path
        category_path = get_category_path(primary_category, obsidian_root)
        ensure_directory(category_path)
        
        # Prepare output path
        output_path = category_path / file_name
        
        # Check if file already exists
        if output_path.exists() and not force:
            print(f"File already exists: {output_path}")
            if interactive:
                overwrite = input("Overwrite? (y/n): ").lower() == 'y'
                if not overwrite:
                    return {
                        "success": False,
                        "message": "File exists and overwrite declined",
                        "path": str(output_path)
                    }
            else:
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                file_name = f"{safe_title}-{timestamp}.md"
                output_path = category_path / file_name
        
        # Process content
        final_content = content
        
        # Check and add frontmatter if needed
        if not content.startswith('---'):
            frontmatter = generate_frontmatter(title, selected_categories, tags)
            final_content = frontmatter + content
        elif tags or selected_categories:
            # Update existing frontmatter
            end_marker = content.find('\n---', 3)
            if end_marker > 0:
                front_matter = content[3:end_marker]
                try:
                    metadata = yaml.safe_load(front_matter)
                    # Add or update categories
                    metadata['categories'] = selected_categories
                    # Add tags if they don't exist
                    if 'tags' not in metadata and tags:
                        metadata['tags'] = tags
                    
                    # Regenerate frontmatter
                    new_frontmatter = '---\n' + yaml.dump(metadata, default_flow_style=False) + '---\n'
                    final_content = new_frontmatter + content[end_marker + 4:]
                except Exception as e:
                    logger.warning(f"Error updating frontmatter: {e}")
        
        # Write the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        logger.info(f"Exported note to {output_path}")
        
        return {
            "success": True,
            "title": title,
            "categories": selected_categories,
            "tags": tags,
            "source_path": file_path,
            "output_path": str(output_path),
            "message": f"Successfully exported to {output_path}"
        }
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return {
            "success": False,
            "source_path": file_path,
            "message": f"Error: {str(e)}"
        }

def process_directory(
    input_dir: str,
    obsidian_root: str,
    schema: Dict[str, Any],
    force: bool = False,
    interactive: bool = True,
    recursive: bool = False,
    move: bool = False
) -> Dict[str, Any]:
    """Process all markdown files in a directory."""
    input_path = Path(input_dir)
    
    # Find all markdown files
    if recursive:
        md_files = list(input_path.glob("**/*.md"))
    else:
        md_files = list(input_path.glob("*.md"))
    
    if not md_files:
        return {
            "success": False,
            "message": f"No markdown files found in {input_dir}"
        }
    
    # Process each file
    results = []
    for file_path in md_files:
        print(f"\nProcessing {file_path.name}...")
        result = process_drafts_file(
            str(file_path),
            obsidian_root,
            schema,
            force,
            interactive
        )
        
        # Move original file to archive if requested and export was successful
        if move and result.get("success"):
            # Create archive directory if needed
            archive_dir = input_path / "archived"
            archive_dir.mkdir(exist_ok=True)
            
            # Generate archive filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            archive_filename = f"{file_path.stem}-{timestamp}{file_path.suffix}"
            archive_path = archive_dir / archive_filename
            
            # Move file
            shutil.move(str(file_path), str(archive_path))
            logger.info(f"Moved original file to {archive_path}")
            result["archived_path"] = str(archive_path)
        
        results.append(result)
    
    # Summarize results
    success_count = sum(1 for r in results if r.get("success"))
    
    return {
        "success": True,
        "total": len(results),
        "success_count": success_count,
        "failed_count": len(results) - success_count,
        "results": results
    }

def watch_directory(
    input_dir: str,
    obsidian_root: str,
    schema: Dict[str, Any],
    force: bool = False,
    interactive: bool = True,
    move: bool = False,
    interval: int = 30
) -> None:
    """Watch a directory for new files and process them."""
    logger.info(f"Watching directory {input_dir} for new files...")
    
    # Track processed files
    processed_files = set()
    
    try:
        while True:
            # Get current files
            input_path = Path(input_dir)
            current_files = set(input_path.glob("*.md"))
            
            # Find new files
            new_files = current_files - processed_files
            
            if new_files:
                logger.info(f"Found {len(new_files)} new files!")
                
                # Process each new file
                for file_path in new_files:
                    logger.info(f"Processing {file_path.name}...")
                    
                    result = process_drafts_file(
                        str(file_path),
                        obsidian_root,
                        schema,
                        force,
                        interactive
                    )
                    
                    # Move original file to archive if requested and export was successful
                    if move and result.get("success"):
                        # Create archive directory if needed
                        archive_dir = input_path / "archived"
                        archive_dir.mkdir(exist_ok=True)
                        
                        # Generate archive filename with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        archive_filename = f"{file_path.stem}-{timestamp}{file_path.suffix}"
                        archive_path = archive_dir / archive_filename
                        
                        # Move file
                        shutil.move(str(file_path), str(archive_path))
                        logger.info(f"Moved original file to {archive_path}")
                
                # Update processed files
                processed_files.update(new_files)
            
            # Wait for next check
            time.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info("Watching stopped by user")

def process_drafts_json(
    json_file: str,
    obsidian_root: str,
    schema: Dict[str, Any],
    force: bool = False,
    interactive: bool = True,
    move: bool = False
) -> Dict[str, Any]:
    """Process a Drafts app JSON export file."""
    try:
        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            drafts_data = json.load(f)
        
        if not isinstance(drafts_data, list):
            return {
                "success": False,
                "message": "Invalid Drafts JSON format: expected an array of drafts"
            }
        
        logger.info(f"Found {len(drafts_data)} drafts in the JSON export")
        
        # Process each draft
        results = []
        for i, draft in enumerate(drafts_data):
            # Extract draft data
            try:
                draft_uuid = draft.get("uuid", f"unknown-{i}")
                content = draft.get("content", "")
                created_at = draft.get("createdAt")
                modified_at = draft.get("modifiedAt")
                
                # Extract tags
                tags = draft.get("tags", [])
                
                # Check if this is a valid draft
                if not content:
                    logger.warning(f"Draft {draft_uuid} has no content, skipping")
                    continue
                
                # Extract title (from content or use a fallback)
                title = extract_title_from_content(content)
                
                print(f"\nProcessing draft: {title} ({draft_uuid})")
                if interactive:
                    print(f"Created: {created_at}")
                    print(f"Modified: {modified_at}")
                    print(f"Tags: {', '.join(tags) if tags else 'None'}")
                    print(f"\nPreview:\n{content[:200]}...")
                
                # Suggest categories
                suggested_categories = suggest_categories(content, title, schema)
                
                # Get user input if interactive
                selected_categories = []
                if interactive:
                    print("\nSuggested categories:")
                    
                    for i, (category, score) in enumerate(suggested_categories):
                        print(f"{i+1}. {category} ({score:.2f})")
                    
                    # Ask user to select categories
                    selection = input("\nSelect categories (comma-separated numbers, Enter for all): ")
                    if selection.strip():
                        # Parse selected indices
                        try:
                            indices = [int(i.strip()) - 1 for i in selection.split(',')]
                            selected_categories = [suggested_categories[i][0] for i in indices if 0 <= i < len(suggested_categories)]
                        except ValueError:
                            print("Invalid selection, using all suggested categories.")
                            selected_categories = [category for category, _ in suggested_categories]
                    else:
                        selected_categories = [category for category, _ in suggested_categories]
                    
                    # Allow adding custom categories
                    custom = input("Add custom category (or Enter to skip): ")
                    if custom.strip():
                        selected_categories.append(custom.strip())
                else:
                    # Use top suggested categories
                    selected_categories = [category for category, _ in suggested_categories]
                
                # Add a default category if none selected
                if not selected_categories:
                    selected_categories.append("inbox")
                
                # Choose primary category (first in list) for file placement
                primary_category = selected_categories[0]
                
                # Generate safe filename
                safe_title = sanitize_filename(title)
                file_name = f"{safe_title}.md"
                
                # Get category path
                category_path = get_category_path(primary_category, obsidian_root)
                ensure_directory(category_path)
                
                # Prepare output path
                output_path = category_path / file_name
                
                # Check if file already exists
                if output_path.exists() and not force:
                    print(f"File already exists: {output_path}")
                    if interactive:
                        overwrite = input("Overwrite? (y/n): ").lower() == 'y'
                        if not overwrite:
                            results.append({
                                "success": False,
                                "uuid": draft_uuid,
                                "title": title,
                                "message": "File exists and overwrite declined",
                                "path": str(output_path)
                            })
                            continue
                    else:
                        # Generate unique filename
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        file_name = f"{safe_title}-{timestamp}.md"
                        output_path = category_path / file_name
                
                # Process content
                final_content = content
                
                # Check and add frontmatter if needed
                if not content.startswith('---'):
                    # Convert Drafts created/modified dates if available
                    try:
                        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00')) if created_at else datetime.now(timezone.utc)
                        created_str = created_date.strftime('%Y-%m-%d %H:%M:%S')
                    except (ValueError, TypeError):
                        created_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                    # Generate frontmatter
                    frontmatter = generate_frontmatter(title, selected_categories, tags)
                    final_content = frontmatter + content
                elif tags or selected_categories:
                    # Update existing frontmatter
                    end_marker = content.find('\n---', 3)
                    if end_marker > 0:
                        front_matter = content[3:end_marker]
                        try:
                            metadata = yaml.safe_load(front_matter)
                            # Add or update categories
                            metadata['categories'] = selected_categories
                            # Add tags if they don't exist
                            if 'tags' not in metadata and tags:
                                metadata['tags'] = tags
                            
                            # Regenerate frontmatter
                            new_frontmatter = '---\n' + yaml.dump(metadata, default_flow_style=False) + '---\n'
                            final_content = new_frontmatter + content[end_marker + 4:]
                        except Exception as e:
                            logger.warning(f"Error updating frontmatter: {e}")
                
                # Write the file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_content)
                
                logger.info(f"Exported draft {draft_uuid} to {output_path}")
                
                # Add to results
                results.append({
                    "success": True,
                    "uuid": draft_uuid,
                    "title": title,
                    "categories": selected_categories,
                    "tags": tags,
                    "output_path": str(output_path),
                    "message": f"Successfully exported to {output_path}"
                })
                
            except Exception as e:
                logger.error(f"Error processing draft {i}: {e}")
                results.append({
                    "success": False,
                    "uuid": draft.get("uuid", f"unknown-{i}"),
                    "message": f"Error: {str(e)}"
                })
        
        # Summarize results
        success_count = sum(1 for r in results if r.get("success"))
        
        return {
            "success": True,
            "total": len(results),
            "success_count": success_count,
            "failed_count": len(results) - success_count,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing Drafts JSON file {json_file}: {e}")
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Export Drafts notes to Obsidian with AI organization")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output", type=str, required=True, help="Obsidian vault root directory")
    parser.add_argument("--schema", type=str, default="category_schema.yaml", help="Category schema file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--non-interactive", action="store_true", help="Run without user interaction")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("--move", action="store_true", help="Move originals to archive after processing")
    parser.add_argument("--watch", action="store_true", help="Watch directory for new files")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval in seconds")
    parser.add_argument("--json", action="store_true", help="Input is a Drafts JSON export file")
    args = parser.parse_args()
    
    # Load schema
    schema = load_schema(args.schema)
    
    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Check if output exists
    output_path = Path(args.output)
    if not output_path.exists():
        logger.error(f"Output path does not exist: {args.output}")
        sys.exit(1)
    
    # Process depending on input type
    if input_path.is_file():
        if args.json or input_path.suffix.lower() == '.json':
            # Process Drafts JSON export
            result = process_drafts_json(
                str(input_path),
                args.output,
                schema,
                args.force,
                not args.non_interactive,
                args.move
            )
            
            if result["success"]:
                print(f"\nProcessed {result['total']} drafts.")
                print(f"Success: {result['success_count']}")
                print(f"Failed: {result['failed_count']}")
            else:
                print(f"Failed: {result['message']}")
        else:
            # Process single markdown file
            result = process_drafts_file(
                str(input_path),
                args.output,
                schema,
                args.force,
                not args.non_interactive
            )
            
            if result["success"]:
                print(f"Successfully exported to {result['output_path']}")
            else:
                print(f"Failed: {result['message']}")
    
    elif input_path.is_dir():
        if args.watch:
            # Watch directory
            watch_directory(
                str(input_path),
                args.output,
                schema,
                args.force,
                not args.non_interactive,
                args.move,
                args.interval
            )
        else:
            # Process directory
            result = process_directory(
                str(input_path),
                args.output,
                schema,
                args.force,
                not args.non_interactive,
                args.recursive,
                args.move
            )
            
            if result["success"]:
                print(f"\nProcessed {result['total']} files.")
                print(f"Success: {result['success_count']}")
                print(f"Failed: {result['failed_count']}")
            else:
                print(f"Failed: {result['message']}")

if __name__ == "__main__":
    main()