#!/usr/bin/env python3
"""
Batch Operations for Markdown Notes - Apply transformations to multiple notes at once
"""

import os
import re
import yaml
import glob
import argparse
from tqdm import tqdm
from pathlib import Path

class BatchOperator:
    """Perform batch operations on a collection of markdown notes."""
    
    def __init__(self, notes_dir, backup=True):
        """Initialize the batch operator."""
        self.notes_dir = notes_dir
        self.backup = backup
        
    def find_notes(self, pattern="**/*.md", exclude_pattern=None):
        """Find all notes matching the pattern."""
        note_files = glob.glob(os.path.join(self.notes_dir, pattern), recursive=True)
        
        # Apply exclusion if provided
        if exclude_pattern:
            exclude_files = set(glob.glob(os.path.join(self.notes_dir, exclude_pattern), recursive=True))
            note_files = [f for f in note_files if f not in exclude_files]
            
        return note_files
    
    def backup_file(self, file_path):
        """Create a backup of a file before modifying it."""
        if not self.backup:
            return
            
        backup_path = file_path + ".bak"
        try:
            with open(file_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
        except Exception as e:
            print(f"Warning: Could not create backup for {file_path}: {e}")
    
    def add_tags(self, tags, pattern="**/*.md", exclude_pattern=None):
        """Add specified tags to all matching notes."""
        note_files = self.find_notes(pattern, exclude_pattern)
        
        if not note_files:
            print(f"No files found matching pattern '{pattern}'")
            return
        
        print(f"Adding tags {tags} to {len(note_files)} files")
        
        for file_path in tqdm(note_files, desc="Adding tags"):
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse front matter
                front_matter, clean_content = self._extract_front_matter(content)
                
                # Add tags
                existing_tags = front_matter.get('tags', [])
                if isinstance(existing_tags, str):
                    existing_tags = [existing_tags]
                
                # Merge tags
                all_tags = set(existing_tags)
                all_tags.update(tags)
                front_matter['tags'] = sorted(list(all_tags))
                
                # Create backup
                self.backup_file(file_path)
                
                # Write updated file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("---\n")
                    f.write(yaml.dump(front_matter, default_flow_style=False))
                    f.write("---\n\n")
                    f.write(clean_content)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    def remove_tags(self, tags, pattern="**/*.md", exclude_pattern=None):
        """Remove specified tags from all matching notes."""
        note_files = self.find_notes(pattern, exclude_pattern)
        
        if not note_files:
            print(f"No files found matching pattern '{pattern}'")
            return
        
        print(f"Removing tags {tags} from {len(note_files)} files")
        
        for file_path in tqdm(note_files, desc="Removing tags"):
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse front matter
                front_matter, clean_content = self._extract_front_matter(content)
                
                # Get existing tags
                existing_tags = front_matter.get('tags', [])
                if isinstance(existing_tags, str):
                    existing_tags = [existing_tags]
                
                # Remove specified tags
                updated_tags = [tag for tag in existing_tags if tag not in tags]
                if updated_tags or 'tags' in front_matter:  # Only update if tags existed before
                    front_matter['tags'] = updated_tags
                
                # Create backup
                self.backup_file(file_path)
                
                # Write updated file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("---\n")
                    f.write(yaml.dump(front_matter, default_flow_style=False))
                    f.write("---\n\n")
                    f.write(clean_content)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    def replace_text(self, old_text, new_text, pattern="**/*.md", exclude_pattern=None, case_sensitive=True):
        """Replace text in all matching notes."""
        note_files = self.find_notes(pattern, exclude_pattern)
        
        if not note_files:
            print(f"No files found matching pattern '{pattern}'")
            return
        
        print(f"Replacing '{old_text}' with '{new_text}' in {len(note_files)} files")
        
        count = 0
        for file_path in tqdm(note_files, desc="Replacing text"):
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Perform replacement
                if case_sensitive:
                    updated_content = content.replace(old_text, new_text)
                else:
                    updated_content = re.sub(re.escape(old_text), new_text, content, flags=re.IGNORECASE)
                
                # Skip if no changes
                if content == updated_content:
                    continue
                
                # Create backup
                self.backup_file(file_path)
                
                # Write updated file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                count += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"Modified {count} files")
    
    def add_front_matter(self, properties, pattern="**/*.md", exclude_pattern=None):
        """Add or update front matter properties in all matching notes."""
        note_files = self.find_notes(pattern, exclude_pattern)
        
        if not note_files:
            print(f"No files found matching pattern '{pattern}'")
            return
        
        print(f"Adding front matter properties to {len(note_files)} files")
        
        for file_path in tqdm(note_files, desc="Adding front matter"):
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse front matter
                front_matter, clean_content = self._extract_front_matter(content)
                
                # Add properties
                for key, value in properties.items():
                    front_matter[key] = value
                
                # Create backup
                self.backup_file(file_path)
                
                # Write updated file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("---\n")
                    f.write(yaml.dump(front_matter, default_flow_style=False))
                    f.write("---\n\n")
                    f.write(clean_content)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    def consolidate_links(self, dry_run=True, pattern="**/*.md", exclude_pattern=None):
        """Consolidate internal links to ensure they are consistent."""
        note_files = self.find_notes(pattern, exclude_pattern)
        
        if not note_files:
            print(f"No files found matching pattern '{pattern}'")
            return
            
        print(f"Analyzing internal links in {len(note_files)} files")
        
        # Map filenames to paths for link resolution
        filename_map = {}
        for file_path in note_files:
            filename = os.path.basename(file_path)
            filename_without_ext = os.path.splitext(filename)[0]
            filename_map[filename_without_ext.lower()] = file_path
        
        # Regular expression for Markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        wiki_link_pattern = r'\[\[([^\]]+)\]\]'
        
        changes = []
        
        for file_path in tqdm(note_files, desc="Analyzing links"):
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all markdown links
                md_links = re.finditer(link_pattern, content)
                for match in md_links:
                    link_text = match.group(1)
                    link_target = match.group(2)
                    
                    # Check if it's an internal link to a note
                    if not link_target.startswith(('http:', 'https:', 'ftp:', '/')):
                        link_without_ext = os.path.splitext(link_target)[0].lower()
                        if link_without_ext in filename_map:
                            correct_target = os.path.basename(filename_map[link_without_ext])
                            correct_target_without_ext = os.path.splitext(correct_target)[0]
                            
                            if link_target != correct_target_without_ext:
                                changes.append({
                                    'file': file_path,
                                    'old': f"[{link_text}]({link_target})",
                                    'new': f"[{link_text}]({correct_target_without_ext})"
                                })
                
                # Find all wiki-style links
                wiki_links = re.finditer(wiki_link_pattern, content)
                for match in wiki_links:
                    link_text = match.group(1)
                    
                    # Handle pipe syntax for display text: [[target|display]]
                    if '|' in link_text:
                        link_target, display_text = link_text.split('|', 1)
                    else:
                        link_target = link_text
                        display_text = link_text
                    
                    link_target_lower = link_target.lower()
                    if link_target_lower in filename_map:
                        correct_target = os.path.basename(filename_map[link_target_lower])
                        correct_target_without_ext = os.path.splitext(correct_target)[0]
                        
                        if link_target != correct_target_without_ext:
                            changes.append({
                                'file': file_path,
                                'old': f"[[{link_text}]]",
                                'new': f"[[{correct_target_without_ext}|{display_text}]]" 
                                if '|' in link_text else f"[[{correct_target_without_ext}]]"
                            })
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Print report
        print(f"Found {len(changes)} links to update")
        
        if changes:
            for change in changes[:10]:  # Show first 10 changes
                print(f"{change['file']}: {change['old']} -> {change['new']}")
            
            if len(changes) > 10:
                print(f"... and {len(changes) - 10} more")
        
        # Apply changes if not dry run
        if not dry_run and changes:
            print("Applying changes...")
            
            files_to_update = {}
            for change in changes:
                if change['file'] not in files_to_update:
                    # Read file content
                    with open(change['file'], 'r', encoding='utf-8') as f:
                        files_to_update[change['file']] = f.read()
            
            # Apply all changes to each file
            updated_count = 0
            for file_path, content in tqdm(files_to_update.items(), desc="Updating files"):
                original_content = content
                
                # Apply changes for this file
                for change in [c for c in changes if c['file'] == file_path]:
                    content = content.replace(change['old'], change['new'])
                
                # Only write if changes were made
                if content != original_content:
                    # Create backup
                    self.backup_file(file_path)
                    
                    # Write updated content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    updated_count += 1
            
            print(f"Updated {updated_count} files")
    
    def format_front_matter(self, pattern="**/*.md", exclude_pattern=None, sort_keys=True):
        """Format and standardize front matter across all notes."""
        note_files = self.find_notes(pattern, exclude_pattern)
        
        if not note_files:
            print(f"No files found matching pattern '{pattern}'")
            return
            
        print(f"Formatting front matter in {len(note_files)} files")
        
        for file_path in tqdm(note_files, desc="Formatting front matter"):
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse front matter
                front_matter, clean_content = self._extract_front_matter(content)
                
                # Skip if no front matter
                if not front_matter:
                    continue
                
                # Sort tags if present
                if 'tags' in front_matter:
                    tags = front_matter['tags']
                    if isinstance(tags, str):
                        tags = [tags]
                    front_matter['tags'] = sorted(tags) if sort_keys else tags
                
                # Create backup
                self.backup_file(file_path)
                
                # Write updated file with sorted keys if requested
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("---\n")
                    f.write(yaml.dump(front_matter, default_flow_style=False, sort_keys=sort_keys))
                    f.write("---\n\n")
                    f.write(clean_content)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    def _extract_front_matter(self, content):
        """Extract YAML front matter from markdown content."""
        front_matter = {}
        clean_content = content
        
        # Check for YAML front matter
        front_matter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if front_matter_match:
            front_matter_text = front_matter_match.group(1)
            try:
                front_matter = yaml.safe_load(front_matter_text) or {}
                clean_content = content[front_matter_match.end():]
            except yaml.YAMLError:
                # If YAML parsing fails, assume it's not valid front matter
                pass
                
        return front_matter, clean_content

def main():
    parser = argparse.ArgumentParser(description="Batch Operations for Markdown Notes")
    parser.add_argument("notes_dir", help="Directory containing markdown notes")
    parser.add_argument("--no-backup", action="store_true", help="Disable file backups before modifications")
    
    subparsers = parser.add_subparsers(dest="command", help="Operation to perform")
    
    # Add tags command
    add_tags_parser = subparsers.add_parser("add-tags", help="Add tags to notes")
    add_tags_parser.add_argument("tags", nargs="+", help="Tags to add")
    add_tags_parser.add_argument("--pattern", default="**/*.md", help="File pattern to match")
    add_tags_parser.add_argument("--exclude", help="Pattern to exclude")
    
    # Remove tags command
    remove_tags_parser = subparsers.add_parser("remove-tags", help="Remove tags from notes")
    remove_tags_parser.add_argument("tags", nargs="+", help="Tags to remove")
    remove_tags_parser.add_argument("--pattern", default="**/*.md", help="File pattern to match")
    remove_tags_parser.add_argument("--exclude", help="Pattern to exclude")
    
    # Replace text command
    replace_parser = subparsers.add_parser("replace", help="Replace text in notes")
    replace_parser.add_argument("old_text", help="Text to replace")
    replace_parser.add_argument("new_text", help="Replacement text")
    replace_parser.add_argument("--pattern", default="**/*.md", help="File pattern to match")
    replace_parser.add_argument("--exclude", help="Pattern to exclude")
    replace_parser.add_argument("--ignore-case", action="store_true", help="Case insensitive matching")
    
    # Add front matter command
    front_matter_parser = subparsers.add_parser("add-front-matter", help="Add front matter properties")
    front_matter_parser.add_argument("properties", nargs="+", help="Properties to add (key=value format)")
    front_matter_parser.add_argument("--pattern", default="**/*.md", help="File pattern to match")
    front_matter_parser.add_argument("--exclude", help="Pattern to exclude")
    
    # Consolidate links command
    links_parser = subparsers.add_parser("consolidate-links", help="Consolidate internal links")
    links_parser.add_argument("--apply", action="store_true", help="Apply changes (dry run by default)")
    links_parser.add_argument("--pattern", default="**/*.md", help="File pattern to match")
    links_parser.add_argument("--exclude", help="Pattern to exclude")
    
    # Format front matter command
    format_parser = subparsers.add_parser("format", help="Format and standardize front matter")
    format_parser.add_argument("--pattern", default="**/*.md", help="File pattern to match")
    format_parser.add_argument("--exclude", help="Pattern to exclude")
    format_parser.add_argument("--no-sort", action="store_true", help="Don't sort front matter keys")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    operator = BatchOperator(args.notes_dir, not args.no_backup)
    
    if args.command == "add-tags":
        operator.add_tags(args.tags, args.pattern, args.exclude)
    elif args.command == "remove-tags":
        operator.remove_tags(args.tags, args.pattern, args.exclude)
    elif args.command == "replace":
        operator.replace_text(args.old_text, args.new_text, args.pattern, args.exclude, not args.ignore_case)
    elif args.command == "add-front-matter":
        # Parse properties from key=value format
        properties = {}
        for prop in args.properties:
            if "=" in prop:
                key, value = prop.split("=", 1)
                properties[key] = value
            else:
                print(f"Warning: Skipping malformed property '{prop}'. Use format key=value")
        
        operator.add_front_matter(properties, args.pattern, args.exclude)
    elif args.command == "consolidate-links":
        operator.consolidate_links(not args.apply, args.pattern, args.exclude)
    elif args.command == "format":
        operator.format_front_matter(args.pattern, args.exclude, not args.no_sort)

if __name__ == "__main__":
    main() 