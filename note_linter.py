#!/usr/bin/env python3
"""
Markdown Note Linter - Check for common issues in markdown notes and fix them
"""

import os
import re
import yaml
import glob
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import Counter

class NoteLinter:
    """Check markdown notes for common issues and optionally fix them."""
    
    def __init__(self, notes_dir, output_dir=None, backup=True):
        """Initialize the note linter."""
        self.notes_dir = notes_dir
        self.output_dir = output_dir
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

    def lint_notes(self, pattern="**/*.md", exclude_pattern=None, fix=False):
        """Lint markdown notes for common issues."""
        note_files = self.find_notes(pattern, exclude_pattern)
        
        if not note_files:
            print(f"No files found matching pattern '{pattern}'")
            return
        
        print(f"Linting {len(note_files)} markdown files...")
        
        # Track issues found
        issues = Counter()
        
        for file_path in tqdm(note_files, desc="Linting notes"):
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract front matter and content
                front_matter, clean_content = self._extract_front_matter(content)
                
                # Check for issues
                file_issues = {}
                
                # Check front matter
                if not front_matter:
                    file_issues['missing_front_matter'] = "No YAML front matter found"
                else:
                    # Check for missing title
                    if 'title' not in front_matter:
                        file_issues['missing_title'] = "Front matter is missing a title"
                    
                    # Check for missing tags
                    if 'tags' not in front_matter:
                        file_issues['missing_tags'] = "Front matter is missing tags"
                    elif not front_matter['tags']:
                        file_issues['empty_tags'] = "Tags array is empty"
                    
                    # Check for malformed tags
                    if 'tags' in front_matter and isinstance(front_matter['tags'], str):
                        file_issues['malformed_tags'] = "Tags should be an array, not a string"
                
                # Check content
                # Check for broken links
                broken_links = self._find_broken_links(clean_content, note_files)
                if broken_links:
                    file_issues['broken_links'] = f"Found {len(broken_links)} broken links: {', '.join(broken_links[:3])}" + ("..." if len(broken_links) > 3 else "")
                
                # Check for duplicate headings
                duplicate_headings = self._find_duplicate_headings(clean_content)
                if duplicate_headings:
                    file_issues['duplicate_headings'] = f"Found duplicate headings: {', '.join(duplicate_headings[:3])}" + ("..." if len(duplicate_headings) > 3 else "")
                
                # Check for trailing whitespace
                if re.search(r'[ \t]+$', content, re.MULTILINE):
                    file_issues['trailing_whitespace'] = "Found lines with trailing whitespace"
                
                # Check for long paragraphs (potential readability issues)
                long_paragraphs = self._find_long_paragraphs(clean_content)
                if long_paragraphs:
                    file_issues['long_paragraphs'] = f"Found {long_paragraphs} excessively long paragraphs (>300 words)"
                
                # Check for missing newline at end of file
                if not content.endswith('\n'):
                    file_issues['missing_final_newline'] = "File doesn't end with a newline"
                
                # Update issue counter
                for issue in file_issues:
                    issues[issue] += 1
                
                # Report issues for this file
                if file_issues:
                    rel_path = os.path.relpath(file_path, self.notes_dir)
                    print(f"\n{rel_path}:")
                    for issue, description in file_issues.items():
                        print(f"  - {description}")
                
                # Fix issues if requested
                if fix and file_issues:
                    fixed_content = self._fix_issues(content, front_matter, clean_content, file_issues, file_path, note_files)
                    
                    if fixed_content != content:
                        output_path = file_path
                        if self.output_dir:
                            rel_path = os.path.relpath(file_path, self.notes_dir)
                            output_path = os.path.join(self.output_dir, rel_path)
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # Create backup if modifying in place
                        if output_path == file_path:
                            self.backup_file(file_path)
                        
                        # Write fixed content
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(fixed_content)
                        
                        print(f"  ✓ Fixed issues in {os.path.relpath(output_path, self.notes_dir)}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Print summary
        print("\nLinting Summary:")
        print(f"Checked {len(note_files)} files")
        
        if not issues:
            print("No issues found! 🎉")
        else:
            print(f"Found {sum(issues.values())} issues across {len(issues)} categories:")
            for issue, count in issues.most_common():
                print(f"  - {issue}: {count} occurrences")
            
            if fix:
                print("\nFixed issues where possible. Some issues may require manual review.")
            else:
                print("\nRun with --fix to attempt automatic fixes.")
    
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
    
    def _find_broken_links(self, content, note_files):
        """Find broken internal links in the content."""
        broken_links = []
        
        # Map of filenames without extension (lowercase) to original filename
        filenames = {}
        for file_path in note_files:
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0].lower()
            filenames[base_name] = filename
        
        # Find all markdown links
        md_links = re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', content)
        for match in md_links:
            link_target = match.group(2)
            
            # Skip external links and anchors
            if link_target.startswith(('http:', 'https:', 'ftp:', '#', '/')):
                continue
            
            # Check if target exists (without extension)
            link_without_ext = os.path.splitext(link_target)[0].lower()
            if link_without_ext not in filenames:
                broken_links.append(link_target)
        
        # Find all wiki-style links
        wiki_links = re.finditer(r'\[\[([^\]]+)\]\]', content)
        for match in wiki_links:
            link_text = match.group(1)
            
            # Handle pipe syntax for display text: [[target|display]]
            if '|' in link_text:
                link_target, _ = link_text.split('|', 1)
            else:
                link_target = link_text
            
            # Check if target exists
            link_target_lower = link_target.lower()
            if link_target_lower not in filenames:
                broken_links.append(link_target)
        
        return broken_links
    
    def _find_duplicate_headings(self, content):
        """Find duplicate headings in the content."""
        headings = {}
        duplicates = []
        
        # Find all headings
        for match in re.finditer(r'^(#{1,6})\s+(.+?)(?:\s+#{1,6})?$', content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            
            # Store heading with its level
            heading_key = f"{level}:{text}"
            if heading_key in headings:
                duplicates.append(text)
            else:
                headings[heading_key] = True
        
        return duplicates
    
    def _find_long_paragraphs(self, content, max_words=300):
        """Find excessively long paragraphs."""
        count = 0
        
        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        
        for paragraph in paragraphs:
            # Skip headings, lists, and code blocks
            if re.match(r'^(#|\s*[-*+]|\s*\d+\.|\s*```)', paragraph.strip()):
                continue
                
            # Count words
            words = len(re.findall(r'\b\w+\b', paragraph))
            if words > max_words:
                count += 1
        
        return count
    
    def _fix_issues(self, content, front_matter, clean_content, issues, file_path, note_files):
        """Fix identified issues in the content."""
        modified_content = content
        modified_front_matter = front_matter.copy() if front_matter else {}
        
        # Fix missing front matter
        if 'missing_front_matter' in issues:
            # Create minimal front matter with filename as title
            filename = os.path.basename(file_path)
            title = os.path.splitext(filename)[0]
            
            # Replace underscores and hyphens with spaces
            title = re.sub(r'[_-]', ' ', title)
            # Capitalize words
            title = ' '.join(word.capitalize() for word in title.split())
            
            modified_front_matter = {
                'title': title,
                'tags': ['untagged']
            }
            
            # Construct new content with front matter
            modified_content = "---\n"
            modified_content += yaml.dump(modified_front_matter, default_flow_style=False)
            modified_content += "---\n\n"
            modified_content += clean_content
        
        # Fix missing title
        elif 'missing_title' in issues:
            # Use filename as title
            filename = os.path.basename(file_path)
            title = os.path.splitext(filename)[0]
            
            # Replace underscores and hyphens with spaces
            title = re.sub(r'[_-]', ' ', title)
            # Capitalize words
            title = ' '.join(word.capitalize() for word in title.split())
            
            modified_front_matter['title'] = title
            
            # Reconstruct front matter
            front_matter_str = "---\n"
            front_matter_str += yaml.dump(modified_front_matter, default_flow_style=False)
            front_matter_str += "---\n\n"
            
            # Replace original front matter
            modified_content = re.sub(r'^---\s*\n.*?\n---\s*\n', front_matter_str, content, flags=re.DOTALL)
        
        # Fix missing or empty tags
        elif 'missing_tags' in issues or 'empty_tags' in issues:
            modified_front_matter['tags'] = ['untagged']
            
            # Reconstruct front matter
            front_matter_str = "---\n"
            front_matter_str += yaml.dump(modified_front_matter, default_flow_style=False)
            front_matter_str += "---\n\n"
            
            # Replace original front matter
            modified_content = re.sub(r'^---\s*\n.*?\n---\s*\n', front_matter_str, content, flags=re.DOTALL)
        
        # Fix malformed tags
        elif 'malformed_tags' in issues:
            tags = modified_front_matter['tags']
            if isinstance(tags, str):
                modified_front_matter['tags'] = [tags]
            
            # Reconstruct front matter
            front_matter_str = "---\n"
            front_matter_str += yaml.dump(modified_front_matter, default_flow_style=False)
            front_matter_str += "---\n\n"
            
            # Replace original front matter
            modified_content = re.sub(r'^---\s*\n.*?\n---\s*\n', front_matter_str, content, flags=re.DOTALL)
        
        # Fix trailing whitespace
        if 'trailing_whitespace' in issues:
            # Remove trailing whitespace from each line
            modified_content = re.sub(r'[ \t]+$', '', modified_content, flags=re.MULTILINE)
        
        # Fix missing final newline
        if 'missing_final_newline' in issues:
            if not modified_content.endswith('\n'):
                modified_content += '\n'
        
        return modified_content

def main():
    parser = argparse.ArgumentParser(description="Markdown Note Linter")
    parser.add_argument("notes_dir", help="Directory containing markdown notes")
    parser.add_argument("--pattern", default="**/*.md", help="File pattern to match")
    parser.add_argument("--exclude", help="Pattern to exclude")
    parser.add_argument("--fix", action="store_true", help="Fix issues automatically where possible")
    parser.add_argument("--output-dir", help="Directory to save fixed files (default: modify in place)")
    parser.add_argument("--no-backup", action="store_true", help="Disable file backups before modifications")
    args = parser.parse_args()
    
    linter = NoteLinter(
        notes_dir=args.notes_dir,
        output_dir=args.output_dir,
        backup=not args.no_backup
    )
    
    linter.lint_notes(
        pattern=args.pattern, 
        exclude_pattern=args.exclude,
        fix=args.fix
    )

if __name__ == "__main__":
    main() 