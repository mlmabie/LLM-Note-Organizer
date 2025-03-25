#!/usr/bin/env python3
"""
LLM Note Organizer - Automatically organize and optimize markdown notes for Obsidian
"""

import os
import re
import yaml
import glob
import nltk
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Optional OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Load NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class NoteOrganizer:
    """Main class for organizing and processing markdown notes."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize the note organizer with configuration."""
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Override config with environment variables if set
        if os.getenv('NOTES_DIR'):
            self.config['notes_dir'] = os.getenv('NOTES_DIR')
        if os.getenv('OUTPUT_DIR'):
            self.config['output_dir'] = os.getenv('OUTPUT_DIR')
            
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.config['embedding_model'])
        
        # Initialize OpenAI if available and configured
        if OPENAI_AVAILABLE and self.config.get('use_openai', False):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                print("Warning: OpenAI API key not found. Disabling OpenAI features.")
                self.config['use_openai'] = False
        else:
            self.config['use_openai'] = False

    def process_notes(self):
        """Process all markdown notes in the configured directory."""
        note_files = glob.glob(os.path.join(self.config['notes_dir'], "**/*.md"), recursive=True)
        
        if not note_files:
            print(f"No markdown files found in {self.config['notes_dir']}")
            return
        
        print(f"Found {len(note_files)} markdown files to process")
        
        for note_file in tqdm(note_files, desc="Processing notes"):
            self.process_single_note(note_file)
    
    def process_single_note(self, note_path):
        """Process a single note file."""
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract existing front matter and content
        front_matter, clean_content = self.extract_front_matter(content)
        
        # Analyze note content
        sections = self.split_into_sections(clean_content)
        
        # Generate embeddings for sections
        section_embeddings = self.embed_sections(sections)
        
        # Determine if note should be split
        should_split, split_points = self.determine_splits(sections, section_embeddings)
        
        # Generate tags
        tags = self.generate_tags(clean_content, sections, section_embeddings)
        
        # Update front matter with new tags
        updated_front_matter = self.update_front_matter(front_matter, tags)
        
        # Save processed note or split notes
        self.save_processed_note(note_path, updated_front_matter, sections, should_split, split_points)

    def extract_front_matter(self, content):
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

    def split_into_sections(self, content):
        """Split note content into logical sections based on headings and paragraphs."""
        # Split by headings
        heading_pattern = r'(#{1,6}\s+.+?\n)'
        sections = re.split(heading_pattern, content)
        
        # Combine headings with their content
        processed_sections = []
        for i in range(0, len(sections)-1, 2):
            if i+1 < len(sections):
                processed_sections.append(sections[i] + sections[i+1])
            else:
                processed_sections.append(sections[i])
                
        # If no headings were found, split by paragraphs
        if len(processed_sections) <= 1:
            paragraphs = re.split(r'\n\s*\n', content)
            processed_sections = [p for p in paragraphs if p.strip()]
        
        return processed_sections

    def embed_sections(self, sections):
        """Generate embeddings for each section using the lightweight model."""
        if not sections:
            return []
            
        # Generate section summaries for embedding
        section_texts = [self.summarize_section(section) for section in sections]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(section_texts)
        
        return embeddings

    def summarize_section(self, section_text, max_length=512):
        """Create a summary of the section for embedding."""
        # Extract heading if present
        heading = ""
        heading_match = re.match(r'(#{1,6}\s+(.+?))\n', section_text)
        if heading_match:
            heading = heading_match.group(2)
        
        # Get first few sentences
        sentences = sent_tokenize(section_text)
        
        # Combine heading with first few sentences
        summary = heading + ". " if heading else ""
        for sentence in sentences:
            if len(summary) + len(sentence) < max_length:
                summary += sentence + " "
            else:
                break
                
        return summary.strip()

    def determine_splits(self, sections, section_embeddings):
        """Determine if a note should be split and where."""
        should_split = False
        split_points = []
        
        # If not enough sections to consider splitting
        if len(sections) < self.config.get('min_sections_for_split', 3):
            return False, []
        
        # Check total length
        total_chars = sum(len(s) for s in sections)
        
        if total_chars > self.config.get('max_section_length', 2000) * 3:
            # Calculate section similarities
            similarities = []
            for i in range(len(section_embeddings)-1):
                sim = cosine_similarity([section_embeddings[i]], [section_embeddings[i+1]])[0][0]
                similarities.append(sim)
            
            # Find natural break points (low similarity between sections)
            threshold = np.mean(similarities) - np.std(similarities)
            for i, sim in enumerate(similarities):
                if sim < threshold:
                    split_points.append(i+1)
            
            # If we found potential split points
            if split_points:
                should_split = True
        
        return should_split, split_points

    def generate_tags(self, content, sections, section_embeddings):
        """Generate appropriate tags for the note content."""
        tags = set(self.config.get('default_tags', []))
        
        # Use OpenAI if configured
        if self.config.get('use_openai', False):
            try:
                openai_tags = self.generate_tags_with_openai(content)
                tags.update(openai_tags)
            except Exception as e:
                print(f"Error using OpenAI for tagging: {e}")
        
        # Use rule-based and embedding-based tagging as fallback or supplement
        category_tags = self.generate_tags_from_categories(content)
        tags.update(category_tags)
        
        return sorted(list(tags))

    def generate_tags_with_openai(self, content):
        """Use OpenAI to suggest tags for the note content."""
        # Truncate content if too long
        max_length = 4000 
        truncated_content = content[:max_length] + ("..." if len(content) > max_length else "")
        
        prompt = f"""
        Please analyze this markdown note content and suggest appropriate tags.
        Extract 3-7 tags that accurately represent the main topics and themes.
        Format your response as a YAML list with tag names in lowercase, without any explanation.
        
        Content:
        {truncated_content}
        """
        
        response = openai.chat.completions.create(
            model=self.config.get('openai_model', 'gpt-3.5-turbo'),
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts relevant tags from markdown notes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        tags_text = response.choices[0].message.content.strip()
        
        # Extract tags from the response
        try:
            # Try parsing as YAML
            tags = yaml.safe_load(tags_text)
            if isinstance(tags, list):
                return tags
            # If response is not a list, try to extract tags with regex
            tag_matches = re.findall(r'[-*]\s*"?([a-z0-9_-]+)"?', tags_text)
            if tag_matches:
                return tag_matches
        except Exception:
            # Fall back to simple extraction
            tag_matches = re.findall(r'#([a-z0-9_-]+)', tags_text)
            if tag_matches:
                return tag_matches
            
        return []

    def generate_tags_from_categories(self, content):
        """Generate tags based on predefined categories matching content."""
        tags = set()
        lowercased_content = content.lower()
        
        # Look for category matches
        for category, category_tags in self.config.get('tag_categories', {}).items():
            for tag in category_tags:
                # Simple text matching
                if tag.lower() in lowercased_content:
                    tags.add(tag)
        
        return tags

    def update_front_matter(self, front_matter, tags):
        """Update front matter with new tags."""
        updated_front_matter = front_matter.copy()
        
        # Merge existing tags with new tags
        existing_tags = front_matter.get('tags', [])
        if isinstance(existing_tags, str):
            existing_tags = [existing_tags]
        
        all_tags = set(existing_tags) | set(tags)
        updated_front_matter['tags'] = sorted(list(all_tags))
        
        return updated_front_matter

    def save_processed_note(self, note_path, front_matter, sections, should_split, split_points):
        """Save the processed note or split notes."""
        original_filename = os.path.basename(note_path)
        output_dir = self.config.get('output_dir', os.path.dirname(note_path))
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if not should_split:
            # Save as a single file with updated front matter
            output_path = os.path.join(output_dir, original_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("---\n")
                f.write(yaml.dump(front_matter, default_flow_style=False))
                f.write("---\n\n")
                f.write("\n".join(sections))
            print(f"Processed note saved to: {output_path}")
        else:
            # Split into multiple files
            base_name = os.path.splitext(original_filename)[0]
            split_sections = self._split_sections_at_points(sections, split_points)
            
            for i, split_content in enumerate(split_sections):
                # Create specific front matter for this split
                split_fm = front_matter.copy()
                
                # Add part number to title
                if 'title' in split_fm:
                    split_fm['title'] = f"{split_fm['title']} (Part {i+1})"
                else:
                    split_fm['title'] = f"{base_name} (Part {i+1})"
                
                # Add links to other parts
                split_fm['part'] = i + 1
                split_fm['total_parts'] = len(split_sections)
                
                # Set filename
                split_filename = f"{base_name}-part{i+1}.md"
                output_path = os.path.join(output_dir, split_filename)
                
                # Save split file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("---\n")
                    f.write(yaml.dump(split_fm, default_flow_style=False))
                    f.write("---\n\n")
                    f.write("\n".join(split_content))
                print(f"Split note part {i+1} saved to: {output_path}")

    def _split_sections_at_points(self, sections, split_points):
        """Split sections at the specified points."""
        result = []
        start_idx = 0
        
        for point in split_points:
            result.append(sections[start_idx:point])
            start_idx = point
        
        # Add the last segment
        result.append(sections[start_idx:])
        
        return result

def main():
    parser = argparse.ArgumentParser(description="LLM Note Organizer")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--file", help="Process a single markdown file instead of the whole directory")
    args = parser.parse_args()
    
    organizer = NoteOrganizer(config_path=args.config)
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            return
        print(f"Processing single file: {args.file}")
        organizer.process_single_note(args.file)
    else:
        organizer.process_notes()

if __name__ == "__main__":
    main() 