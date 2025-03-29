#!/usr/bin/env python3
"""
Human-in-the-loop note processing script for LLM Note Organizer.

This script:
1. Loads notes from a specified directory
2. Processes each note using various tagging and categorization methods
3. Presents results to the user for approval/modification
4. Saves the approved results for later use in training/tuning
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

# Optional imports - will be used if available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not found. Install with 'pip install rich' for better display.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Load configuration
def load_config(config_path: str = "category_schema.yaml") -> Dict[str, Any]:
    """Load category schema from file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

# Simple keyword-based tagger
class KeywordTagger:
    """Simple tagger using keyword matching from the category schema."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.categories = config["categories"]
        self.threshold = config["tagging"]["confidence_thresholds"]["keyword"]
        
    def extract_category_keywords(self) -> Dict[str, List[str]]:
        """Extract all keywords from the category schema."""
        all_keywords = {}
        
        def process_category(category_dict, prefix=""):
            for category_name, category_data in category_dict.items():
                full_name = f"{prefix}.{category_name}" if prefix else category_name
                if "keywords" in category_data:
                    all_keywords[full_name] = category_data["keywords"]
                if "subcategories" in category_data:
                    process_category(category_data["subcategories"], full_name)
        
        process_category(self.categories)
        return all_keywords
    
    def tag_note(self, content: str, title: Optional[str] = None) -> List[Tuple[str, float]]:
        """Tag a note using keyword matching."""
        content = content.lower()
        title_boost = 2.0  # Keywords in title count more
        
        # Get keywords for all categories
        category_keywords = self.extract_category_keywords()
        
        # Score each category
        scores = {}
        for category, keywords in category_keywords.items():
            score = 0
            content_length = len(content.split())
            
            # Count keyword occurrences in content
            for keyword in keywords:
                keyword = keyword.lower()
                # Count exact matches
                score += content.count(f" {keyword} ") * 1.0
                # Count partial matches (at word boundaries)
                score += len(re.findall(fr'\b{re.escape(keyword)}\b', content)) * 0.8
            
            # Check title if provided
            if title:
                title = title.lower()
                for keyword in keywords:
                    keyword = keyword.lower()
                    if keyword in title:
                        score += title_boost
            
            # Normalize score based on content length
            normalized_score = score / max(1, content_length / 100)
            if normalized_score > 0:
                scores[category] = min(normalized_score, 0.95)  # Cap at 0.95
        
        # Filter by threshold and sort
        results = [(category, score) for category, score in scores.items() 
                  if score >= self.threshold]
        
        return sorted(results, key=lambda x: x[1], reverse=True)

# LLM-based tagger if available
class LLMTagger:
    """LLM-based tagger using OpenAI or Anthropic."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threshold = config["tagging"]["confidence_thresholds"]["llm"]
        self.llm_config = config["llm"]
        
        # Set up LLM client
        self.openai_client = None
        self.anthropic_client = None
        
        if OPENAI_AVAILABLE:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        if ANTHROPIC_AVAILABLE:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    
    def get_category_structure(self) -> str:
        """Convert the category structure to a string format for prompts."""
        categories = self.config["categories"]
        
        result = []
        
        def process_category(category_dict, level=0, prefix=""):
            for name, data in category_dict.items():
                full_name = f"{prefix}.{name}" if prefix else name
                indent = "  " * level
                category_desc = data.get("description", "")
                result.append(f"{indent}- {full_name}: {category_desc}")
                
                if "subcategories" in data:
                    process_category(data["subcategories"], level + 1, full_name)
        
        process_category(categories)
        return "\n".join(result)
    
    def tag_note(self, content: str, title: Optional[str] = None) -> List[Tuple[str, float]]:
        """Tag a note using LLM."""
        if not self.openai_client and not self.anthropic_client:
            return []  # No LLM available
        
        # Truncate content if too long
        max_length = 8000
        if len(content) > max_length:
            content = content[:max_length] + "...[content truncated]"
        
        # Create system and user prompts
        system_prompt = self.llm_config["system_prompt"]
        category_structure = self.get_category_structure()
        
        user_prompt = f"""
Please analyze the following note and suggest the most appropriate categories from our schema.
For each category, provide a confidence score between 0.0 and 1.0.

Category Schema:
{category_structure}

Note Title: {title or "Untitled"}

Note Content:
{content}

Return your analysis in the following JSON format:
{{
  "categories": [
    {{"category": "category_name", "confidence": 0.95}},
    {{"category": "another.subcategory", "confidence": 0.8}}
  ]
}}

Only include categories with a confidence score of {self.threshold} or higher.
"""
        
        # Try OpenAI first
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_config["default_model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                if "categories" in result:
                    return [(item["category"], item["confidence"]) for item in result["categories"]]
            except Exception as e:
                print(f"Error using OpenAI for tagging: {e}")
        
        # Try Anthropic if OpenAI failed or isn't available
        if self.anthropic_client:
            try:
                prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.anthropic_client.messages.create(
                    model=self.llm_config.get("alternative_models", ["claude-3-haiku-20240307"])[0],
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract JSON from Anthropic response
                content = response.content[0].text
                json_str = re.search(r'({.*})', content, re.DOTALL)
                if json_str:
                    result = json.loads(json_str.group(0))
                    if "categories" in result:
                        return [(item["category"], item["confidence"]) for item in result["categories"]]
            except Exception as e:
                print(f"Error using Anthropic for tagging: {e}")
        
        return []

# Embedding-based tagger if available
class EmbeddingTagger:
    """Embedding-based tagger using sentence transformers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threshold = config["tagging"]["confidence_thresholds"]["embedding"]
        self.embedding_config = config["embeddings"]
        self.model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                model_name = self.embedding_config["default_model"]
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                self.model = None
    
    def tag_note(self, content: str, title: Optional[str] = None) -> List[Tuple[str, float]]:
        """Tag a note using embeddings."""
        if not self.model:
            return []  # Model not available
        
        try:
            # Truncate content if too long
            max_length = 10000
            if len(content) > max_length:
                content = content[:max_length] + "...[content truncated]"
            
            # Get category descriptions
            categories = self.config["categories"]
            category_descriptions = {}
            
            def process_category(category_dict, prefix=""):
                for name, data in category_dict.items():
                    full_name = f"{prefix}.{name}" if prefix else name
                    description = data.get("description", "")
                    keywords = data.get("keywords", [])
                    
                    # Create a rich description combining official description and keywords
                    rich_description = f"{description}. Related to: {', '.join(keywords)}"
                    category_descriptions[full_name] = rich_description
                    
                    if "subcategories" in data:
                        process_category(data["subcategories"], full_name)
            
            process_category(categories)
            
            # Create query embedding
            if title:
                # If title is provided, use it to enrich the embedding
                query_text = f"{title}. {content}"
            else:
                query_text = content
                
            query_embedding = self.model.encode(query_text)
            
            # Create embeddings for each category description
            category_embeddings = {}
            for category, description in category_descriptions.items():
                category_embeddings[category] = self.model.encode(description)
            
            # Calculate similarity scores
            scores = {}
            import numpy as np
            for category, embedding in category_embeddings.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                
                # Scale similarity to a 0-1 range (cosine similarity is between -1 and 1)
                scaled_similarity = (similarity + 1) / 2
                
                if scaled_similarity >= self.threshold:
                    scores[category] = scaled_similarity
            
            # Sort and return results
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        except Exception as e:
            print(f"Error in embedding-based tagging: {e}")
            return []

# Note processor
class NoteProcessor:
    """Process notes using multiple tagging methods."""
    
    def __init__(self, config_path: str = "category_schema.yaml"):
        self.config = load_config(config_path)
        self.keyword_tagger = KeywordTagger(self.config)
        
        # Initialize LLM tagger if available
        if OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE:
            self.llm_tagger = LLMTagger(self.config)
        else:
            self.llm_tagger = None
            print("Neither OpenAI nor Anthropic libraries available. LLM tagging disabled.")
        
        # Initialize embedding tagger if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_tagger = EmbeddingTagger(self.config)
        else:
            self.embedding_tagger = None
            print("Sentence Transformers library not available. Embedding tagging disabled.")
        
        # Set up console for rich display
        if RICH_AVAILABLE:
            self.console = Console()
        
    def extract_title_from_content(self, content: str) -> str:
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
    
    def process_note(self, file_path: str) -> Dict[str, Any]:
        """Process a single note file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title
            title = self.extract_title_from_content(content)
            
            # Get categories from different methods
            keyword_categories = self.keyword_tagger.tag_note(content, title)
            
            llm_categories = []
            if self.llm_tagger:
                llm_categories = self.llm_tagger.tag_note(content, title)
            
            embedding_categories = []
            if self.embedding_tagger:
                embedding_categories = self.embedding_tagger.tag_note(content, title)
            
            # Prepare result
            result = {
                "file_path": file_path,
                "title": title,
                "timestamp": datetime.now().isoformat(),
                "categories": {
                    "keyword": keyword_categories,
                    "llm": llm_categories,
                    "embedding": embedding_categories
                },
                "content_preview": content[:500] + ("..." if len(content) > 500 else ""),
                "approved_categories": [],
                "status": "pending"
            }
            
            return result
        
        except Exception as e:
            print(f"Error processing note {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "status": "error"
            }
    
    def display_results(self, result: Dict[str, Any]) -> None:
        """Display processing results to the user."""
        if RICH_AVAILABLE:
            self.console.print(f"\n[bold cyan]Note:[/bold cyan] {result['title']}")
            self.console.print(f"[dim]Path: {result['file_path']}[/dim]")
            
            # Display content preview
            self.console.print(Panel(result["content_preview"], title="Content Preview", border_style="dim"))
            
            # Display categories from different methods
            table = Table(title="Suggested Categories")
            table.add_column("Category", style="cyan")
            table.add_column("Keyword", style="green")
            table.add_column("LLM", style="yellow")
            table.add_column("Embedding", style="blue")
            
            # Collect all unique categories
            all_categories = set()
            for method in ["keyword", "llm", "embedding"]:
                for category, _ in result["categories"][method]:
                    all_categories.add(category)
            
            # Create lookup dictionaries for each method
            lookups = {}
            for method in ["keyword", "llm", "embedding"]:
                lookups[method] = {category: score for category, score in result["categories"][method]}
            
            # Add rows for each category
            for category in sorted(all_categories):
                table.add_row(
                    category,
                    f"{lookups['keyword'].get(category, 0):.2f}" if category in lookups["keyword"] else "-",
                    f"{lookups['llm'].get(category, 0):.2f}" if category in lookups["llm"] else "-",
                    f"{lookups['embedding'].get(category, 0):.2f}" if category in lookups["embedding"] else "-"
                )
            
            self.console.print(table)
        else:
            # Fallback to standard print
            print(f"\nNote: {result['title']}")
            print(f"Path: {result['file_path']}")
            print("\nContent Preview:")
            print(result["content_preview"])
            
            print("\nSuggested Categories:")
            print("Category".ljust(30) + "Keyword".ljust(10) + "LLM".ljust(10) + "Embedding".ljust(10))
            print("-" * 60)
            
            # Collect all unique categories
            all_categories = set()
            for method in ["keyword", "llm", "embedding"]:
                for category, _ in result["categories"][method]:
                    all_categories.add(category)
            
            # Create lookup dictionaries for each method
            lookups = {}
            for method in ["keyword", "llm", "embedding"]:
                lookups[method] = {category: score for category, score in result["categories"][method]}
            
            # Add rows for each category
            for category in sorted(all_categories):
                kw_score = f"{lookups['keyword'].get(category, 0):.2f}" if category in lookups["keyword"] else "-"
                llm_score = f"{lookups['llm'].get(category, 0):.2f}" if category in lookups["llm"] else "-"
                emb_score = f"{lookups['embedding'].get(category, 0):.2f}" if category in lookups["embedding"] else "-"
                
                print(f"{category.ljust(30)}{kw_score.ljust(10)}{llm_score.ljust(10)}{emb_score.ljust(10)}")
    
    def get_user_approval(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get user approval for categories."""
        if RICH_AVAILABLE:
            # Rich interactive prompt
            self.console.print("\n[bold]Select categories to approve:[/bold]")
            
            # Collect all unique categories with their best scores
            all_categories = {}
            for method in ["keyword", "llm", "embedding"]:
                for category, score in result["categories"][method]:
                    if category not in all_categories or score > all_categories[category][0]:
                        all_categories[category] = (score, method)
            
            # Sort by score
            sorted_categories = sorted(all_categories.items(), key=lambda x: x[1][0], reverse=True)
            
            approved = []
            for i, (category, (score, method)) in enumerate(sorted_categories):
                choice = Confirm.ask(
                    f"[cyan]{i+1}.[/cyan] {category} ({method}, {score:.2f})",
                    default=score > 0.8
                )
                if choice:
                    approved.append(category)
            
            # Allow user to add custom categories
            if Confirm.ask("\nAdd custom categories?"):
                while True:
                    custom = Prompt.ask("Enter category (or empty to finish)")
                    if not custom:
                        break
                    approved.append(custom)
            
            result["approved_categories"] = approved
            result["status"] = "approved"
            
            # Show final approval
            self.console.print("\n[bold green]Approved Categories:[/bold green]")
            for category in approved:
                self.console.print(f"- {category}")
        else:
            # Fallback to standard input
            print("\nSelect categories to approve:")
            
            # Collect all unique categories with their best scores
            all_categories = {}
            for method in ["keyword", "llm", "embedding"]:
                for category, score in result["categories"][method]:
                    if category not in all_categories or score > all_categories[category][0]:
                        all_categories[category] = (score, method)
            
            # Sort by score
            sorted_categories = sorted(all_categories.items(), key=lambda x: x[1][0], reverse=True)
            
            approved = []
            for i, (category, (score, method)) in enumerate(sorted_categories):
                default = "Y" if score > 0.8 else "N"
                choice = input(f"{i+1}. {category} ({method}, {score:.2f}) [Y/N, default: {default}]: ")
                if (choice.upper() == "Y") or (not choice and default == "Y"):
                    approved.append(category)
            
            # Allow user to add custom categories
            if input("\nAdd custom categories? [Y/N]: ").upper() == "Y":
                while True:
                    custom = input("Enter category (or empty to finish): ")
                    if not custom:
                        break
                    approved.append(custom)
            
            result["approved_categories"] = approved
            result["status"] = "approved"
            
            # Show final approval
            print("\nApproved Categories:")
            for category in approved:
                print(f"- {category}")
        
        return result
    
    def process_files(self, directory: str, output_file: str, limit: Optional[int] = None) -> None:
        """Process all markdown files in a directory."""
        # Get all markdown files
        path = Path(directory)
        md_files = list(path.glob("**/*.md"))
        
        # Apply limit if specified
        if limit and len(md_files) > limit:
            print(f"Found {len(md_files)} files, processing the first {limit}...")
            md_files = md_files[:limit]
        else:
            print(f"Found {len(md_files)} markdown files to process...")
        
        # Process files
        results = []
        for i, file_path in enumerate(md_files):
            print(f"Processing file {i+1}/{len(md_files)}: {file_path}")
            result = self.process_note(str(file_path))
            
            # Display results and get approval
            self.display_results(result)
            approved_result = self.get_user_approval(result)
            
            # Add to results
            results.append(approved_result)
            
            # Save results periodically
            if (i + 1) % 10 == 0 or i == len(md_files) - 1:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved results to {output_file}")
        
        print(f"Processed {len(results)} files. Results saved to {output_file}")

# Main function
def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process markdown notes with human-in-the-loop approval.")
    parser.add_argument("--dir", type=str, default="./notes", help="Directory containing markdown notes")
    parser.add_argument("--config", type=str, default="category_schema.yaml", help="Path to category schema config")
    parser.add_argument("--output", type=str, default="processed_notes.json", help="Output file for processed notes")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process")
    
    args = parser.parse_args()
    
    # Ensure directory exists
    if not os.path.isdir(args.dir):
        print(f"Error: Directory {args.dir} does not exist.")
        sys.exit(1)
    
    # Ensure config file exists
    if not os.path.isfile(args.config):
        print(f"Error: Config file {args.config} does not exist.")
        sys.exit(1)
    
    # Process notes
    processor = NoteProcessor(args.config)
    processor.process_files(args.dir, args.output, args.limit)

if __name__ == "__main__":
    main()