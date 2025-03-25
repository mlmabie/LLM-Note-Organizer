#!/usr/bin/env python3
"""
Markdown Notes Analyzer - Analyze markdown notes to provide insights about your collection
"""

import os
import re
import yaml
import glob
import nltk
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Load NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class NotesAnalyzer:
    """Analyzes a collection of markdown notes to provide insights."""
    
    def __init__(self, notes_dir, output_dir='./note_analysis', config_path=None):
        """Initialize the notes analyzer with configuration."""
        self.notes_dir = notes_dir
        self.output_dir = output_dir
        self.config = {}
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up stop words for text analysis
        self.stop_words = set(stopwords.words('english'))
        
    def analyze(self):
        """Perform comprehensive analysis on the notes collection."""
        print("Starting notes analysis...")
        
        # Collect all markdown files
        note_files = glob.glob(os.path.join(self.notes_dir, "**/*.md"), recursive=True)
        
        if not note_files:
            print(f"No markdown files found in {self.notes_dir}")
            return
        
        print(f"Found {len(note_files)} markdown files to analyze")
        
        # Extract metadata and content
        notes_data = self._extract_notes_data(note_files)
        
        # Perform analyses
        print("Generating statistics...")
        stats = self._generate_statistics(notes_data)
        self._save_statistics(stats)
        
        print("Analyzing tags...")
        tags_analysis = self._analyze_tags(notes_data)
        self._save_tags_analysis(tags_analysis)
        
        print("Creating content clusters...")
        clusters = self._cluster_notes(notes_data)
        self._save_clusters(clusters)
        
        print("Generating visualizations...")
        self._generate_visualizations(notes_data, tags_analysis, clusters)
        
        print(f"Analysis complete. Results saved to {self.output_dir}")
        
        # Print summary
        self._print_summary(stats, tags_analysis, clusters)
    
    def _extract_notes_data(self, note_files):
        """Extract metadata and content from note files."""
        notes_data = []
        
        for file_path in tqdm(note_files, desc="Extracting note data"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                front_matter, clean_content = self._extract_front_matter(content)
                
                # Get word count and creation date
                word_count = len(re.findall(r'\b\w+\b', clean_content))
                creation_date = front_matter.get('date', front_matter.get('created', None))
                
                # Extract tags
                tags = front_matter.get('tags', [])
                if isinstance(tags, str):
                    tags = [tags]
                
                # Create a summary for embedding
                summary = self._create_note_summary(front_matter, clean_content)
                
                notes_data.append({
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'title': front_matter.get('title', os.path.splitext(os.path.basename(file_path))[0]),
                    'content': clean_content,
                    'front_matter': front_matter,
                    'tags': tags,
                    'word_count': word_count,
                    'creation_date': creation_date,
                    'summary': summary
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return notes_data
    
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
    
    def _create_note_summary(self, front_matter, content, max_length=1000):
        """Create a summary of the note for embedding."""
        # Start with title if available
        summary = front_matter.get('title', '') + ". "
        
        # Add description if available
        if 'description' in front_matter:
            summary += front_matter['description'] + " "
        
        # Get first few paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        for paragraph in paragraphs:
            # Skip code blocks and very short paragraphs
            if re.match(r'```', paragraph) or len(paragraph.strip()) < 20:
                continue
                
            # Remove markdown formatting
            clean_paragraph = re.sub(r'[#*_`~\[\]\(\)\{\}]', '', paragraph)
            
            if len(summary) + len(clean_paragraph) < max_length:
                summary += clean_paragraph + " "
            else:
                # Add as much as possible up to the limit
                remaining = max_length - len(summary)
                if remaining > 50:  # Only add if we can add a meaningful amount
                    summary += clean_paragraph[:remaining] + "..."
                break
                
        return summary.strip()
    
    def _generate_statistics(self, notes_data):
        """Generate statistical information about the notes collection."""
        stats = {
            'total_notes': len(notes_data),
            'total_words': sum(note['word_count'] for note in notes_data),
            'avg_words_per_note': sum(note['word_count'] for note in notes_data) / max(1, len(notes_data)),
            'notes_with_tags': sum(1 for note in notes_data if note['tags']),
            'notes_without_tags': sum(1 for note in notes_data if not note['tags']),
            'unique_tags': len(set(tag for note in notes_data for tag in note['tags'])),
            'word_freq': Counter(),
            'shortest_note': min(notes_data, key=lambda x: x['word_count']) if notes_data else None,
            'longest_note': max(notes_data, key=lambda x: x['word_count']) if notes_data else None,
        }
        
        # Word frequency analysis
        for note in notes_data:
            words = word_tokenize(note['content'].lower())
            # Filter out stop words and non-alphabetic words
            filtered_words = [word for word in words if word.isalpha() and word not in self.stop_words]
            stats['word_freq'].update(filtered_words)
        
        return stats
    
    def _analyze_tags(self, notes_data):
        """Analyze tag usage and relationships in the notes collection."""
        tag_analysis = {
            'tag_counts': Counter(),
            'co_occurring_tags': defaultdict(Counter),
            'suggestions': []
        }
        
        # Count tag occurrences
        all_tags = []
        for note in notes_data:
            tag_analysis['tag_counts'].update(note['tags'])
            all_tags.extend(note['tags'])
            
            # Analyze co-occurring tags
            for i, tag1 in enumerate(note['tags']):
                for tag2 in note['tags'][i+1:]:
                    tag_analysis['co_occurring_tags'][tag1][tag2] += 1
                    tag_analysis['co_occurring_tags'][tag2][tag1] += 1
        
        # Suggest tags for notes without tags
        for note in notes_data:
            if not note['tags']:
                # Find similar notes with tags
                similar_notes = self._find_similar_notes(note, notes_data, max_results=5)
                potential_tags = Counter()
                for sim, similar_note in similar_notes:
                    for tag in similar_note['tags']:
                        potential_tags[tag] += sim  # Weight by similarity
                
                # Suggest the top tags
                if potential_tags:
                    suggested_tags = [tag for tag, _ in potential_tags.most_common(3)]
                    tag_analysis['suggestions'].append({
                        'note': note['path'],
                        'title': note['title'],
                        'suggested_tags': suggested_tags
                    })
        
        return tag_analysis
    
    def _find_similar_notes(self, target_note, notes_data, max_results=5):
        """Find notes similar to the target note using embeddings."""
        # Generate embedding for the target note
        target_embedding = self.embedding_model.encode(target_note['summary'])
        
        similarities = []
        for note in notes_data:
            if note['path'] == target_note['path']:
                continue  # Skip the same note
                
            # Get or generate embedding
            if not hasattr(note, 'embedding'):
                note['embedding'] = self.embedding_model.encode(note['summary'])
                
            # Calculate similarity
            sim = cosine_similarity([target_embedding], [note['embedding']])[0][0]
            similarities.append((sim, note))
        
        # Sort by similarity and return top matches
        similarities.sort(reverse=True)
        return similarities[:max_results]
    
    def _cluster_notes(self, notes_data):
        """Cluster notes based on content similarity."""
        if len(notes_data) < 5:
            return {'error': 'Not enough notes for meaningful clustering'}
            
        # Generate embeddings for all notes
        embeddings = []
        for note in tqdm(notes_data, desc="Generating embeddings"):
            note['embedding'] = self.embedding_model.encode(note['summary'])
            embeddings.append(note['embedding'])
            
        embeddings_array = np.array(embeddings)
        
        # Determine optimal number of clusters
        max_clusters = min(20, len(notes_data) // 3)  # Set a reasonable upper limit
        if max_clusters < 2:
            max_clusters = 2
            
        scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings_array)
            scores.append(kmeans.inertia_)
        
        # Find elbow point (simplified method)
        k = 2  # Default
        if len(scores) > 1:
            diffs = np.diff(scores)
            elbow_idx = np.argmax(np.diff(diffs)) if len(diffs) > 1 else 0
            k = elbow_idx + 3  # +3 because we started at k=2 and need to account for the diff operations
            
        # Cluster the notes
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(embeddings_array)
        
        # Organize results
        cluster_results = {
            'num_clusters': k,
            'clusters': defaultdict(list)
        }
        
        for i, cluster_id in enumerate(clusters):
            cluster_results['clusters'][int(cluster_id)].append(notes_data[i])
            
        # Generate topic labels for clusters
        for cluster_id, cluster_notes in cluster_results['clusters'].items():
            # Get most common words in cluster
            word_freq = Counter()
            for note in cluster_notes:
                words = word_tokenize(note['summary'].lower())
                # Filter out stop words and non-alphabetic words
                filtered_words = [word for word in words if word.isalpha() and word not in self.stop_words]
                word_freq.update(filtered_words)
                
            # Generate topic label from top words
            top_words = [word for word, _ in word_freq.most_common(5)]
            cluster_results['clusters'][cluster_id] = {
                'notes': cluster_notes,
                'topic': ', '.join(top_words),
                'size': len(cluster_notes)
            }
            
        return cluster_results
    
    def _save_statistics(self, stats):
        """Save statistical information to a file."""
        output_path = os.path.join(self.output_dir, 'statistics.md')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Note Collection Statistics\n\n")
            f.write(f"**Total Notes:** {stats['total_notes']}\n\n")
            f.write(f"**Total Words:** {stats['total_words']}\n\n")
            f.write(f"**Average Words per Note:** {stats['avg_words_per_note']:.1f}\n\n")
            f.write(f"**Notes with Tags:** {stats['notes_with_tags']} ({stats['notes_with_tags']/max(1, stats['total_notes'])*100:.1f}%)\n\n")
            f.write(f"**Notes without Tags:** {stats['notes_without_tags']} ({stats['notes_without_tags']/max(1, stats['total_notes'])*100:.1f}%)\n\n")
            f.write(f"**Unique Tags:** {stats['unique_tags']}\n\n")
            
            if stats['shortest_note']:
                f.write("## Shortest Note\n\n")
                f.write(f"**Title:** {stats['shortest_note']['title']}\n\n")
                f.write(f"**Path:** {stats['shortest_note']['path']}\n\n")
                f.write(f"**Word Count:** {stats['shortest_note']['word_count']}\n\n")
            
            if stats['longest_note']:
                f.write("## Longest Note\n\n")
                f.write(f"**Title:** {stats['longest_note']['title']}\n\n")
                f.write(f"**Path:** {stats['longest_note']['path']}\n\n")
                f.write(f"**Word Count:** {stats['longest_note']['word_count']}\n\n")
            
            f.write("## Most Common Words\n\n")
            for word, count in stats['word_freq'].most_common(30):
                f.write(f"- {word}: {count}\n")
    
    def _save_tags_analysis(self, tags_analysis):
        """Save tag analysis to a file."""
        output_path = os.path.join(self.output_dir, 'tags_analysis.md')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Tag Analysis\n\n")
            
            f.write("## Tag Frequency\n\n")
            for tag, count in tags_analysis['tag_counts'].most_common():
                f.write(f"- {tag}: {count}\n")
            
            f.write("\n## Co-occurring Tags\n\n")
            for tag, co_tags in tags_analysis['co_occurring_tags'].items():
                if co_tags:
                    f.write(f"### {tag}\n\n")
                    for co_tag, count in co_tags.most_common(5):
                        f.write(f"- {co_tag}: {count}\n")
                    f.write("\n")
            
            f.write("\n## Tag Suggestions for Untagged Notes\n\n")
            for suggestion in tags_analysis['suggestions']:
                f.write(f"### {suggestion['title']}\n\n")
                f.write(f"**Path:** {suggestion['note']}\n\n")
                f.write(f"**Suggested Tags:** {', '.join(suggestion['suggested_tags'])}\n\n")
    
    def _save_clusters(self, clusters):
        """Save cluster analysis to a file."""
        output_path = os.path.join(self.output_dir, 'note_clusters.md')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Note Clusters\n\n")
            
            if 'error' in clusters:
                f.write(f"**Error:** {clusters['error']}\n")
                return
                
            f.write(f"**Number of Clusters:** {clusters['num_clusters']}\n\n")
            
            for cluster_id, cluster_data in clusters['clusters'].items():
                f.write(f"## Cluster {cluster_id+1}: {cluster_data['topic']}\n\n")
                f.write(f"**Size:** {cluster_data['size']} notes\n\n")
                
                f.write("### Notes in this cluster\n\n")
                for note in cluster_data['notes']:
                    tags_str = f" [*{', '.join(note['tags'])}*]" if note['tags'] else ""
                    f.write(f"- {note['title']}{tags_str} ({note['word_count']} words)\n")
                f.write("\n")
    
    def _generate_visualizations(self, notes_data, tags_analysis, clusters):
        """Generate visualizations for the analysis."""
        # Word cloud
        if notes_data:
            all_text = ' '.join(note['content'] for note in notes_data)
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                stopwords=self.stop_words,
                max_words=100
            ).generate(all_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'wordcloud.png'), dpi=300)
            plt.close()
        
        # Tag frequency chart
        if tags_analysis['tag_counts']:
            top_tags = dict(tags_analysis['tag_counts'].most_common(15))
            
            plt.figure(figsize=(12, 6))
            plt.bar(top_tags.keys(), top_tags.values())
            plt.xticks(rotation=45, ha='right')
            plt.title('Most Common Tags')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'tag_frequency.png'), dpi=300)
            plt.close()
        
        # Note length distribution
        if notes_data:
            word_counts = [note['word_count'] for note in notes_data]
            
            plt.figure(figsize=(10, 5))
            plt.hist(word_counts, bins=20)
            plt.title('Note Length Distribution')
            plt.xlabel('Word Count')
            plt.ylabel('Number of Notes')
            plt.axvline(np.mean(word_counts), color='r', linestyle='dashed', linewidth=1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'note_length_distribution.png'), dpi=300)
            plt.close()
        
        # Cluster visualization with PCA
        if 'num_clusters' in clusters and notes_data:
            # Collect embeddings and cluster assignments
            embeddings = []
            cluster_labels = []
            
            for cluster_id, cluster_data in clusters['clusters'].items():
                for note in cluster_data['notes']:
                    embeddings.append(note['embedding'])
                    cluster_labels.append(cluster_id)
            
            # Apply PCA to reduce to 2 dimensions
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            # Plot clusters
            plt.figure(figsize=(12, 8))
            for cluster_id in set(cluster_labels):
                # Get points belonging to this cluster
                mask = [label == cluster_id for label in cluster_labels]
                points = reduced_embeddings[mask]
                
                # Plot points
                plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster_id+1}", alpha=0.7)
            
            plt.title('Note Clusters (PCA Visualization)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'note_clusters.png'), dpi=300)
            plt.close()
    
    def _print_summary(self, stats, tags_analysis, clusters):
        """Print a summary of the analysis to the console."""
        print("\n=== Notes Analysis Summary ===")
        print(f"Total Notes: {stats['total_notes']}")
        print(f"Total Words: {stats['total_words']}")
        print(f"Average Words per Note: {stats['avg_words_per_note']:.1f}")
        print(f"Notes with Tags: {stats['notes_with_tags']} ({stats['notes_with_tags']/max(1, stats['total_notes'])*100:.1f}%)")
        print(f"Notes without Tags: {stats['notes_without_tags']} ({stats['notes_without_tags']/max(1, stats['total_notes'])*100:.1f}%)")
        
        print("\nTop 5 Tags:")
        for tag, count in tags_analysis['tag_counts'].most_common(5):
            print(f"- {tag}: {count}")
        
        print("\nTag Suggestions Provided for:", len(tags_analysis['suggestions']), "notes")
        
        if 'num_clusters' in clusters:
            print(f"\nNotes organized into {clusters['num_clusters']} clusters")
            for cluster_id, cluster_data in clusters['clusters'].items():
                print(f"- Cluster {cluster_id+1}: {cluster_data['topic']} ({cluster_data['size']} notes)")
        
        print(f"\nFull analysis results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Markdown Notes Analyzer")
    parser.add_argument("notes_dir", help="Directory containing markdown notes to analyze")
    parser.add_argument("--output", default="./note_analysis", help="Directory to save analysis results")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()
    
    analyzer = NotesAnalyzer(
        notes_dir=args.notes_dir,
        output_dir=args.output,
        config_path=args.config
    )
    
    analyzer.analyze()

if __name__ == "__main__":
    main() 