# LLM Note Organizer

A tool for organizing your notes from Drafts app into Obsidian with AI-powered categorization and tagging.

## Features

- **AI-powered categorization**: Automatically suggests categories for your notes
- **Multiple classification methods**: Uses keyword matching, LLMs, and embeddings
- **Human-in-the-loop approval**: Review and approve suggested categories
- **Drafts JSON export support**: Works directly with Drafts app exports
- **Obsidian integration**: Organizes notes into appropriate folders
- **Category evaluation**: Compare the accuracy of different classification methods

## Getting Started

### Prerequisites

- Python 3.8+
- Drafts app (for exporting notes)
- Obsidian (for viewing organized notes)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llm-note-organizer.git
   cd llm-note-organizer
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Set up API keys for improved categorization:
   ```
   export OPENAI_API_KEY="your-openai-key"  # for LLM tagging
   export ANTHROPIC_API_KEY="your-anthropic-key"  # alternative for LLM tagging
   ```

### Exporting from Drafts

1. Open the Drafts app on your device
2. Select the drafts you want to export (or select all)
3. Tap the "Share" button and choose "Export JSON"
4. Save the JSON file to your computer

### Using the Tool

#### Processing Drafts JSON Export

```bash
python drafts_to_obsidian.py --input drafts_export.json --output /path/to/obsidian/vault
```

This will:
1. Load the Drafts JSON export
2. Process each draft with the AI categorization system
3. Ask for your approval on categories (in interactive mode)
4. Save the drafts to your Obsidian vault in organized folders

#### Command-Line Options

```
--input PATH          Input file (JSON or Markdown) or directory
--output PATH         Obsidian vault root directory
--schema PATH         Category schema file (default: category_schema.yaml)
--force               Overwrite existing files
--non-interactive     Run without user interaction
--recursive           Process directories recursively
--move                Move originals to archive after processing
--watch               Watch directory for new files
--interval SECONDS    Watch interval in seconds (default: 30)
--json                Force processing as Drafts JSON export
```

#### Testing Different Classifiers

To evaluate and improve the classification system:

1. Process a sample of notes with human approval:
   ```bash
   python process_notes.py --dir /path/to/notes --limit 20
   ```

2. Compare classifier performance:
   ```bash
   python compare_classifiers.py
   ```

3. View the generated plots in the `./plots` directory

## Customizing Categories

Edit `category_schema.yaml` to customize the category structure. The schema includes:

- Hierarchical categories and subcategories
- Keywords for each category
- Tagging confidence thresholds
- Embedding model configuration
- LLM settings for tagging

## Using with iCloud

For a streamlined workflow with iCloud:

1. Export drafts from Drafts app to iCloud Drive
2. Set up a watched folder:
   ```bash
   python drafts_to_obsidian.py --input /path/to/icloud/drafts --output /path/to/obsidian --watch
   ```
3. New drafts will be automatically processed when they appear in the folder

## Examples

### Basic Usage

```bash
# Process a Drafts JSON export
python drafts_to_obsidian.py --input my_drafts.json --output ~/Documents/Obsidian

# Process a directory of markdown files
python drafts_to_obsidian.py --input ~/Downloads/drafts --output ~/Documents/Obsidian --recursive

# Non-interactive mode for batch processing
python drafts_to_obsidian.py --input drafts_export.json --output ~/Obsidian --non-interactive
```

### Advanced Usage

```bash
# Process notes with human review to train the system
python process_notes.py --dir ~/Notes --output processed.json

# Compare classifier performance
python compare_classifiers.py --input processed.json --plot-dir ./analysis

# Run with a custom category schema
python drafts_to_obsidian.py --input drafts.json --output ~/Obsidian --schema my_schema.yaml
```

## License

MIT