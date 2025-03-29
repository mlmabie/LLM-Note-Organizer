# Drafts to Obsidian Workflow Guide

This guide explains how to use LLM Note Organizer to process JSON exports from the Drafts app and organize them into a structured Obsidian vault.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process a Drafts JSON export (non-interactive mode)
python drafts_to_obsidian.py --input /path/to/DraftsExport.draftsExport --output /path/to/ObsidianVault --json --non-interactive

# Process a Drafts JSON export (interactive mode to review categories)
python drafts_to_obsidian.py --input /path/to/DraftsExport.draftsExport --output /path/to/ObsidianVault --json
```

## Step-by-Step Process

### 1. Export Your Notes from Drafts

1. Open the Drafts app on your device
2. Select the drafts you want to export (or tap "Select All" to process everything)
3. Tap the "Share" button in the bottom toolbar
4. Choose "Export JSON" from the options
5. Save the file (usually named `DraftsExport.draftsExport`) to your computer

### 2. Process the Drafts Export

```bash
# Create an output directory for your Obsidian vault (if needed)
mkdir -p ~/Documents/ObsidianVault

# Process the export file
cd /path/to/LLM\ Note\ Organizer
python drafts_to_obsidian.py --input /path/to/DraftsExport.draftsExport --output ~/Documents/ObsidianVault --json
```

This will:
1. Load the Drafts JSON export file
2. Process each draft with the AI-powered categorization system
3. Assign appropriate categories based on content analysis
4. Convert the drafts to Markdown with proper frontmatter
5. Save them to the appropriate folders in your Obsidian vault

### 3. Check the Results

To see how your notes were organized:

```bash
# View the category structure
find ~/Documents/ObsidianVault -type d | sort

# See how many notes are in each category
for dir in $(find ~/Documents/ObsidianVault -type d); do 
  count=$(find "$dir" -maxdepth 1 -type f -name "*.md" | wc -l)
  echo "$dir: $count"
done | sort -k2 -nr | head -10
```

### 4. Open Your Vault in Obsidian

1. Open Obsidian
2. Click "Open folder as vault"
3. Select your output directory (e.g., `~/Documents/ObsidianVault`)
4. Browse your organized notes

## Customizing the Category Schema

The system uses `category_schema.yaml` to determine how to categorize your notes. You can edit this file to match your preferred organizational structure:

```bash
# Edit the category schema
nano category_schema.yaml
```

After modifying the schema, you can re-run the processing to apply your new categorization rules.

## Advanced Usage

### Interactive vs. Non-Interactive Mode

- **Interactive Mode**: Reviews each note and lets you confirm categories
  ```bash
  python drafts_to_obsidian.py --input export.json --output vault --json
  ```

- **Non-Interactive Mode**: Automatically accepts suggested categories
  ```bash
  python drafts_to_obsidian.py --input export.json --output vault --json --non-interactive
  ```

### Process Only Recent Notes

To process only your most recent notes:

1. In Drafts, sort by modification date
2. Select only recent notes
3. Export as JSON
4. Process with the tool

### Watch for New Exports

Set up a watched folder to automatically process new exports:

```bash
python drafts_to_obsidian.py --input ~/Downloads --output ~/ObsidianVault --watch --json
```

## Comparing Classification Methods

To evaluate different classification approaches:

```bash
# Process a sample of notes with human approval
python process_notes.py --dir ~/Notes --limit 20

# Compare classifier performance
python compare_classifiers.py

# View the plots
open ./plots/
```

## Troubleshooting

- **Error reading JSON**: Ensure your export file is valid (try re-exporting)
- **Missing categories**: Edit the category schema to include your preferred categories
- **Misclassified notes**: Run in interactive mode to review and correct classifications