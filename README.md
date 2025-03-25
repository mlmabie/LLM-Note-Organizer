# Note Organizer

A powerful note organization system for Obsidian and other markdown-based note systems, using modern embedding techniques and LLM capabilities.

## Features

- **Semantic Search**: Find notes based on meaning, not just keywords
- **Automatic Tagging**: Intelligent tag suggestions using embeddings and LLMs
- **Markdown Processing**: Parses and processes sloppy markdown content into sections
- **Efficient Embeddings**: Uses Clustered Compositional Embeddings (CCE) for fast, memory-efficient embeddings
- **RESTful API**: Access all functionality through a well-documented API
- **File Upload**: Easily import your existing markdown files
- **Front Matter Support**: Extracts and processes YAML front matter

## Installation

### Requirements

- Python 3.9+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/note-organizer.git
   cd note-organizer
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Initialize the system:
   ```bash
   python -m note_organizer init
   ```

This will create a default configuration file at `~/.config/note_organizer/config.yaml` and set up the database.

## Configuration

The default configuration is created at `~/.config/note_organizer/config.yaml` and includes:

```yaml
debug: true
notes_dir: /path/to/your/notes
database:
  url: sqlite:///notes.db
  echo: false
api:
  host: 127.0.0.1
  port: 8000
  cors_origins:
  - http://localhost:3000
embedding:
  model_name: all-MiniLM-L6-v2
  use_cce: true
  cache_dir: .cache
openai:
  api_key: ''
  model: gpt-3.5-turbo
log:
  level: INFO
  path: logs
```

You should edit this file to:
- Set your `notes_dir` to point to your Obsidian vault or other markdown notes location
- Add your OpenAI API key if you want to use OpenAI for tagging (optional)

## Usage

### Start the API Server

```bash
python -m note_organizer api
```

This will start a FastAPI server on http://127.0.0.1:8000 (or your configured host/port).

### Process Your Notes

```bash
python -m note_organizer process --path /path/to/your/notes
```

This will scan your notes directory, index all markdown files, and process them.

### API Documentation

Once the server is running, you can access the API documentation at:
- http://127.0.0.1:8000/docs

### Example API Calls

#### Get all tags

```bash
curl -X GET http://127.0.0.1:8000/tags
```

#### Search notes

```bash
curl -X GET "http://127.0.0.1:8000/search?query=your%20search%20query"
```

#### Create a new note

```bash
curl -X POST http://127.0.0.1:8000/notes \
  -H "Content-Type: application/json" \
  -d '{"title":"New Note","content":"# New Note\n\nThis is a new note.","tags":["example"]}'
```

## Architecture

The system is organized into several modules:

- **API**: FastAPI server and routes
- **Core**: Configuration and utility functions
- **DB**: Database models and connections
- **Services**: Business logic and processing

### Key Services

- **EmbeddingService**: Generates and manages text embeddings using sentence-transformers and CCE
- **TaggingService**: Creates and manages tags using rule-based and LLM approaches
- **ProcessorService**: Processes markdown files, extracts sections, and manages the processing pipeline

## Development

### Running Tests

```bash
pytest
```

### Building Documentation

```bash
cd docs
make html
```

## License

MIT License

## Acknowledgements

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) for providing embedding models
- [DSPy](https://github.com/stanfordnlp/dspy) for LLM programming
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [SQLAlchemy](https://www.sqlalchemy.org/) for database ORM
- [PyYAML](https://pyyaml.org/) for YAML processing 
