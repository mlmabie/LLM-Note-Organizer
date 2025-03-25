"""
Configuration management for Note Organizer.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

import yaml
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    validator,
    SecretStr,
    DirectoryPath,
)
from pydantic_settings import BaseSettings

from note_organizer.services.tagging import LLMProvider


class LogConfig(BaseModel):
    """Log configuration."""

    level: str = "INFO"
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    rotation: str = "20 MB"
    retention: str = "1 week"
    path: str = "logs"


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = "all-MiniLM-L6-v2"
    use_cce: bool = True
    cce_centroids: int = Field(1024, ge=100, le=10000)  # Number of centroids for CCE
    cce_dim: int = Field(64, ge=16, le=128)  # Dimension for CCE vectors
    cache_dir: str = ".cache"


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""

    api_key: Optional[SecretStr] = None
    model: str = "gpt-3.5-turbo"
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(150, ge=1, le=4000)
    timeout: int = Field(30, ge=1, le=600)


class ClaudeConfig(BaseModel):
    """Anthropic Claude configuration."""

    api_key: Optional[SecretStr] = None
    model: str = "claude-instant-1"
    temperature: float = Field(0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(150, ge=1, le=4000)
    timeout: int = Field(30, ge=1, le=600)


class GoogleAIConfig(BaseModel):
    """Google AI configuration."""

    api_key: Optional[SecretStr] = None
    model: str = "gemini-pro"
    temperature: float = Field(0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(150, ge=1, le=2048)
    timeout: int = Field(30, ge=1, le=600)


class TaggingConfig(BaseModel):
    """Auto-tagging configuration."""

    use_dspy: bool = True
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_tags_per_note: int = Field(10, ge=1, le=50)
    default_tags: List[str] = Field(default_factory=lambda: ["inbox", "needs_processing"])
    tag_categories: Dict[str, List[str]] = Field(default_factory=dict)


class NoteProcessingConfig(BaseModel):
    """Note processing configuration."""

    min_section_length: int = Field(300, ge=50)  # Minimum characters for a section
    max_section_length: int = Field(2000, ge=500)  # Maximum recommended characters per section
    min_sections_for_split: int = Field(3, ge=2)  # Minimum number of sections to consider splitting


class APIConfig(BaseModel):
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    cors_origins: List[str] = Field(default_factory=list)
    allowed_hosts: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    enable_docs: bool = True
    workers: int = Field(4, ge=1, le=32)
    

class DBConfig(BaseModel):
    """Database configuration."""

    url: str = "sqlite:///./note_organizer.db"
    min_connections: int = Field(1, ge=1)
    max_connections: int = Field(10, ge=1)
    use_cache: bool = True
    cache_ttl: int = Field(300, ge=1)  # 5 minutes in seconds


class Settings(BaseSettings):
    """Application settings."""

    # Base settings
    app_name: str = "note_organizer"
    debug: bool = False
    environment: str = "dev"
    
    # Directory settings
    notes_dir: str = Field(default_factory=lambda: os.path.join(os.path.expanduser("~"), "notes"))
    output_dir: Optional[str] = None
    
    # LLM provider
    llm_provider: LLMProvider = LLMProvider.NONE
    
    # Specific configurations
    log: LogConfig = Field(default_factory=LogConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    tagging: TaggingConfig = Field(default_factory=TaggingConfig)
    note_processing: NoteProcessingConfig = Field(default_factory=NoteProcessingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    db: DBConfig = Field(default_factory=DBConfig)
    
    # LLM configurations
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    google: GoogleAIConfig = Field(default_factory=GoogleAIConfig)

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        use_enum_values = True
        
    @validator("output_dir", pre=True)
    def set_output_dir(cls, v, values):
        """Set output_dir to notes_dir if not specified."""
        if not v:
            return values.get("notes_dir")
        return v
    
    @validator("notes_dir", "output_dir")
    def create_directory_if_missing(cls, v):
        """Create directory if it doesn't exist."""
        if v:
            os.makedirs(v, exist_ok=True)
        return v
    
    @property
    def active_llm_api_key(self) -> Optional[str]:
        """Get the active LLM API key based on the selected provider."""
        if self.llm_provider == LLMProvider.OPENAI and self.openai.api_key:
            return self.openai.api_key.get_secret_value() if self.openai.api_key else None
        elif self.llm_provider == LLMProvider.CLAUDE and self.claude.api_key:
            return self.claude.api_key.get_secret_value() if self.claude.api_key else None
        elif self.llm_provider == LLMProvider.GOOGLE and self.google.api_key:
            return self.google.api_key.get_secret_value() if self.google.api_key else None
        return None


def load_config(config_path: Union[str, Path]) -> Settings:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    # Convert api_key values to SecretStr if they're plain strings
    for provider in ["openai", "claude", "google"]:
        if provider in config_data and "api_key" in config_data[provider]:
            key = config_data[provider]["api_key"]
            if isinstance(key, str):
                config_data[provider]["api_key"] = key
    
    return Settings(**config_data)


# Create a global instance of settings
settings = Settings()


def update_settings(new_settings: Settings):
    """Update the global settings instance."""
    global settings
    settings = new_settings 