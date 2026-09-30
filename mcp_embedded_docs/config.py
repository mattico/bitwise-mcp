"""Configuration management."""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""
    # Semantic search on top of keyword search. Off means no torch import, no
    # model in memory, and ingest skips embedding; keyword search stands alone.
    enabled: bool = True
    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    batch_size: int = 32


class LLMFallbackConfig(BaseModel):
    """LLM fallback configuration."""
    enabled: bool = False
    provider: str = "openrouter"
    api_key_env: str = "OPENROUTER_API_KEY"
    model: str = "anthropic/claude-3-haiku"
    cache_results: bool = True


class ChunkingConfig(BaseModel):
    """Chunking configuration."""
    target_size: int = 2500
    overlap: int = 200
    preserve_tables: bool = True


class SearchConfig(BaseModel):
    """Search configuration."""
    # Weights of each channel in reciprocal-rank fusion.
    keyword_weight: float = 0.5
    semantic_weight: float = 0.5
    top_k_default: int = 5


class IndexConfig(BaseModel):
    """Index storage configuration."""
    directory: Path = Path("./index")
    vector_file: str = "vectors.faiss"
    metadata_db: str = "metadata.db"
    documents_file: str = "documents.json"


class Config(BaseModel):
    """Main configuration."""
    doc_dirs: List[Path] = Field(default_factory=lambda: [Path("./docs")])
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llm_fallback: LLMFallbackConfig = Field(default_factory=LLMFallbackConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from file or use defaults.

        The file is `config_path`, else $BITWISE_MCP_CONFIG, else ./config.yaml.
        Relative paths inside it resolve against the file's directory, so the
        server finds the same index whatever its working directory is.
        $BITWISE_MCP_INDEX_DIR overrides index.directory.
        """
        if config_path is None:
            env_path = os.getenv("BITWISE_MCP_CONFIG")
            config_path = Path(env_path) if env_path else Path("config.yaml")

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config = cls(**data)
            base = config_path.resolve().parent
            config.doc_dirs = [d if d.is_absolute() else base / d for d in config.doc_dirs]
            if not config.index.directory.is_absolute():
                config.index.directory = base / config.index.directory
        else:
            config = cls()

        env_index = os.getenv("BITWISE_MCP_INDEX_DIR")
        if env_index:
            config.index.directory = Path(env_index)
        return config

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if not self.llm_fallback.enabled:
            return None
        return os.getenv(self.llm_fallback.api_key_env)