import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache()
def find_env_file() -> str:
    """Find .env file by recursively going upwards in the directory tree."""
    current_path = Path(os.getcwd())
    while current_path != current_path.parent:
        env_file = current_path / ".env"
        if env_file.exists():
            return str(env_file.absolute())
        current_path = current_path.parent
    return ".env"


class BaseCommonSettings(BaseSettings):
    """
    Base settings class with common configuration.
    """

    model_config = SettingsConfigDict(
        env_file=find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class QdrantSettings(BaseCommonSettings):
    """
    Settings for the Qdrant service.
    """

    collection_name: str = "colpali"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str


class ColpaliSettings(BaseCommonSettings):
    """
    Settings for the Colpali service.
    """

    model_name: str = "vidore/colqwen2.5-v0.2"


class SupabaseSettings(BaseCommonSettings):
    """
    Settings for the Supabase service.
    """

    supabase_url: str
    supabase_key: str
    bucket: str = "colpali"


class OpenRouterSettings(BaseCommonSettings):
    """
    Settings for the OpenRouter service.
    """

    openrouter_api_key: str
    openrouter_model: str = "openrouter/colpali-llama-2-70b-chat-v1"


class Settings(BaseCommonSettings):
    """
    Main settings class that aggregates all service settings.
    """

    model_config = SettingsConfigDict(
        env_file=find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    colpali: ColpaliSettings = Field(default_factory=ColpaliSettings)
    supabase: SupabaseSettings = Field(default_factory=SupabaseSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)


@lru_cache()
def get_settings() -> Settings:
    """
    Get the application settings.
    """
    return Settings()
