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
    qdrant_api_key: str | None = None


class ColpaliSettings(BaseCommonSettings):
    """
    Settings for the Colpali service.
    """

    model_name: str = "vidore/colqwen2.5-v0.2"


class ObjectStorageSettings(BaseCommonSettings):
    """
    Settings for the Object Storage service.
    """

    endpoint_url: str
    access_key: str
    secret_access_key: str


class OpenRouterSettings(BaseCommonSettings):
    """
    Settings for the OpenRouter service.
    """

    openrouter_api_key: str
    openrouter_url: str = "https://api.openrouter.ai/v1"
    openrouter_model: str = "google/gemini-2.0-flash-001"


class Settings(BaseCommonSettings):
    """
    Main settings class that aggregates all service settings.
    """

    model_config = SettingsConfigDict(
        env_file=find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    colpali: ColpaliSettings = Field(default_factory=ColpaliSettings)
    object_storage: ObjectStorageSettings = Field(default_factory=ObjectStorageSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)


@lru_cache()
def get_settings() -> Settings:
    """
    Get the application settings.
    """
    return Settings()
