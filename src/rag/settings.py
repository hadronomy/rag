import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class QdrantSettings(BaseSettings):
    """
    Settings for the Qdrant service.
    """

    collection_name: str = os.environ.get(
        "QDRANT_COLLECTION_NAME", "default_collection"
    )
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.environ.get("QDRANT_API_KEY", "")


class ColpaliSettings(BaseSettings):
    """
    Settings for the Colpali service.
    """

    model_name: str = "vidore/colqwen2.5-v0.2"


class SupabaseSettings(BaseSettings):
    """
    Settings for the Supabase service.
    """

    supabase_url: str = os.environ.get("SUPABASE_URL", "")
    supabase_key: str = os.environ.get("SUPABASE_KEY", "")
    bucket: str = "colpali"


class OpenRouterSettings(BaseSettings):
    """
    Settings for the OpenRouter service.
    """

    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_model: str = "openrouter/colpali-llama-2-70b-chat-v1"


class Settings(BaseSettings):
    """
    Main settings class that aggregates all service settings.
    """

    qdrant: QdrantSettings = QdrantSettings()
    colpali: ColpaliSettings = ColpaliSettings()
    supabase: SupabaseSettings = SupabaseSettings()
    openrouter: OpenRouterSettings = OpenRouterSettings()


@lru_cache()
def get_settings() -> Settings:
    """
    Get the application settings.
    """
    return Settings()
