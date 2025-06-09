from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class QdrantSettings(BaseSettings):
    """
    Settings for the Qdrant service.
    """

    collection_name: str = Field(
        "colpali",
        validation_alias=AliasChoices("qdrant_collection_name", "collection_name"),
    )
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str


class ColpaliSettings(BaseSettings):
    """
    Settings for the Colpali service.
    """

    model_name: str = "vidore/colqwen2.5-v0.2"


class SupabaseSettings(BaseSettings):
    """
    Settings for the Supabase service.
    """

    supabase_url: str
    supabase_key: str
    bucket: str = "colpali"


class OpenRouterSettings(BaseSettings):
    """
    Settings for the OpenRouter service.
    """

    openrouter_api_key: str
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
