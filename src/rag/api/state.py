from typing import TypedDict

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from qdrant_client import AsyncQdrantClient

from rag.services.image_manager import S3JPEGManager


class State(TypedDict):
    """
    Represents the state of the RAG system.
    """

    model: ColQwen2_5
    processor: ColQwen2_5_Processor
    qdrant_client: AsyncQdrantClient
    image_manager: S3JPEGManager
    collection_name: str
