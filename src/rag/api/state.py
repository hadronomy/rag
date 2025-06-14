from typing import TypedDict

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from qdrant_client import AsyncQdrantClient


class State(TypedDict):
    """
    Represents the state of the RAG system.
    """

    model: ColQwen2_5
    processor: ColQwen2_5_Processor
    qdrant_client: AsyncQdrantClient
    collection_name: str
