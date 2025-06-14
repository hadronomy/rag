from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.api.state import State
from rag.models.loaders import ColQwen2_5Loader
from rag.services import qdrant
from rag.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    """Lifespan context manager for the FastAPI application."""
    settings = get_settings()

    loader = ColQwen2_5Loader(model_name=settings.colpali.model_name)
    model, processor = loader.load()

    qdrant_client = qdrant.create_qdrant_client(settings)

    await qdrant.create_collection(qdrant_client)

    yield State(
        model=model,
        processor=processor,
        qdrant_client=qdrant_client,
        collection_name=settings.qdrant.collection_name,
    )

    await qdrant_client.close()
