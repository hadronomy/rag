from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.api.state import State
from rag.models.loaders import ColQwen2_5Loader
from rag.services import qdrant
from rag.services.image_manager import S3JPEGManager
from rag.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    """Lifespan context manager for the FastAPI application."""
    settings = get_settings()

    qdrant_client = qdrant.create_qdrant_client(settings)

    # Initialize image manager using async context manager
    async with S3JPEGManager(
        bucket_name="rag-images",
        endpoint_url=settings.object_storage.endpoint_url,
        access_key_id=settings.object_storage.access_key,
        secret_access_key=settings.object_storage.secret_access_key,
    ) as image_manager:
        await qdrant.create_collection(qdrant_client)

        # Use context manager for proper resource cleanup
        with ColQwen2_5Loader(model_name=settings.colpali.model_name) as loader:
            model, processor = loader.load()

            yield State(
                model=model,
                processor=processor,
                qdrant_client=qdrant_client,
                image_manager=image_manager,
                collection_name=settings.qdrant.collection_name,
            )

    await qdrant_client.close()
