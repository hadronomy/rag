from loguru import logger
from qdrant_client import AsyncQdrantClient, models
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.settings import Settings, get_settings


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant.qdrant_url,
        api_key=settings.qdrant.qdrant_api_key,
    )


async def create_collection(qdrant_client: AsyncQdrantClient):
    """
    Create a **Qdrant** collection if it does not exist.
    """
    settings = get_settings()

    collection_name = settings.qdrant.collection_name

    if await qdrant_client.collection_exists(collection_name):
        logger.warning(
            f"Collection '{collection_name}' already exists. Skipping creation."
        )
        return

    logger.info(f"Creating collection '{collection_name}'...")

    await qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            on_disk=False,
        ),
        on_disk_payload=False,
    )

    logger.info(f"Collection '{collection_name}' created successfully.")

    await qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="session_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def upsert_with_retry(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    points: list[models.PointStruct],
) -> None:
    await qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )
