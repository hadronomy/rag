from typing import Annotated

from fastapi import Depends, FastAPI, Request
from qdrant_client import AsyncQdrantClient

from .lifespan import lifespan

app = FastAPI(lifespan=lifespan)

# Example:


async def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    return request.state.qdrant_client


@app.get("/")
async def read_root(
    qdrant_client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
):
    print(f"Qdrant client: {qdrant_client}")
    return {"message": "Welcome to the RAG API!"}
