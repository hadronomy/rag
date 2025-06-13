from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.services import qdrant


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application."""
    # Initialize resources here if needed
    await qdrant.create_collection()
    yield
    # Cleanup resources here if needed
