from fastapi import Request
from qdrant_client import AsyncQdrantClient

from rag.models.loaders import ColQwen2_5_Processor, ColQwen2_5Loader
from rag.services.image_manager import S3JPEGManager


async def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    return request.state.qdrant_client


async def get_colpali_model(request: Request) -> ColQwen2_5Loader:
    return request.state.model


async def get_colpali_processor(request: Request) -> ColQwen2_5_Processor:
    return request.state.processor


async def get_image_manager(request: Request) -> S3JPEGManager:
    return request.state.image_manager


async def get_collection_name(request: Request) -> str:
    return request.state.collection_name
