import time
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from scalar_fastapi import get_scalar_api_reference

from rag.api.dependencies import get_qdrant_client
from rag.instrumentation import instrument_app, setup_telemetry

from .auth import api_key_header
from .lifespan import lifespan

tracer, meter = setup_telemetry()

app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation (RAG) with Qdrant",
    version="17-06-2025",
    lifespan=lifespan,
)

instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Error handling
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


class HealthStatus(BaseModel):
    status: str
    timestamp: str


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    return HealthStatus(
        status="ok",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    )


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )


@app.get("/")
async def read_root(
    qdrant_client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    api_key_header: str = Security(api_key_header),
):
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the RAG API!"}
