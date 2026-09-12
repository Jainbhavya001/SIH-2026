"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000
(from the `backend/` directory, with the virtualenv activated)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_documents, routes_face, routes_health
from app.config import get_settings
from app.utils.logging_config import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-Based Fake Identity & Document Screening System — OCR extraction, "
        "document validation, tampering detection, and face verification for "
        "border checkpoint screening. Prototype for SIH Problem Statement 26188."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_documents.router)
app.include_router(routes_face.router)


@app.get("/")
async def root():
    return {
        "message": f"{settings.APP_NAME} API",
        "docs": "/docs",
        "health": "/api/health",
    }
