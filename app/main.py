from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.routes import analyses
from app.api.v1.routes import health
from app.core.config import settings

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.cors_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(analyses.router, prefix="/api/v1", tags=["analyses"])
app.mount(
    "/",
    StaticFiles(
        directory=Path(__file__).resolve().parent.parent / "frontend",
        html=True,
    ),
    name="frontend",
)


