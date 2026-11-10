from fastapi import FastAPI

from app.api.v1.routes import health
from app.core.config import settings

app = FastAPI(title=settings.app_name)

app.include_router(health.router, prefix="/api/v1", tags=["health"])


