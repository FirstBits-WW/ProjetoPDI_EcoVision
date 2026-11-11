from datetime import date

from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}
