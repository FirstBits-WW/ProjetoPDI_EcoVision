from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.schemas.analysis import AnalysisResponse, ComparisonResponse
from app.services.analysis import analyze_image, compare_conditions
from app.services.detector import DetectorUnavailableError
from app.services.image_processing import InvalidImageError

router = APIRouter()


def _read_upload(image: UploadFile) -> bytes:
    image_bytes = image.file.read(settings.max_image_size_bytes + 1)
    if len(image_bytes) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "Imagem muito grande. O limite é "
                f"{settings.max_image_size_bytes // (1024 * 1024)} MB."
            ),
        )
    return image_bytes


@router.post("/analyses/compare", response_model=ComparisonResponse)
def compare_analysis(
    image: UploadFile = File(...),
    gamma: float = Form(default=1.25, gt=0.0, le=3.0),
) -> ComparisonResponse:
    """Original × CLAHE × gama na mesma imagem, com o mesmo detector."""
    image_bytes = _read_upload(image)
    try:
        return compare_conditions(image_bytes, gamma=gamma)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DetectorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/analyses", response_model=AnalysisResponse)
def create_analysis(
    image: UploadFile = File(...),
    gamma: float = Form(default=1.0, gt=0.0, le=3.0),
    contrast: float = Form(default=1.0, ge=0.5, le=2.0),
    brightness: float = Form(default=0.0, ge=-60.0, le=60.0),
    clahe: bool = Form(default=False),
    denoise: bool = Form(default=False),
) -> AnalysisResponse:
    image_bytes = _read_upload(image)
    try:
        return analyze_image(
            image_bytes,
            gamma=gamma,
            contrast=contrast,
            brightness=brightness,
            clahe_enabled=clahe,
            denoise=denoise,
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DetectorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
