from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from app.core.config import settings
from app.schemas.analysis import ApiDetection, AnalysisResponse
from app.services.detector import get_detector
from app.services.image_processing import (
    decode_image,
    encode_jpeg,
    preprocess_image,
)


def analyze_image(
    image_bytes: bytes,
    gamma: float,
    contrast: float,
    brightness: float,
    clahe_enabled: bool,
    denoise: bool,
) -> AnalysisResponse:
    started_at = perf_counter()
    image = decode_image(image_bytes)
    processed = preprocess_image(
        image,
        gamma=gamma,
        contrast=contrast,
        brightness=brightness,
        clahe_enabled=clahe_enabled,
        denoise=denoise,
    )
    detector = get_detector()
    detections = [
        ApiDetection(
            label=detection.class_name,
            confidence=detection.confidence,
            box=(
                detection.box.x1,
                detection.box.y1,
                detection.box.x2 - detection.box.x1,
                detection.box.y2 - detection.box.y1,
            ),
        )
        for detection in detector.detect(processed)
    ]
    return AnalysisResponse(
        analysis_id=str(uuid4()),
        model=settings.yolo_model_path,
        gamma=gamma,
        detections=detections,
        processed_image=encode_jpeg(processed),
        time_ms=round((perf_counter() - started_at) * 1000),
    )
