from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str
    confidence: float
    box: BoundingBox


class ApiDetection(BaseModel):
    label: str
    confidence: float
    box: tuple[float, float, float, float]


class AnalysisResponse(BaseModel):
    analysis_id: str
    model: str
    gamma: float
    detections: list[ApiDetection]
    processed_image: str
    time_ms: int
