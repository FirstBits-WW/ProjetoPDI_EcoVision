from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.schemas.analysis import BoundingBox, Detection


class DetectorUnavailableError(RuntimeError):
    pass


class YoloDetector:
    def __init__(self, model_path: str, confidence: float) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailableError(
                "A dependência 'ultralytics' não está instalada."
            ) from exc

        if model_path != "yolo11n.pt" and not Path(model_path).is_file():
            raise DetectorUnavailableError(
                f"Os pesos YOLO configurados não foram encontrados: {model_path}"
            )

        self.model_path = model_path
        self.confidence = confidence
        try:
            self.model: Any = YOLO(model_path)
        except ConnectionError as exc:
            raise DetectorUnavailableError(
                "Não foi possível baixar os pesos YOLO. Configure "
                "YOLO_MODEL_PATH com um arquivo de pesos local."
            ) from exc
        self.class_names = self._normalize_names(self.model.names)

    @staticmethod
    def _normalize_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        return {class_id: str(name) for class_id, name in enumerate(names)}

    def detect(self, image: Any) -> list[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        return [
            Detection(
                class_name=self.class_names[int(class_id)],
                confidence=float(score),
                box=BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                ),
            )
            for box, score, class_id in zip(boxes, scores, class_ids)
        ]


@lru_cache(maxsize=1)
def get_detector() -> YoloDetector:
    return YoloDetector(settings.yolo_model_path, settings.yolo_confidence)
