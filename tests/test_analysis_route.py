import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api.v1.routes import analyses
from app.main import app
from app.schemas.analysis import BoundingBox, Detection
from app.services import analysis as analysis_service
from app.services.detector import DetectorUnavailableError


def test_analysis_route_matches_frontend_contract_and_controls(monkeypatch) -> None:
    received = {}

    class FakeDetector:
        def detect(self, image):
            return [
                Detection(
                    class_name="bottle",
                    confidence=0.9,
                    box=BoundingBox(x1=1, y1=2, x2=6, y2=8),
                )
            ]

    def fake_preprocess(image, **options):
        received.update(options)
        return image

    monkeypatch.setattr(analysis_service, "preprocess_image", fake_preprocess)
    monkeypatch.setattr(analysis_service, "get_detector", lambda: FakeDetector())
    client = TestClient(app)
    success, encoded_image = cv2.imencode(
        ".png", np.full((12, 16, 3), 80, dtype=np.uint8)
    )
    assert success

    response = client.post(
        "/api/v1/analyses",
        files={"image": ("sample.png", encoded_image.tobytes(), "image/png")},
        data={
            "gamma": "0.7",
            "contrast": "1.5",
            "brightness": "10",
            "clahe": "true",
            "denoise": "true",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["gamma"] == 0.7
    assert received == {
        "gamma": 0.7,
        "contrast": 1.5,
        "brightness": 10.0,
        "clahe_enabled": True,
        "denoise": True,
    }
    assert body["detections"] == [
        {
            "label": "bottle",
            "confidence": 0.9,
            "box": [1.0, 2.0, 5.0, 6.0],
        }
    ]
    assert body["processed_image"].startswith("data:image/jpeg;base64,")
    assert body["time_ms"] >= 0


def test_frontend_is_served_from_root() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert 'API_URL: "/api/v1/analyses"' in response.text


def test_analysis_route_rejects_non_image_upload() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/analyses",
        files={"image": ("sample.jpg", b"not an image", "image/jpeg")},
    )

    assert response.status_code == 422


def test_analysis_route_reports_unavailable_detector(monkeypatch) -> None:
    def unavailable_detector():
        raise DetectorUnavailableError("Configure YOLO_MODEL_PATH.")

    monkeypatch.setattr(analysis_service, "get_detector", unavailable_detector)
    client = TestClient(app)
    success, encoded_image = cv2.imencode(
        ".png", np.zeros((8, 8, 3), dtype=np.uint8)
    )
    assert success

    response = client.post(
        "/api/v1/analyses",
        files={"image": ("sample.png", encoded_image.tobytes(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Configure YOLO_MODEL_PATH."
