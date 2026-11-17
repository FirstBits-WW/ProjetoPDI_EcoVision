import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.analysis import BoundingBox, Detection
from app.services import analysis as analysis_service


def _png() -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((20, 30, 3), 60, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_compare_runs_same_detector_on_three_conditions(monkeypatch) -> None:
    vistos = []

    class FakeDetector:
        def detect(self, image):
            vistos.append(float(image.mean()))
            return [
                Detection(class_name="lata", confidence=0.8,
                          box=BoundingBox(x1=1, y1=2, x2=5, y2=9))
            ]

    monkeypatch.setattr(analysis_service, "get_detector", lambda: FakeDetector())
    response = TestClient(app).post(
        "/api/v1/analyses/compare",
        files={"image": ("a.png", _png(), "image/png")},
        data={"gamma": "1.5"},
    )

    assert response.status_code == 200
    body = response.json()
    assert [c["condition"] for c in body["conditions"]] == ["original", "clahe", "gamma"]
    assert body["gamma"] == 1.5
    assert all(c["counts"] == {"lata": 1} for c in body["conditions"])
    assert body["conditions"][0]["detections"][0]["box"] == [1.0, 2.0, 4.0, 7.0]
    assert all(c["processed_image"].startswith("data:image/jpeg") for c in body["conditions"])
    # gama > 1 clareia a imagem; o detector recebeu versões diferentes
    assert vistos[2] > vistos[0]


def test_compare_rejects_invalid_image() -> None:
    response = TestClient(app).post(
        "/api/v1/analyses/compare",
        files={"image": ("a.png", b"not an image", "image/png")},
    )
    assert response.status_code == 422


def test_results_endpoint_serves_summary_and_examples(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "results_dir", str(tmp_path))
    client = TestClient(app)
    assert client.get("/api/v1/resultados").status_code == 404

    exemplos = tmp_path / "detalhes" / "exemplos"
    exemplos.mkdir(parents=True)
    (tmp_path / "detalhes" / "resumo.json").write_text(
        json.dumps({"metricas": {"original": {"todas": {"f1_score": 0.5}}}}),
        encoding="utf-8",
    )
    (exemplos / "01_x_comparacao.jpg").write_bytes(b"jpg")

    body = client.get("/api/v1/resultados").json()
    assert body["metricas"]["original"]["todas"]["f1_score"] == 0.5
    assert body["imagens_exemplo"] == ["/resultados/detalhes/exemplos/01_x_comparacao.jpg"]
