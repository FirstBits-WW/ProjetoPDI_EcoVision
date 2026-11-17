import csv
import json

import cv2
import numpy as np

from app.schemas.analysis import BoundingBox, Detection
from scripts import evaluate_taco as taco_script


def test_taco_evaluation_writes_metrics_for_supported_categories(
    tmp_path, monkeypatch, capsys
) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    encoded, image_bytes = cv2.imencode(".png", image)
    assert encoded
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "sample.png").write_bytes(image_bytes.tobytes())

    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 1, "name": "Bottle"},
                    {"id": 2, "name": "Can"},
                ],
                "images": [{"id": 10, "file_name": "sample.png"}],
                "annotations": [
                    {"image_id": 10, "category_id": 1, "bbox": [1, 1, 5, 5]}
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeDetector:
        class_names = {0: "bottle"}

        def detect(self, image, confidence=None):
            return [
                Detection(
                    class_name="bottle",
                    confidence=0.9,
                    box=BoundingBox(x1=1, y1=1, x2=6, y2=6),
                ),
                # Abaixo do limiar de 0.25: entra no AP, não no P/R/F1.
                Detection(
                    class_name="bottle",
                    confidence=0.1,
                    box=BoundingBox(x1=0, y1=0, x2=2, y2=2),
                ),
            ]

    monkeypatch.setattr(taco_script, "get_detector", lambda: FakeDetector())
    output_path = tmp_path / "metrics.csv"

    details_dir = tmp_path / "detalhes"
    totals = taco_script.evaluate_taco(
        annotations_path,
        images_dir,
        output_path,
        confidence=0.25,
        details_dir=details_dir,
        examples=1,
    )

    assert set(totals) == {"original", "clahe", "gamma"}
    assert all(
        (
            counts.true_positives,
            counts.false_positives,
            counts.false_negatives,
        )
        == (1, 0, 0)
        for counts in totals.values()
    )
    with output_path.open(encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    # 3 condições × (todas + bottle)
    assert len(rows) == 6
    assert {row["class"] for row in rows} == {"todas", "bottle"}
    assert all(row["matched_categories"] == "Bottle" for row in rows)
    assert all(float(row["ap50"]) == 1.0 for row in rows)
    assert "Can" in capsys.readouterr().out

    resumo = json.loads((details_dir / "resumo.json").read_text(encoding="utf-8"))
    assert resumo["metricas"]["clahe"]["todas"]["f1_score"] == 1.0
    assert (details_dir / "por_imagem.csv").is_file()
    assert (details_dir / "por_iluminacao.csv").is_file()
    assert len(list((details_dir / "exemplos").glob("*_comparacao.jpg"))) == 1
