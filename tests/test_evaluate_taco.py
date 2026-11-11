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

        def detect(self, image):
            return [
                Detection(
                    class_name="bottle",
                    confidence=0.9,
                    box=BoundingBox(x1=1, y1=1, x2=6, y2=6),
                )
            ]

    monkeypatch.setattr(taco_script, "get_detector", lambda: FakeDetector())
    output_path = tmp_path / "metrics.csv"

    totals = taco_script.evaluate_taco(annotations_path, images_dir, output_path)

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
    assert len(rows) == 3
    assert all(row["matched_categories"] == "Bottle" for row in rows)
    assert "Can" in capsys.readouterr().out
