from app.schemas.analysis import BoundingBox, Detection
from app.services.evaluation import (
    intersection_over_union,
    match_detections,
)


def detection(
    label: str,
    confidence: float,
    box: tuple[float, float, float, float],
) -> Detection:
    return Detection(
        class_name=label,
        confidence=confidence,
        box=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
    )


def test_match_detections_uses_class_and_iou_threshold() -> None:
    truth = [detection("bottle", 1.0, (0, 0, 10, 10))]
    predictions = [
        detection("cup", 0.99, (0, 0, 10, 10)),
        detection("bottle", 0.8, (1, 1, 11, 11)),
        detection("bottle", 0.7, (0, 0, 10, 10)),
    ]

    result = match_detections(predictions, truth, iou_threshold=0.5)

    assert (result.true_positives, result.false_positives, result.false_negatives) == (
        1,
        2,
        0,
    )
    assert result.precision == 1 / 3
    assert result.recall == 1.0


def test_intersection_over_union_returns_zero_for_disjoint_boxes() -> None:
    assert intersection_over_union((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
