from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.schemas.analysis import Detection


@dataclass(frozen=True)
class Counts:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1_score(self) -> float:
        denominator = self.precision + self.recall
        return (
            2 * self.precision * self.recall / denominator
            if denominator
            else 0.0
        )


def intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _box(detection: Detection) -> tuple[float, float, float, float]:
    return (
        detection.box.x1,
        detection.box.y1,
        detection.box.x2,
        detection.box.y2,
    )


def match_detections(
    predictions: Iterable[Detection],
    ground_truth: Iterable[Detection],
    iou_threshold: float = 0.5,
) -> Counts:
    predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    ground_truth = list(ground_truth)
    matched_ground_truth: set[int] = set()
    true_positives = 0
    false_positives = 0

    for prediction in predictions:
        best_index = None
        best_iou = iou_threshold
        for index, expected in enumerate(ground_truth):
            if index in matched_ground_truth:
                continue
            if prediction.class_name.casefold() != expected.class_name.casefold():
                continue
            overlap = intersection_over_union(_box(prediction), _box(expected))
            if overlap >= best_iou:
                best_index = index
                best_iou = overlap

        if best_index is None:
            false_positives += 1
        else:
            matched_ground_truth.add(best_index)
            true_positives += 1

    return Counts(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=len(ground_truth) - true_positives,
    )
