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


@dataclass(frozen=True)
class MatchResult:
    """Resultado da associação de um conjunto de predições a um gabarito.

    `flags[i]` indica se `predictions[i]` (já ordenadas por confiança
    decrescente) é verdadeiro positivo e `gt_indices[i]` qual caixa do
    gabarito ela encontrou (ou None).
    """

    predictions: list[Detection]
    flags: list[bool]
    matched_ground_truth: set[int]
    ground_truth_count: int
    gt_indices: tuple[int | None, ...] = ()

    def counts(self, min_confidence: float = 0.0) -> Counts:
        tp = fp = 0
        for prediction, is_tp in zip(self.predictions, self.flags):
            if prediction.confidence < min_confidence:
                continue
            if is_tp:
                tp += 1
            else:
                fp += 1
        return Counts(tp, fp, self.ground_truth_count - tp)

    def matched_at(self, min_confidence: float) -> set[int]:
        return {
            gt_index
            for prediction, gt_index in zip(self.predictions, self.gt_indices)
            if gt_index is not None and prediction.confidence >= min_confidence
        }


def match_with_flags(
    predictions: Iterable[Detection],
    ground_truth: Iterable[Detection],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Associação gulosa por confiança decrescente (critério PASCAL VOC/COCO).

    Como as predições são processadas da maior para a menor confiança, o
    resultado para qualquer limiar de confiança é o mesmo que se obteria
    descartando antes as predições abaixo dele.
    """
    predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    ground_truth = list(ground_truth)
    matched_ground_truth: set[int] = set()
    flags: list[bool] = []
    gt_indices: list[int | None] = []

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

        if best_index is not None:
            matched_ground_truth.add(best_index)
        flags.append(best_index is not None)
        gt_indices.append(best_index)

    return MatchResult(
        predictions=predictions,
        flags=flags,
        matched_ground_truth=matched_ground_truth,
        ground_truth_count=len(ground_truth),
        gt_indices=tuple(gt_indices),
    )


def match_detections(
    predictions: Iterable[Detection],
    ground_truth: Iterable[Detection],
    iou_threshold: float = 0.5,
) -> Counts:
    return match_with_flags(predictions, ground_truth, iou_threshold).counts()


def average_precision(
    scored_predictions: Iterable[tuple[float, bool]], ground_truth_count: int
) -> float:
    """AP com interpolação em todos os pontos (envelope da curva P×R)."""
    if ground_truth_count == 0:
        return 0.0
    ordered = sorted(scored_predictions, key=lambda item: item[0], reverse=True)
    recalls = [0.0]
    precisions = [1.0]
    tp = fp = 0
    for _, is_tp in ordered:
        if is_tp:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / ground_truth_count)
        precisions.append(tp / (tp + fp))

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    return sum(
        (recalls[index] - recalls[index - 1]) * precisions[index]
        for index in range(1, len(recalls))
    )
