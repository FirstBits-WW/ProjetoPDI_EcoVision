from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2

from app.schemas.analysis import BoundingBox, Detection
from app.services.detector import get_detector
from app.services.evaluation import Counts, match_detections
from app.services.image_processing import create_variants


def _normalize_label(label: str) -> str:
    return " ".join(label.casefold().split())


def _ground_truth_box(annotation: dict, category_name: str) -> Detection:
    x, y, width, height = (float(value) for value in annotation["bbox"])
    return Detection(
        class_name=category_name,
        confidence=1.0,
        box=BoundingBox(x1=x, y1=y, x2=x + width, y2=y + height),
    )


def evaluate_taco(
    annotation_file: Path,
    images_dir: Path,
    output_file: Path,
    limit: int | None = None,
) -> dict[str, Counts]:
    dataset = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = {
        int(category["id"]): str(category["name"])
        for category in dataset["categories"]
    }
    detector = get_detector()
    detector_labels = {
        _normalize_label(label) for label in detector.class_names.values()
    }
    supported_categories = {
        category_id: label
        for category_id, label in categories.items()
        if _normalize_label(label) in detector_labels
    }
    if not supported_categories:
        raise ValueError(
            "Nenhuma categoria da TACO coincide com as classes do detector "
            f"({', '.join(sorted(detector_labels))})."
        )

    unsupported_labels = sorted(
        label
        for label in categories.values()
        if _normalize_label(label) not in detector_labels
    )
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    counts_by_variant = {
        "original": [0, 0, 0],
        "clahe": [0, 0, 0],
        "gamma": [0, 0, 0],
    }
    images = dataset["images"]
    if limit is not None:
        images = images[:limit]

    for image_record in images:
        image_path = images_dir / image_record["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

        expected = [
            _ground_truth_box(
                annotation,
                supported_categories[int(annotation["category_id"])],
            )
            for annotation in annotations_by_image[int(image_record["id"])]
            if int(annotation["category_id"]) in supported_categories
        ]
        variants = create_variants(image, gamma=1.25)
        supported_labels = {
            _normalize_label(label) for label in supported_categories.values()
        }
        for name, variant in variants.items():
            predictions = [
                detection
                for detection in detector.detect(variant)
                if _normalize_label(detection.class_name)
                in supported_labels
            ]
            result = match_detections(predictions, expected, iou_threshold=0.5)
            aggregate = counts_by_variant[name]
            aggregate[0] += result.true_positives
            aggregate[1] += result.false_positives
            aggregate[2] += result.false_negatives

    totals = {
        name: Counts(*values) for name, values in counts_by_variant.items()
    }
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "condition",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1_score",
                "images_evaluated",
                "iou_threshold",
                "matched_categories",
            ],
        )
        writer.writeheader()
        matched_names = ";".join(sorted(supported_categories.values()))
        for name, counts in totals.items():
            writer.writerow(
                {
                    "condition": name,
                    "true_positives": counts.true_positives,
                    "false_positives": counts.false_positives,
                    "false_negatives": counts.false_negatives,
                    "precision": counts.precision,
                    "recall": counts.recall,
                    "f1_score": counts.f1_score,
                    "images_evaluated": len(images),
                    "iou_threshold": 0.5,
                    "matched_categories": matched_names,
                }
            )

    print(f"Resultados salvos em: {output_file}")
    print(f"Categorias avaliadas: {', '.join(sorted(supported_categories.values()))}")
    if unsupported_labels:
        print(
            "Categorias TACO ignoradas por incompatibilidade com o modelo: "
            + ", ".join(unsupported_labels)
        )
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara original, CLAHE e gama em imagens anotadas da TACO."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("taco_metrics.csv"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    evaluate_taco(args.annotations, args.images_dir, args.output, args.limit)


if __name__ == "__main__":
    main()
