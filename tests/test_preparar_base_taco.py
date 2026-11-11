import json

import pytest

from scripts.preparar_base_taco import classe_alvo, preparar_base


def test_classe_alvo_matches_words_and_excludes_bottle_caps() -> None:
    assert classe_alvo({"name": "Drink can"}) == "lata"
    assert classe_alvo({"name": "Clear plastic bottle"}) == "garrafa"
    assert classe_alvo({"name": "Plastic bag & wrapper"}) == "sacola"
    assert classe_alvo({"name": "Bottle cap"}) is None
    assert classe_alvo({"name": "Candy wrapper"}) is None


def test_preparar_base_splits_without_leakage_and_preserves_image_paths(
    tmp_path,
) -> None:
    image_root = tmp_path / "TACO"
    annotations_path = tmp_path / "annotations.json"
    output_dir = tmp_path / "subset"
    image_records = []
    annotations = []
    categories = [
        {"id": 10, "name": "Drink can"},
        {"id": 20, "name": "Clear plastic bottle"},
        {"id": 30, "name": "Plastic bag & wrapper"},
        {"id": 40, "name": "Bottle cap"},
    ]

    for image_id in range(10):
        file_name = f"batch_1/image_{image_id}.jpg"
        image_records.append({"id": image_id, "file_name": file_name})
        image_path = image_root / "batch_1" / f"image_{image_id}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"synthetic image")
        for category_id in (10, 20, 30, 40):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                }
            )

    annotations_path.write_text(
        json.dumps(
            {
                "info": {"description": "fixture"},
                "images": image_records,
                "annotations": annotations,
                "categories": categories,
            }
        ),
        encoding="utf-8",
    )

    splits = preparar_base(
        annotations_path,
        output_dir,
        imagens_dir=image_root,
        copiar=True,
    )

    train_ids = {image["id"] for image in splits["train"]["images"]}
    test_ids = {image["id"] for image in splits["test"]["images"]}
    assert len(train_ids) == 8
    assert len(test_ids) == 2
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(range(10))
    for split_name, split in splits.items():
        assert split["info"] == {"description": "fixture"}
        assert {category["name"] for category in split["categories"]} == {
            "lata",
            "garrafa",
            "sacola",
        }
        assert {annotation["category_id"] for annotation in split["annotations"]} == {
            1,
            2,
            3,
        }
        for image in split["images"]:
            assert (
                output_dir / split_name / image["file_name"]
            ).is_file()
        saved = json.loads(
            (output_dir / f"annotations_{split_name}.json").read_text(
                encoding="utf-8"
            )
        )
        assert saved == split


def test_preparar_base_rejects_invalid_train_fraction(tmp_path) -> None:
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="--treino"):
        preparar_base(annotations_path, tmp_path / "subset", fracao_treino=1.0)
