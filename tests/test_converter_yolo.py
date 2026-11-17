import json

import cv2
import numpy as np

from scripts.converter_yolo import converter


def test_converter_writes_normalized_labels_and_data_yaml(tmp_path) -> None:
    base = tmp_path / "subset"
    categorias = [
        {"id": 1, "name": "lata"},
        {"id": 2, "name": "garrafa"},
        {"id": 3, "name": "sacola"},
    ]
    for indice, split in enumerate(("train", "val", "test")):
        pasta = base / split / "batch_1"
        pasta.mkdir(parents=True)
        cv2.imwrite(str(pasta / f"img{indice}.png"), np.zeros((100, 200, 3), np.uint8))
        (base / f"annotations_{split}.json").write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": indice,
                            "file_name": f"batch_1/img{indice}.png",
                            "width": 200,
                            "height": 100,
                        }
                    ],
                    "annotations": [
                        {"id": 1, "image_id": indice, "category_id": 2,
                         "bbox": [50, 25, 100, 50]},
                        {"id": 2, "image_id": indice, "category_id": 3,
                         "bbox": [190, 90, 0.2, 0.2]},
                    ],
                    "categories": categorias,
                }
            ),
            encoding="utf-8",
        )

    saida = tmp_path / "yolo"
    resumos = converter(base, saida, max_lado=100)

    assert resumos["train"] == {"imagens": 1, "caixas": 1, "caixas_descartadas": 1}
    label = (saida / "labels" / "train" / "batch_1_img0.txt").read_text()
    assert label.strip() == "1 0.500000 0.500000 0.500000 0.500000"
    imagem = cv2.imread(str(saida / "images" / "train" / "batch_1_img0.jpg"))
    assert imagem.shape[:2] == (50, 100)
    yaml = (saida / "data.yaml").read_text()
    assert "val: images/val" in yaml
    assert "  0: lata\n  1: garrafa\n  2: sacola" in yaml
