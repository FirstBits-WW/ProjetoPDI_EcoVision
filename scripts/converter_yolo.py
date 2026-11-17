"""Converte as divisões COCO geradas por preparar_base_taco para o formato YOLO.

Estrutura gerada (compatível com Ultralytics):

    saida/
      data.yaml
      images/{train,val,test}/<lote>_<arquivo>.jpg
      labels/{train,val,test}/<lote>_<arquivo>.txt

As imagens são redimensionadas para no máximo --max-lado pixels no lado maior
(as fotos originais da TACO chegam a ~4000 px, o que só deixa o treino lento) e
gravadas já na orientação das anotações, sem depender de EXIF.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from app.services.image_processing import load_annotated_image
from scripts.preparar_base_taco import SPLITS


def _nome_plano(file_name: str) -> str:
    return file_name.replace("\\", "/").strip("/").replace("/", "_")


def _linha_yolo(
    bbox: list[float], width: int, height: int, class_index: int
) -> str | None:
    x, y, w, h = (float(value) for value in bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    return (
        f"{class_index} {cx:.6f} {cy:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def converter_split(
    coco: dict[str, Any],
    imagens_dir: Path,
    saida: Path,
    split: str,
    class_index_por_id: dict[int, int],
    max_lado: int,
) -> dict[str, int]:
    anotacoes_por_imagem: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for anotacao in coco["annotations"]:
        anotacoes_por_imagem[int(anotacao["image_id"])].append(anotacao)

    pasta_imagens = saida / "images" / split
    pasta_labels = saida / "labels" / split
    pasta_imagens.mkdir(parents=True, exist_ok=True)
    pasta_labels.mkdir(parents=True, exist_ok=True)
    resumo = {"imagens": 0, "caixas": 0, "caixas_descartadas": 0}

    for registro in coco["images"]:
        origem = imagens_dir / registro["file_name"]
        imagem = load_annotated_image(
            origem, registro.get("width"), registro.get("height")
        )
        altura, largura = imagem.shape[:2]
        linhas = []
        for anotacao in anotacoes_por_imagem[int(registro["id"])]:
            linha = _linha_yolo(
                anotacao["bbox"],
                largura,
                altura,
                class_index_por_id[int(anotacao["category_id"])],
            )
            if linha is None:
                resumo["caixas_descartadas"] += 1
            else:
                linhas.append(linha)

        escala = max_lado / max(altura, largura)
        if escala < 1:
            imagem = cv2.resize(
                imagem,
                (round(largura * escala), round(altura * escala)),
                interpolation=cv2.INTER_AREA,
            )
        nome = Path(_nome_plano(registro["file_name"])).with_suffix(".jpg")
        if not cv2.imwrite(
            str(pasta_imagens / nome), imagem, [cv2.IMWRITE_JPEG_QUALITY, 95]
        ):
            raise RuntimeError(f"Falha ao gravar {pasta_imagens / nome}")
        (pasta_labels / nome.with_suffix(".txt")).write_text(
            "\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8"
        )
        resumo["imagens"] += 1
        resumo["caixas"] += len(linhas)
    return resumo


def converter(base: Path, saida: Path, max_lado: int = 1600) -> dict[str, dict]:
    """`base` é a pasta de saída de preparar_base_taco (com --copiar)."""
    resumos: dict[str, dict] = {}
    nomes: list[str] | None = None
    for split in SPLITS:
        coco = json.loads(
            (base / f"annotations_{split}.json").read_text(encoding="utf-8")
        )
        categorias = sorted(coco["categories"], key=lambda c: int(c["id"]))
        nomes_split = [str(c["name"]) for c in categorias]
        if nomes is None:
            nomes = nomes_split
        elif nomes != nomes_split:
            raise ValueError("As divisões têm categorias diferentes.")
        class_index_por_id = {
            int(c["id"]): index for index, c in enumerate(categorias)
        }
        resumos[split] = converter_split(
            coco, base / split, saida, split, class_index_por_id, max_lado
        )

    assert nomes is not None
    yaml_linhas = [
        f"path: {saida.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *(f"  {index}: {nome}" for index, nome in enumerate(nomes)),
    ]
    (saida / "data.yaml").write_text("\n".join(yaml_linhas) + "\n", encoding="utf-8")
    return resumos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Pasta gerada por preparar_base_taco --copiar.",
    )
    parser.add_argument("--saida", type=Path, default=Path("dataset/yolo_3classes"))
    parser.add_argument("--max-lado", type=int, default=1600)
    args = parser.parse_args()
    for split, resumo in converter(args.base, args.saida, args.max_lado).items():
        print(f"{split}: {resumo}")
    print(f"data.yaml: {args.saida / 'data.yaml'}")


if __name__ == "__main__":
    main()
