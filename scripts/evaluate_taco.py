"""Compara original, CLAHE e correção de gama no conjunto de teste da TACO.

Para cada imagem, as três versões passam pelo MESMO detector, com os mesmos
pesos e parâmetros. O detector roda uma única vez por versão com confiança
baixa (--conf-ap); a partir dessas predições calculam-se:

* Precision, Recall e F1 no limiar de operação (--conf, padrão YOLO_CONFIDENCE);
* AP@0.5 por classe e mAP@0.5 (curva precisão × revocação completa);
* os mesmos números por classe e por faixa de iluminação da imagem.

Com --detalhes, grava também a tabela por imagem, um resumo JSON para o
frontend e imagens anotadas (verde = gabarito, azul = acerto, vermelho =
falso positivo, laranja = resíduo não detectado).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.schemas.analysis import BoundingBox, Detection
from app.services.detector import get_detector
from app.services.evaluation import (
    Counts,
    MatchResult,
    average_precision,
    match_with_flags,
)
from app.services.image_processing import create_variants, load_annotated_image

CONDICOES = ("original", "clahe", "gamma")
TODAS = "todas"
IOU = 0.5


def _normalize_label(label: str) -> str:
    return " ".join(label.casefold().split())


def _ground_truth_box(annotation: dict, category_name: str) -> Detection:
    x, y, width, height = (float(value) for value in annotation["bbox"])
    return Detection(
        class_name=category_name,
        confidence=1.0,
        box=BoundingBox(x1=x, y1=y, x2=x + width, y2=y + height),
    )


def luminancia_media(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].mean())


@dataclass
class ResultadoImagem:
    file_name: str
    luminancia: float
    gabarito: list[Detection]
    por_condicao: dict[str, MatchResult] = field(default_factory=dict)
    faixa: str = ""


def _contagens(
    resultados: list[ResultadoImagem],
    condicao: str,
    limiar: float,
    classe: str | None = None,
) -> Counts:
    tp = fp = fn = 0
    for resultado in resultados:
        match = resultado.por_condicao[condicao]
        encontrados = match.matched_at(limiar)
        for prediction, is_tp in zip(match.predictions, match.flags):
            if prediction.confidence < limiar:
                continue
            if classe and _normalize_label(prediction.class_name) != classe:
                continue
            if is_tp:
                tp += 1
            else:
                fp += 1
        for index, expected in enumerate(resultado.gabarito):
            if classe and _normalize_label(expected.class_name) != classe:
                continue
            if index not in encontrados:
                fn += 1
    return Counts(tp, fp, fn)


def _ap(resultados: list[ResultadoImagem], condicao: str, classe: str) -> float:
    pontuadas: list[tuple[float, bool]] = []
    total_gabarito = 0
    for resultado in resultados:
        match = resultado.por_condicao[condicao]
        pontuadas.extend(
            (prediction.confidence, is_tp)
            for prediction, is_tp in zip(match.predictions, match.flags)
            if _normalize_label(prediction.class_name) == classe
        )
        total_gabarito += sum(
            _normalize_label(expected.class_name) == classe
            for expected in resultado.gabarito
        )
    return average_precision(pontuadas, total_gabarito)


def _linha(condicao: str, classe: str, counts: Counts, ap50: float | None) -> dict:
    return {
        "condition": condicao,
        "class": classe,
        "true_positives": counts.true_positives,
        "false_positives": counts.false_positives,
        "false_negatives": counts.false_negatives,
        "precision": round(counts.precision, 4),
        "recall": round(counts.recall, 4),
        "f1_score": round(counts.f1_score, 4),
        "ap50": "" if ap50 is None else round(ap50, 4),
    }


def _atribuir_faixas(resultados: list[ResultadoImagem]) -> tuple[float, float]:
    """Divide as imagens em três faixas de luminância de mesmo tamanho."""
    valores = sorted(r.luminancia for r in resultados)
    if not valores:
        return (0.0, 0.0)
    corte_1 = valores[len(valores) // 3]
    corte_2 = valores[(2 * len(valores)) // 3]
    for resultado in resultados:
        if resultado.luminancia < corte_1:
            resultado.faixa = "escura"
        elif resultado.luminancia < corte_2:
            resultado.faixa = "intermediaria"
        else:
            resultado.faixa = "clara"
    return (corte_1, corte_2)


# ---------------------------------------------------------------- desenho
CORES = {
    "gabarito": (80, 200, 80),
    "acerto": (230, 140, 30),
    "falso_positivo": (40, 40, 230),
    "nao_detectado": (0, 165, 255),
}


def _desenhar(
    image: np.ndarray,
    resultado: ResultadoImagem,
    condicao: str,
    limiar: float,
    altura: int = 720,
) -> np.ndarray:
    fator = altura / image.shape[0]
    canvas = cv2.resize(
        image,
        (round(image.shape[1] * fator), altura),
        interpolation=cv2.INTER_AREA if fator < 1 else cv2.INTER_LINEAR,
    )
    espessura = 2
    match = resultado.por_condicao[condicao]
    encontrados = match.matched_at(limiar)

    def caixa(det: Detection, cor, texto: str | None = None) -> None:
        p1 = (int(det.box.x1 * fator), int(det.box.y1 * fator))
        p2 = (int(det.box.x2 * fator), int(det.box.y2 * fator))
        cv2.rectangle(canvas, p1, p2, cor, espessura)
        if texto:
            escala = 0.5
            (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, espessura)
            y = max(p1[1], h + 4)
            cv2.rectangle(canvas, (p1[0], y - h - 4), (p1[0] + w + 4, y), cor, -1)
            cv2.putText(canvas, texto, (p1[0] + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        escala, (255, 255, 255), 1, cv2.LINE_AA)

    for index, expected in enumerate(resultado.gabarito):
        cor = CORES["gabarito"] if index in encontrados else CORES["nao_detectado"]
        caixa(expected, cor, None if index in encontrados else f"{expected.class_name}?")
    for prediction, is_tp in zip(match.predictions, match.flags):
        if prediction.confidence < limiar:
            continue
        cor = CORES["acerto"] if is_tp else CORES["falso_positivo"]
        caixa(prediction, cor, f"{prediction.class_name} {prediction.confidence:.2f}")
    return canvas


def _rotular(painel: np.ndarray, texto: str) -> np.ndarray:
    faixa = np.full((44, painel.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(faixa, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    return np.vstack([faixa, painel])


def _salvar_exemplos(
    resultados: list[ResultadoImagem],
    images_dir: Path,
    registros: dict[str, dict],
    destino: Path,
    quantidade: int,
    limiar: float,
    gamma: float,
) -> list[str]:
    def prioridade(resultado: ResultadoImagem) -> tuple:
        f1 = [
            resultado.por_condicao[c].counts(limiar).f1_score for c in CONDICOES
        ]
        erros = sum(
            resultado.por_condicao[c].counts(limiar).false_positives
            + resultado.por_condicao[c].counts(limiar).false_negatives
            for c in CONDICOES
        )
        return (max(f1) - min(f1), erros)

    escolhidos = sorted(resultados, key=prioridade, reverse=True)[:quantidade]
    destino.mkdir(parents=True, exist_ok=True)
    salvos = []
    for ordem, resultado in enumerate(escolhidos, start=1):
        registro = registros[resultado.file_name]
        imagem = load_annotated_image(
            images_dir / resultado.file_name,
            registro.get("width"),
            registro.get("height"),
        )
        variantes = create_variants(imagem, gamma=gamma)
        paineis = []
        base = f"{ordem:02d}_{Path(resultado.file_name.replace('/', '_')).stem}"
        for condicao in CONDICOES:
            desenho = _desenhar(variantes[condicao], resultado, condicao, limiar)
            cv2.imwrite(str(destino / f"{base}_{condicao}.jpg"), desenho)
            c = resultado.por_condicao[condicao].counts(limiar)
            paineis.append(
                _rotular(
                    desenho,
                    f"{condicao}  TP {c.true_positives}  FP {c.false_positives}"
                    f"  FN {c.false_negatives}",
                )
            )
        cv2.imwrite(str(destino / f"{base}_comparacao.jpg"), np.hstack(paineis))
        salvos.append(base)
    return salvos


# ---------------------------------------------------------------- principal
def evaluate_taco(
    annotation_file: Path,
    images_dir: Path,
    output_file: Path,
    limit: int | None = None,
    gamma: float = 1.25,
    confidence: float | None = None,
    confidence_ap: float = 0.001,
    details_dir: Path | None = None,
    examples: int = 12,
) -> dict[str, Counts]:
    limiar = settings.yolo_confidence if confidence is None else confidence
    if confidence_ap > limiar:
        raise ValueError("--conf-ap deve ser menor ou igual a --conf.")

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
            f"({', '.join(sorted(detector_labels))}). Configure YOLO_MODEL_PATH "
            "com os pesos treinados para lata/garrafa/sacola."
        )
    unsupported_labels = sorted(
        label
        for label in categories.values()
        if _normalize_label(label) not in detector_labels
    )
    classes = sorted(
        {_normalize_label(label) for label in supported_categories.values()}
    )

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    images = dataset["images"]
    if limit is not None:
        images = images[:limit]
    registros = {str(record["file_name"]): record for record in images}

    resultados: list[ResultadoImagem] = []
    for posicao, image_record in enumerate(images, start=1):
        image = load_annotated_image(
            images_dir / image_record["file_name"],
            image_record.get("width"),
            image_record.get("height"),
        )
        resultado = ResultadoImagem(
            file_name=str(image_record["file_name"]),
            luminancia=luminancia_media(image),
            gabarito=[
                _ground_truth_box(
                    annotation,
                    supported_categories[int(annotation["category_id"])],
                )
                for annotation in annotations_by_image[int(image_record["id"])]
                if int(annotation["category_id"]) in supported_categories
            ],
        )
        for name, variant in create_variants(image, gamma=gamma).items():
            predictions = [
                detection
                for detection in detector.detect(variant, confidence=confidence_ap)
                if _normalize_label(detection.class_name) in classes
            ]
            resultado.por_condicao[name] = match_with_flags(
                predictions, resultado.gabarito, iou_threshold=IOU
            )
        resultados.append(resultado)
        if posicao % 25 == 0:
            print(f"{posicao}/{len(images)} imagens avaliadas")

    # ---- tabela principal: condição × classe
    linhas = []
    totals: dict[str, Counts] = {}
    resumo: dict[str, dict] = {}
    for condicao in CONDICOES:
        aps = {classe: _ap(resultados, condicao, classe) for classe in classes}
        geral = _contagens(resultados, condicao, limiar)
        totals[condicao] = geral
        linhas.append(_linha(condicao, TODAS, geral, sum(aps.values()) / len(aps)))
        resumo[condicao] = {TODAS: linhas[-1]}
        for classe in classes:
            linhas.append(
                _linha(
                    condicao,
                    classe,
                    _contagens(resultados, condicao, limiar, classe),
                    aps[classe],
                )
            )
            resumo[condicao][classe] = linhas[-1]

    comum = {
        "images_evaluated": len(images),
        "iou_threshold": IOU,
        "confidence_threshold": limiar,
        "gamma": gamma,
        "matched_categories": ";".join(sorted(supported_categories.values())),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[*linhas[0], *comum])
        writer.writeheader()
        for linha in linhas:
            writer.writerow({**linha, **comum})

    print(f"Resultados salvos em: {output_file}")
    print(f"Categorias avaliadas: {', '.join(sorted(supported_categories.values()))}")
    if unsupported_labels:
        print(
            "Categorias TACO ignoradas por incompatibilidade com o modelo: "
            + ", ".join(unsupported_labels)
        )
    print(f"\n{'condição':<10}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>8}")
    for condicao in CONDICOES:
        r = resumo[condicao][TODAS]
        print(
            f"{condicao:<10}{r['precision']:>8.3f}{r['recall']:>8.3f}"
            f"{r['f1_score']:>8.3f}{r['ap50']:>8.3f}"
        )

    if details_dir is not None:
        _gravar_detalhes(
            details_dir, resultados, resumo, comum, limiar, images_dir,
            registros, examples, gamma, classes,
        )
    return totals


def _gravar_detalhes(
    details_dir: Path,
    resultados: list[ResultadoImagem],
    resumo: dict,
    comum: dict,
    limiar: float,
    images_dir: Path,
    registros: dict[str, dict],
    examples: int,
    gamma: float,
    classes: list[str],
) -> None:
    details_dir.mkdir(parents=True, exist_ok=True)

    with (details_dir / "por_imagem.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["file_name", "luminancia", "faixa", "objetos_gabarito"]
            + [f"{c}_{m}" for c in CONDICOES for m in ("tp", "fp", "fn", "f1")]
        )
        cortes = _atribuir_faixas(resultados)
        for r in resultados:
            valores = []
            for condicao in CONDICOES:
                c = r.por_condicao[condicao].counts(limiar)
                valores += [
                    c.true_positives, c.false_positives, c.false_negatives,
                    round(c.f1_score, 4),
                ]
            writer.writerow(
                [r.file_name, round(r.luminancia, 1), r.faixa, len(r.gabarito)]
                + valores
            )

    faixas: dict[str, dict] = {}
    with (details_dir / "por_iluminacao.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["faixa", "imagens", "condition", "true_positives", "false_positives",
             "false_negatives", "precision", "recall", "f1_score"]
        )
        for faixa in ("escura", "intermediaria", "clara"):
            grupo = [r for r in resultados if r.faixa == faixa]
            faixas[faixa] = {"imagens": len(grupo)}
            for condicao in CONDICOES:
                c = _contagens(grupo, condicao, limiar)
                faixas[faixa][condicao] = _linha(condicao, TODAS, c, None)
                writer.writerow(
                    [faixa, len(grupo), condicao, c.true_positives,
                     c.false_positives, c.false_negatives, round(c.precision, 4),
                     round(c.recall, 4), round(c.f1_score, 4)]
                )

    exemplos = _salvar_exemplos(
        resultados, images_dir, registros, details_dir / "exemplos",
        examples, limiar, gamma,
    )
    (details_dir / "resumo.json").write_text(
        json.dumps(
            {
                **comum,
                "classes": classes,
                "objetos_gabarito": sum(len(r.gabarito) for r in resultados),
                "metricas": resumo,
                "iluminacao": {
                    "cortes_luminancia_L": [round(v, 1) for v in cortes],
                    "faixas": faixas,
                },
                "exemplos": exemplos,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Detalhes (por imagem, iluminação, exemplos, resumo.json): {details_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara original, CLAHE e gama em imagens anotadas da TACO."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("taco_metrics.csv"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gamma", type=float, default=1.25)
    parser.add_argument(
        "--conf",
        type=float,
        help="Limiar de confiança para P/R/F1 (padrão: YOLO_CONFIDENCE).",
    )
    parser.add_argument(
        "--conf-ap",
        type=float,
        default=0.001,
        help="Confiança mínima usada para montar a curva do AP.",
    )
    parser.add_argument(
        "--detalhes",
        type=Path,
        help="Pasta para tabela por imagem, iluminação, exemplos e resumo.json.",
    )
    parser.add_argument("--exemplos", type=int, default=12)
    args = parser.parse_args()
    evaluate_taco(
        args.annotations,
        args.images_dir,
        args.output,
        limit=args.limit,
        gamma=args.gamma,
        confidence=args.conf,
        confidence_ap=args.conf_ap,
        details_dir=args.detalhes,
        examples=args.exemplos,
    )


if __name__ == "__main__":
    main()
