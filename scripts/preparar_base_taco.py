from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


IDS_ALVO = {"lata": 1, "garrafa": 2, "sacola": 3}


def classe_alvo(categoria: dict[str, Any]) -> str | None:
    texto = " ".join(
        (str(categoria.get("name", "")), str(categoria.get("supercategory", "")))
    ).casefold()
    palavras = set(re.findall(r"[a-z]+", texto))

    if "bottle" in palavras and "cap" not in palavras:
        return "garrafa"
    if "can" in palavras:
        return "lata"
    if "bag" in palavras:
        return "sacola"
    return None


def _caminho_relativo(file_name: str) -> tuple[str, ...]:
    caminho = PurePosixPath(file_name.replace("\\", "/"))
    if (
        not caminho.parts
        or caminho.is_absolute()
        or PureWindowsPath(file_name).drive
        or PureWindowsPath(file_name).is_absolute()
        or any(parte in {".", ".."} for parte in caminho.parts)
    ):
        raise ValueError(f"Caminho de imagem inválido nas anotações: {file_name}")
    return caminho.parts


def _montar_coco(
    imagens: list[dict[str, Any]],
    anotacoes: list[dict[str, Any]],
    ids_imagens: set[int],
    metadados: dict[str, Any],
) -> dict[str, Any]:
    resultado = dict(metadados)
    resultado.update(
        {
            "images": [imagem for imagem in imagens if imagem["id"] in ids_imagens],
            "annotations": [
                anotacao
                for anotacao in anotacoes
                if anotacao["image_id"] in ids_imagens
            ],
            "categories": [
                {"id": class_id, "name": label, "supercategory": "residuo"}
                for label, class_id in IDS_ALVO.items()
            ],
        }
    )
    return resultado


def _contagem_por_classe(anotacoes: list[dict[str, Any]]) -> dict[str, int]:
    nomes_por_id = {class_id: name for name, class_id in IDS_ALVO.items()}
    contagens = {name: 0 for name in IDS_ALVO}
    for anotacao in anotacoes:
        contagens[nomes_por_id[anotacao["category_id"]]] += 1
    return contagens


def preparar_base(
    anotacoes_path: Path,
    saida: Path,
    imagens_dir: Path | None = None,
    fracao_treino: float = 0.8,
    seed: int = 42,
    copiar: bool = False,
    sobrescrever: bool = False,
) -> dict[str, dict[str, Any]]:
    if not 0 < fracao_treino < 1:
        raise ValueError("--treino deve estar entre 0 e 1, sem incluir os extremos.")
    if copiar and imagens_dir is None:
        raise ValueError("--copiar exige que --imagens seja informado.")

    coco = json.loads(anotacoes_path.read_text(encoding="utf-8"))
    required_keys = {"images", "annotations", "categories"}
    if not required_keys.issubset(coco):
        raise ValueError(
            "O JSON precisa conter as chaves COCO: images, annotations e categories."
        )

    categoria_para_classe = {
        categoria["id"]: classe
        for categoria in coco["categories"]
        if (classe := classe_alvo(categoria)) is not None
    }
    classes_presentes = set(categoria_para_classe.values())
    classes_ausentes = set(IDS_ALVO) - classes_presentes
    if classes_ausentes:
        raise ValueError(
            "Não foram encontradas categorias para: "
            + ", ".join(sorted(classes_ausentes))
        )

    imagem_por_id = {imagem["id"]: imagem for imagem in coco["images"]}
    anotacoes_filtradas = []
    ids_imagens_alvo: set[int] = set()
    for anotacao in coco["annotations"]:
        classe = categoria_para_classe.get(anotacao["category_id"])
        if classe is None:
            continue
        imagem_id = anotacao["image_id"]
        if imagem_id not in imagem_por_id:
            raise ValueError(
                f"Anotação referencia imagem inexistente: {imagem_id}"
            )
        anotacao_nova = dict(anotacao)
        anotacao_nova["category_id"] = IDS_ALVO[classe]
        anotacoes_filtradas.append(anotacao_nova)
        ids_imagens_alvo.add(imagem_id)

    imagens_filtradas = [
        imagem for imagem in coco["images"] if imagem["id"] in ids_imagens_alvo
    ]
    if len(imagens_filtradas) < 2:
        raise ValueError(
            "São necessárias pelo menos duas imagens com as classes selecionadas."
        )

    random.Random(seed).shuffle(imagens_filtradas)
    ponto_corte = min(
        max(int(len(imagens_filtradas) * fracao_treino), 1),
        len(imagens_filtradas) - 1,
    )
    ids_treino = {imagem["id"] for imagem in imagens_filtradas[:ponto_corte]}
    ids_teste = {imagem["id"] for imagem in imagens_filtradas[ponto_corte:]}
    if ids_treino & ids_teste:
        raise RuntimeError("Uma imagem foi atribuída ao treino e ao teste.")

    metadados = {key: coco[key] for key in ("info", "licenses") if key in coco}
    conjuntos = {
        "train": _montar_coco(
            imagens_filtradas, anotacoes_filtradas, ids_treino, metadados
        ),
        "test": _montar_coco(
            imagens_filtradas, anotacoes_filtradas, ids_teste, metadados
        ),
    }

    for nome, conjunto in conjuntos.items():
        contagens = _contagem_por_classe(conjunto["annotations"])
        classes_sem_exemplos = [
            classe for classe, total in contagens.items() if total == 0
        ]
        if classes_sem_exemplos:
            raise ValueError(
                f"A divisão {nome} ficou sem exemplos de: "
                + ", ".join(classes_sem_exemplos)
                + ". Tente outra semente ou aumente o conjunto."
            )

    arquivos_json = {
        split: saida / f"annotations_{split}.json" for split in conjuntos
    }
    if not sobrescrever:
        existentes = [str(path) for path in arquivos_json.values() if path.exists()]
        if existentes:
            raise FileExistsError(
                "Arquivos de saída já existem; use --sobrescrever para substituí-los: "
                + ", ".join(existentes)
            )

    copias: list[tuple[Path, Path]] = []
    if copiar:
        if imagens_dir is None:
            raise ValueError("--copiar exige que --imagens seja informado.")
        raiz_imagens = imagens_dir.resolve()
        for split, conjunto in conjuntos.items():
            for imagem in conjunto["images"]:
                partes = _caminho_relativo(str(imagem["file_name"]))
                origem = raiz_imagens.joinpath(*partes).resolve()
                try:
                    origem.relative_to(raiz_imagens)
                except ValueError as exc:
                    raise ValueError(
                        f"O caminho da imagem sai do diretório informado: {origem}"
                    ) from exc
                if not origem.is_file():
                    raise FileNotFoundError(f"Imagem não encontrada: {origem}")
                destino = saida / split
                for parte in partes:
                    destino /= parte
                if destino.exists() and not sobrescrever:
                    raise FileExistsError(
                        f"Imagem de saída já existe; use --sobrescrever: {destino}"
                    )
                copias.append((origem, destino))

    saida.mkdir(parents=True, exist_ok=True)
    for origem, destino in copias:
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origem, destino)

    for split, conjunto in conjuntos.items():
        with arquivos_json[split].open("w", encoding="utf-8") as arquivo:
            json.dump(conjunto, arquivo, ensure_ascii=False, indent=2)

    return conjuntos


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filtra as categorias TACO de lata, garrafa e sacola e cria "
            "divisões COCO de treino e teste."
        )
    )
    parser.add_argument("--anotacoes", type=Path, required=True)
    parser.add_argument("--imagens", type=Path)
    parser.add_argument("--saida", type=Path, default=Path("subset_3classes"))
    parser.add_argument("--treino", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copiar", action="store_true")
    parser.add_argument("--sobrescrever", action="store_true")
    args = parser.parse_args()

    try:
        conjuntos = preparar_base(
            anotacoes_path=args.anotacoes,
            saida=args.saida,
            imagens_dir=args.imagens,
            fracao_treino=args.treino,
            seed=args.seed,
            copiar=args.copiar,
            sobrescrever=args.sobrescrever,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    for split, conjunto in conjuntos.items():
        print(
            f"{split}: {len(conjunto['images'])} imagens, "
            f"{len(conjunto['annotations'])} anotações, "
            f"{_contagem_por_classe(conjunto['annotations'])}"
        )
    print(f"Arquivos salvos em: {args.saida}")


if __name__ == "__main__":
    main()
