"""Resultados do experimento na TACO (gerados por scripts/evaluate_taco.py)."""

import json

from fastapi import APIRouter, HTTPException

from app.core.config import settings

router = APIRouter()


@router.get("/resultados")
def get_results() -> dict:
    """Devolve resultados/detalhes/resumo.json e a lista de imagens de exemplo.

    Basta copiar a pasta `resultados` gerada no Colab para a raiz do projeto.
    """
    base = settings.results_dir_path
    resumo = base / "detalhes" / "resumo.json"
    if not resumo.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Arquivo não encontrado: {resumo}. Copie a pasta 'resultados' do Colab.",
        )
    dados = json.loads(resumo.read_text(encoding="utf-8"))
    exemplos = sorted((base / "detalhes" / "exemplos").glob("*_comparacao.jpg"))
    dados["imagens_exemplo"] = [
        f"/resultados/detalhes/exemplos/{caminho.name}" for caminho in exemplos
    ]
    return dados
