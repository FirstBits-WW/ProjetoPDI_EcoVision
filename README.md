# EcoVision — PDI na detecção de resíduos (base TACO)

Projeto da disciplina de Processamento Digital de Imagens (Prof.ª Lyndainês
Santos, UNIFOR). Pergunta investigada: aplicar **CLAHE** ou **correção de
gama** antes do detector melhora Precision, Recall e F1 em comparação com a
imagem original? O mesmo detector YOLO11, com os mesmos pesos e limiar, é
avaliado nas três condições.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `app/` | API FastAPI: pré-processamento (OpenCV), detector YOLO, rotas |
| `frontend/index.html` | Interface web servida pela própria API |
| `scripts/` | Preparação da TACO, conversão para YOLO e avaliação das três condições |
| `notebooks/treino_colab.ipynb` | Treino e avaliação completos no Google Colab (GPU) |
| `tests/` | Testes automatizados (`pytest`) |
| `docs/` | Notas sobre a base TACO |

Técnicas de PDI (em `app/services/image_processing.py`):

- **CLAHE** no canal L do espaço LAB (`clipLimit=2.0`, grade 8×8), para realçar o contraste local sem distorcer as cores;
- **correção de gama** por tabela de consulta (LUT), `saída = 255 · (entrada/255)^(1/γ)`, com γ = 1,25 (clareia);
- **imagem original** como linha de base.

## Como executar (Windows)

Requer Python 3.13. O pacote já vem com o modelo treinado (`best.pt`) e os
resultados da avaliação (`resultados/`). Abra um terminal (PowerShell) na pasta
do projeto e rode:

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m uvicorn app.main:app --port 8000
```

- Interface: <http://localhost:8000/>
- Documentação da API: <http://localhost:8000/docs>

Para parar o servidor: `Ctrl+C` no terminal. A primeira análise demora alguns
segundos (carregamento do modelo); as seguintes são rápidas.

Para usar outros pesos ou outra pasta de resultados, copie `.env.example` para
`.env` e ajuste `YOLO_MODEL_PATH` e `RESULTS_DIR`.

## API

| Método e rota | O que faz |
|---|---|
| `POST /api/v1/analyses/compare` | Campo `image` (+ `gamma`, padrão 1,25). Roda o mesmo detector na imagem original, com CLAHE e com gama e devolve, para cada condição, as detecções, a contagem por classe e a imagem processada. |
| `POST /api/v1/analyses` | Ajustes livres numa única imagem: `gamma`, `contrast`, `brightness`, `clahe`, `denoise`. |
| `GET /api/v1/resultados` | Conteúdo de `resultados/detalhes/resumo.json` e a lista de imagens de exemplo. |
| `GET /api/v1/health` | Verificação de funcionamento. |

As caixas usam `[x, y, largura, altura]` em pixels da imagem enviada.

## Treino e avaliação na TACO

Os resultados apresentados vêm da 1ª rodada: YOLO11s, imagens de 1024 px,
355 imagens de treino, 50 de validação e 103 de teste.

O caminho recomendado é o notebook **`notebooks/treino_colab.ipynb`** (Google
Colab com GPU). Ele baixa a TACO, prepara as divisões, treina o YOLO11 e roda a
comparação original × CLAHE × gama, salvando os resultados no Google Drive.
Para usar: arraste o zip deste projeto para o painel "Arquivos" do Colab (ou
deixe-o em `MyDrive/EcoVision/`, ou preencha `REPO_URL`) e execute as células
em ordem; se o zip não for encontrado, a célula abre um botão de upload.

Os mesmos passos podem ser rodados localmente:

```powershell
# 1. Seleciona lata/garrafa/sacola e divide 70% treino / 10% validação / 20% teste
py -3.13 -m scripts.preparar_base_taco `
  --anotacoes caminho\para\TACO\data\annotations.json `
  --imagens caminho\para\TACO\data `
  --saida dataset\subset_3classes --copiar

# 2. Converte para o formato YOLO (gera dataset\yolo_3classes\data.yaml)
py -3.13 -m scripts.converter_yolo --base dataset\subset_3classes --saida dataset\yolo_3classes

# 3. Treina (somente com imagens originais)
yolo detect train data=dataset\yolo_3classes\data.yaml model=yolo11s.pt imgsz=1024 epochs=100 patience=25 seed=42

# 4. Compara as três condições no TESTE com os pesos treinados
$env:YOLO_MODEL_PATH="runs\detect\train\weights\best.pt"
py -3.13 -m scripts.evaluate_taco `
  --annotations dataset\subset_3classes\annotations_test.json `
  --images-dir dataset\subset_3classes\test `
  --output resultados\taco_metrics.csv --detalhes resultados\detalhes
```

O mapeamento de classes usa os nomes oficiais das categorias TACO (dicionário
`CATEGORIAS_TACO` em `scripts/preparar_base_taco.py`); a supercategoria não é
usada, para que filme plástico e embalagens não sejam contados como sacola.

A avaliação roda o detector uma vez por imagem e condição com confiança baixa e
calcula Precision/Recall/F1 no limiar `YOLO_CONFIDENCE` (padrão 0,25) e AP@0.5
pela curva completa, com IoU ≥ 0,50. Com `--detalhes` gera também
`por_imagem.csv`, `por_iluminacao.csv` (imagens divididas em terços pela
luminância média), `resumo.json` e imagens anotadas em `exemplos/`
(verde = gabarito, azul = acerto, vermelho = falso positivo, laranja = não detectado).
Use `--limit 10` para um teste rápido.

## Testes

```powershell
py -3.13 -m pip install -r requirements-dev.txt
py -3.13 -m pytest -q
```
