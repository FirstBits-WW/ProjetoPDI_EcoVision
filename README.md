# ProjetoPDI


Local run:

```bash
py -3.13 -m pip install -r requirements.txt
py -3.13 -m uvicorn app.main:app --reload --port 8007
```

The API documentation is available at `http://localhost:8007/docs`.
The HTML demo is served at `http://localhost:8007/` and submits to the API on
the same origin.
On Windows, use Python 3.13 for the full dependency set; installation under
Python 3.15 currently requires a C++ compiler to build PyYAML.

## Image analysis

`POST /api/v1/analyses` accepts multipart fields `image`, `gamma` (default
`1.0`, range `(0, 3]`), `contrast` (default `1.0`, range `0.5–2`), `brightness`
(default `0`, range `-60–60`), `clahe`, and `denoise`. It returns
`detections` (`label`, `confidence`, and `[x, y, width, height]` boxes),
`processed_image` as a JPEG data URL, and `time_ms`, matching the HTML demo.

The default `yolo11n.pt` weights are a provisional COCO model, not a model
trained specifically for litter. The returned labels are the model's actual
classes; the TACO evaluation script only scores categories whose names exactly
match a model class, and reports excluded categories. Configure alternative
weights and the confidence threshold with `YOLO_MODEL_PATH` and
`YOLO_CONFIDENCE`. The default model weights may be downloaded by Ultralytics
on first use. If that download is blocked by TLS/network policy, set
`YOLO_MODEL_PATH` to an existing local weights file; do not disable TLS
verification.

## TACO evaluation

Prepare o subconjunto e faça a divisão de treino/teste sem compartilhar
imagens entre os splits:

```powershell
py -3.13 -m scripts.preparar_base_taco `
  --anotacoes caminho\para\TACO\data\annotations.json `
  --imagens caminho\para\TACO\data `
  --saida dataset\subset_3classes `
  --copiar
```

The script maps TACO category names to lata, garrafa, and sacola, and writes
COCO-format train/test annotations. The image files themselves are not stored
in this repository. See [the TACO dataset notes and bibliography](docs/base_taco.md)
for the dataset source, license, references, and evaluation command.

Use the TACO annotation JSON and corresponding image directory to compare the
three processing conditions with IoU >= 0.50:

```bash
py -3.13 -m scripts.evaluate_taco --annotations dataset\subset_3classes\annotations_test.json --images-dir dataset\subset_3classes\test --output taco_metrics.csv
```

For a quick, deterministic smoke test, add `--limit 10`. The CSV contains TP,
FP, FN, Precision, Recall, and F1-Score for each condition. The gamma condition
uses the UI's gamma value `1.25`. Configure `YOLO_MODEL_PATH` with weights
trained on the three selected classes before evaluating.

Run the automated tests with:

```bash
py -3.13 -m pip install -r requirements-dev.txt
py -3.13 -m pytest -q
```
