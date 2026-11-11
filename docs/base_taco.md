# Base TACO e referências

## Levantamento da base

A TACO (Trash Annotations in Context) contém fotografias de resíduos em
ambientes reais, anotadas para detecção e segmentação. O artigo original
descreve aproximadamente 1.500 imagens, 4.784 anotações, 60 categorias e 28
supercategorias. Os números do contador comunitário do site podem divergir e
não devem ser confundidos com a versão oficial do artigo.

As anotações oficiais usam o formato COCO. O projeto seleciona três grupos de
categorias — lata, garrafa e sacola — e divide as imagens em treino e teste,
com semente fixa e sem reutilizar uma imagem nos dois conjuntos.

O repositório não contém as imagens da TACO. Obtenha as imagens e o arquivo
oficial `annotations.json` seguindo as instruções do repositório do dataset.
O dataset no Zenodo está sob CC BY 4.0; atribua os autores e verifique as
licenças das fotos individuais antes de redistribuí-las. As ferramentas do
repositório TACO usam licença MIT.

## Preparar o subconjunto

```powershell
py -3.13 -m scripts.preparar_base_taco `
  --anotacoes caminho\para\TACO\data\annotations.json `
  --imagens caminho\para\TACO\data `
  --saida dataset\subset_3classes `
  --copiar
```

O script grava `annotations_train.json` e `annotations_test.json`, com as
categorias `lata`, `garrafa` e `sacola`. As imagens copiadas mantêm seus
caminhos relativos, permitindo usar `subset_3classes\train` e
`subset_3classes\test` como diretórios de imagens. Se os arquivos de saída já
existirem, use `--sobrescrever` somente se quiser substituí-los.

Para calcular métricas no conjunto de teste, configure `YOLO_MODEL_PATH` com
pesos treinados para essas três classes e rode:

```powershell
py -3.13 -m scripts.evaluate_taco `
  --annotations dataset\subset_3classes\annotations_test.json `
  --images-dir dataset\subset_3classes\test `
  --output taco_metrics.csv
```

O script de preparação exige que as três classes apareçam em ambos os splits.
Se a amostra ficar pequena ou uma classe estiver ausente, ajuste a amostra ou a
semente em vez de interpretar métricas sem exemplos daquela classe.

## Referências bibliográficas

1. PROENÇA, Pedro F.; SIMÕES, Pedro. *TACO: Trash Annotations in Context for
   Litter Detection*. arXiv:2003.06975, 2020.
   <https://arxiv.org/abs/2003.06975>.
2. REDMON, Joseph et al. *You Only Look Once: Unified, Real-Time Object
   Detection*. IEEE CVPR, 2016. <https://arxiv.org/abs/1506.02640>.
3. REDMON, Joseph; FARHADI, Ali. *YOLOv3: An Incremental Improvement*.
   arXiv:1804.02767, 2018. <https://arxiv.org/abs/1804.02767>.
4. ZUIDERVELD, Karel. *Contrast Limited Adaptive Histogram Equalization*.
   In: HECKBERT, Paul S. (ed.). *Graphics Gems IV*. Academic Press, 1994,
   p. 474–485.
5. PIZER, Stephen M. et al. *Adaptive Histogram Equalization and Its
   Variations*. Computer Vision, Graphics, and Image Processing, v. 39, n. 3,
   p. 355–368, 1987.
6. GONZALEZ, Rafael C.; WOODS, Richard E. *Digital Image Processing*. 4. ed.
   Pearson, 2018.
7. BRADSKI, Gary. *The OpenCV Library*. Dr. Dobb's Journal of Software Tools,
   2000.
8. LIN, Tsung-Yi et al. *Microsoft COCO: Common Objects in Context*. ECCV,
   2014. <https://arxiv.org/abs/1405.0312>.
9. THUNG, Gary; YANG, Mindy. *Classification of Trash for Recyclability
   Status (TrashNet)*. Stanford University, CS229 Project Report, 2016.
10. EVERINGHAM, Mark et al. *The PASCAL Visual Object Classes (VOC)
    Challenge*. International Journal of Computer Vision, v. 88, n. 2,
    p. 303–338, 2010.

As referências de relevância social (IBGE e Prefeitura de Fortaleza) estão no
documento da Etapa 1 e devem ser mantidas junto à proposta acadêmica.
