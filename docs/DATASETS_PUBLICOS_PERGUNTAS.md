# Datasets públicos para perguntas em português

Esta seleção prioriza resposta fundamentada em um texto de apoio. Os dados são
baixados para `raw/publicos-perguntas-v1/`, sem modificar os datasets que já
existem em `raw/`.

| Dataset | Uso no projeto | Licença | Observações |
| --- | --- | --- | --- |
| [FaQuAD](https://github.com/liafacom/faquad) | Perguntas e respostas sobre textos em português | CC BY 4.0 | Cerca de 900 perguntas; domínio de ensino superior; anotação humana |
| [Pirá](https://huggingface.co/datasets/paulopirozelli/pira) | QA com texto de apoio e casos sem resposta | CC BY 4.0 | 2.261 pares revisados; oceano, costa brasileira e clima |
| [RAG QA Full-Text PT-BR](https://huggingface.co/datasets/Madras1/rag-qa-fulltext-ptbr) | Amostra ampla de perguntas com trechos de evidência | ODC-By 1.0 | Quase 1 milhão de pares; anotações sintéticas; o padrão local limita a 5.000 |

O RAG QA PT-BR confirma que os trechos citados existem no contexto, mas isso
não confirma que o conteúdo da página original é verdadeiro. Por esse motivo,
o importador mantém sua origem e o dataset mesclado registra a limitação em
`metadata.json`.

Datasets muito grandes de perguntas sem texto de apoio não foram incluídos por
padrão. Eles ajudam conversação geral, porém não atendem tão bem ao requisito
de justificar a resposta com evidências e aumentam bastante o custo do treino.

## Fluxo sem treinamento

```bash
.venv/bin/python scripts/baixar_datasets_perguntas.py
.venv/bin/python scripts/preparar_dataset_perguntas_publicas.py
```

O primeiro comando cria apenas `raw/publicos-perguntas-v1/`. O segundo lê esse
diretório e `data/perguntas-v2/`, deduplica e grava um novo conjunto em
`data/perguntas-mescladas-v1/`. Nenhum dos dois importa `mlx_lm` ou inicia um
treino.

Para repetir um download interrompido, use `--resume`. Para limitar ou ampliar
a amostra sintética, use `--rag-limit`. O preparador recusa um diretório de
saída não vazio, evitando substituição acidental.

Quando você decidir treinar no futuro, use explicitamente:

```bash
.venv/bin/python scripts/treinar_visual.py \
  --data-dir data/perguntas-mescladas-v1 \
  --adapters-dir models/adapters-perguntas-v2
```

Esse comando é apenas uma referência e não é executado pela preparação.
