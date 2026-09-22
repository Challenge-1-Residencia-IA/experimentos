# experimentos

Repositório de experimentação: fine-tuning e avaliação de modelos (atualmente Qwen 2.5 7B via MLX/Ollama) para a tarefa de classificação de desinformação do Challenge 1. Separado do [`bot-core`](https://github.com/Challenge-1-Residencia-IA/bot-core) porque experimentação tem ciclo de iteração mais rápido e instável — quando um resultado amadurece (prompt validado, modelo fine-tunado estável), ele migra para lá.

## Estrutura

Inspirada no [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/), reduzida ao que o projeto precisa:

```
data/
├── raw/          # datasets baixados (Fake.br-Corpus, FakeTrue.Br) — não versionado
├── interim/      # dados intermediários (ex.: esquema comum unificado dos datasets)
└── processed/    # pool final de exemplos para few-shot/avaliação — não versionado
notebooks/        # EDA e exploração — numeração 01-, 02-, 03-...
models/           # adaptadores LoRA/QLoRA locais, antes do upload ao Hugging Face Hub — não versionado
reports/
└── figures/      # gráficos exportados da EDA
scripts/          # scripts de fine-tuning e avaliação
prompts/          # versões de prompt em teste (few-shot, boundary definitions etc.)
results/          # resultados de avaliação versionados (JSON/Markdown)
tests/            # testes das funções de avaliação/métricas
```

Só a estrutura das pastas de dados/modelos é versionada (arquivos `.gitkeep`) — o conteúdo real fica de fora do Git (ver `.gitignore`).

## Ambiente local

```bash
make create_environment   # ou: python3 -m venv .venv
source .venv/bin/activate
make requirements          # ou: pip install -r requirements.txt
cp .env.example .env
```

Pré-requisitos: Ollama instalado, modelo baixado (`ollama pull qwen2.5:7b`) — ver `Documentacao/docs/Sprint1/config-script.md` para o passo a passo já validado no time.

## Modelos e datasets

Pesos de modelo e datasets grandes **não são versionados aqui** (ver `.gitignore`). Modelos fine-tunados (adaptadores LoRA/QLoRA) são hospedados no [Hugging Face Hub](https://huggingface.co) e referenciados por `HF_MODEL_REPO` no `.env` — mesma convenção documentada no README do [`bot-core`](https://github.com/Challenge-1-Residencia-IA/bot-core#modelos-e-datasets). Localmente, ficam em `models/` antes do upload.

Datasets baixados ficam em `data/raw/` — baixe da fonte original (ver `Documentacao/docs/historias-de-usuario.md` e o dataset da Sprint 1) ou aponte os scripts para o caminho local.

## Resultados

Cada rodada de avaliação salva um resultado versionado em `results/` (ex.: `resultados_v1.json`), para permitir comparar prompts/modelos ao longo do tempo — ver metodologia em `Documentacao/docs/metricas-e-avaliacao.md`.
