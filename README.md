<p align="center">
  <img src="docs/assets/logo-sophia.jpeg" alt="Sophia — Verificadora de fatos" width="260">
</p>

# Sophia — experimentos

Pesquisa, dados e avaliação da solução de IA da **Sophia — Verificadora de fatos**, uma tutora no
WhatsApp que ajuda a avaliar a confiabilidade de uma informação: aponta técnicas de manipulação,
mostra checagens e fontes e guia a verificação, **sem dar veredito de verdadeiro ou falso**.

Este repositório guarda os datasets, a análise exploratória, o golden set, a tutora RAG validada e os
relatórios de avaliação. O produto em produção (conector de WhatsApp, LangGraph, RAG com busca
híbrida, CI/CD) está no [`bot-core`](https://github.com/Challenge-1-Residencia-IA/bot-core).

## Entregáveis neste repositório

| Entregável | Onde está | Resumo |
|---|---|---|
| **Datasets** | `data/raw/*.dvc`, `dvc.yaml`, `dados/` | Fake.br-Corpus e FakeTrue.Br versionados com DVC (armazenamento no DagsHub). `make pipeline` reproduz as visões em `data/processed/`: base de checagens (1.075), avaliação da busca (1.791 pares), candidatos a golden set e a few-shot. Detalhes em [`reports/preparacao-dados.md`](reports/preparacao-dados.md). |
| **Análise exploratória (EDA)** | [`notebooks/01`](notebooks/01-eda-fakebr-corpus.ipynb), [`02`](notebooks/02-eda-faketrue-br.ipynb), [`03`](notebooks/03-eda-comparativo.ipynb) | Notebooks executados, com as saídas. Glossário das métricas em [`references/glossario-metricas.md`](references/glossario-metricas.md). |
| **Golden set v1** | [`reports/golden-set-v1.md`](reports/golden-set-v1.md), [`references/guia-anotacao-golden-set.md`](references/guia-anotacao-golden-set.md) | 65 mensagens reais de boato e 151 afirmações anotadas pelo time (natureza, tema, técnicas, fonte esperada), com concordância entre anotadores (kappa). `make golden-set` regera. |
| **Solução de IA** | `rag/`, `prompts/`, [`scripts/registrar_modelo_rag_mlflow.py`](scripts/registrar_modelo_rag_mlflow.py) | Tutora RAG (Qwen 2.5 7B via Ollama + embeddings E5 + busca na web) empacotada no formato MLflow pyfunc como `tutora-rag-v1`. Nenhum modelo foi treinado: o Qwen é usado como está, e o que o time construiu é o sistema em volta dele (prompts, configuração, guardrails e base de evidências). |
| **Guardrails** | `guardrails/` | Entrada (injeção, escopo com NeMo Guardrails, caracteres invisíveis), blindagem das fontes no prompt e saída (remove veredito, avisos em temas de risco). |
| **Avaliação e validação** | [`notebooks/04-validacao-modelo.ipynb`](notebooks/04-validacao-modelo.ipynb), `avaliacao/`, [`reports/avaliacoes/`](reports/avaliacoes/) | Tutora RAG empacotada avaliada contra o golden set, com registro no MLflow. |
| **Relatório técnico da solução de IA** | [Documentação (GitHub Pages)](https://challenge-1-residencia-ia.github.io/Documentacao/relatorio-tecnico-modelo/) | Dados, arquitetura, avaliação, limitações e próximos passos. |

Requisitos, histórias de usuário, organização do time e descrição do produto estão no site da
[Documentação](https://challenge-1-residencia-ia.github.io/Documentacao/).

## Principais achados

**Datasets (EDA)**

- **Fake.br:** o comprimento do texto sozinho acerta 94% (fakes com mediana de 157 palavras, verdadeiras com 918), e 93% das fakes vêm de um único site. O corpus mede estilo de redação mais que veracidade.
- **FakeTrue.Br:** os pares são boato × checagem do boato, não notícia falsa × verdadeira. Por isso o dataset serve de base de checagens e de golden set, não para treinar um classificador de veredito.
- **Comparativo:** um classificador simples, usado só como teste na EDA, acerta 88–90% dentro de cada dataset, mas só 57–62% treinando num e testando no outro. Esse resultado sustentou a decisão de não treinar um classificador binário.

**Validação da tutora RAG `tutora-rag-v1`** (golden set v1, 65 itens, busca na web + guardrails)

| Requisito | Resultado |
|---|---|
| GR-01: nenhuma resposta com veredito | 100% |
| RC-04: fontes com veículo, data e link | 80% |
| GR-06: aviso em tema de risco | 64% |
| IA-03: busca encontra a checagem esperada | 11% |
| Guardrails: mensagens legítimas recusadas / injeções barradas | 0 de 75 / 100% |
| Latência (mediana / p95) | 7,6 s / 21,7 s |

O ponto fraco (IA-03) levou à base própria de checagens com busca híbrida no `bot-core`. Lá, a
checagem esperada aparece entre as 5 primeiras em 89% do golden set, e esse número funciona como
portão de qualidade a cada deploy.

## Como rodar

Python 3.11 a 3.13 (recomendado 3.12, em `.python-version`), `venv` + `pip`.

```bash
make create_environment && source .venv/bin/activate
make requirements                 # dependências + stopwords do NLTK
pip install -r requirements-rag.txt && pip install -e .
dvc pull                          # datasets (credenciais do DagsHub em .dvc/config.local)
make pipeline                     # preparar dados + golden set (dvc repro)
make test
```

A tutora usa o [Ollama](https://ollama.com/) com `qwen2.5:7b`. O passo a passo completo (corpus,
menu, MLflow, DagsHub, solução de problemas) está em [`docs/GUIA_TUTORA_RAG.md`](docs/GUIA_TUTORA_RAG.md).

## Estrutura

```text
dados/        preparação dos datasets e do golden set (pipeline DVC)
notebooks/    EDA (01–03) e validação da tutora RAG (04)
rag/          tutora RAG e empacotamento MLflow
guardrails/   proteções de entrada, blindagem e saída
avaliacao/    avaliação contra o golden set, por requisito
prompts/      prompts versionados
reports/      relatórios gerados (preparação, golden set, avaliações)
references/   glossário de métricas e guia de anotação
scripts/      comandos de linha (indexar, perguntar, empacotar no MLflow, benchmarks)
docs/         guia técnico da tutora e registros de experimentos anteriores
```

Os documentos em `docs/` (benchmark AVeriTeC, conversão do dataset, DagsHub) e a pasta `results/`
registram experimentos das sprints anteriores e foram mantidos como histórico.

## Outros repositórios

| Repositório | Conteúdo |
|---|---|
| [`bot-core`](https://github.com/Challenge-1-Residencia-IA/bot-core) | Produto em produção: conector de WhatsApp, núcleo LangGraph, RAG com busca híbrida, guardrails, CI/CD com portão de qualidade |
| [`Documentacao`](https://github.com/Challenge-1-Residencia-IA/Documentacao) | Site do produto ([GitHub Pages](https://challenge-1-residencia-ia.github.io/Documentacao/)): requisitos, histórias, arquitetura, relatório técnico |
| [`bot-test`](https://github.com/Challenge-1-Residencia-IA/bot-test) | Protótipo do conector de WhatsApp Web (Playwright), incorporado ao `bot-core` |
