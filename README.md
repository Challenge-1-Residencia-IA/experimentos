# experimentos

Repositório de experimentação: fine-tuning e avaliação de modelos (atualmente Qwen 2.5 7B via MLX/Ollama) para a tarefa de classificação de desinformação do Challenge 1. Separado do [`bot-core`](https://github.com/Challenge-1-Residencia-IA/bot-core) porque experimentação tem ciclo de iteração mais rápido e instável — quando um resultado amadurece (prompt validado, modelo fine-tunado estável), ele migra para lá.

## Estrutura

```
scripts/    # scripts de fine-tuning e avaliação
prompts/    # versões de prompt em teste (few-shot, boundary definitions etc.)
results/    # resultados de avaliação versionados (JSON/Markdown, não os datasets em si)
```

## Ambiente local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

O rastreamento de experimentos é feito com MLflow. Por padrão, o treino usa
`mlflow.db` e salva artefatos em `mlartifacts/`, ambos locais e ignorados pelo
Git. Para abrir a interface após executar um treino:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Depois, acesse `http://127.0.0.1:5000`. Para usar outro servidor, defina
`MLFLOW_TRACKING_URI` (e opcionalmente `MLFLOW_EXPERIMENT_NAME`) no `.env` ou
passe `--tracking-uri` e `--experiment-name` ao script de treino.

Pré-requisitos: Ollama instalado, modelo baixado (`ollama pull qwen2.5:7b`) — ver `Documentacao/docs/Sprint1/config-script.md` para o passo a passo já validado no time.

## Modelos e datasets

Pesos de modelo e datasets grandes **não são versionados aqui** (ver `.gitignore`). Modelos fine-tunados (adaptadores LoRA/QLoRA) são hospedados no [Hugging Face Hub](https://huggingface.co) e referenciados por `HF_MODEL_REPO` no `.env` — mesma convenção documentada no README do [`bot-core`](https://github.com/Challenge-1-Residencia-IA/bot-core#modelos-e-datasets).

Datasets de teste usados na avaliação ficam fora do Git — baixe da fonte original (ver `Documentacao/docs/historias-de-usuario.md` e o dataset da Sprint 1) ou aponte os scripts para o caminho local.

## Resultados

Cada rodada de avaliação salva um resultado versionado em `results/` (ex.: `resultados_v1.json`), para permitir comparar prompts/modelos ao longo do tempo — ver metodologia em `Documentacao/docs/metricas-e-avaliacao.md`.

## Execução reproduzível

Todos os scripts aceitam caminhos explícitos e usam a raiz deste repositório como padrão.

```bash
# Rodada completa: split estratificado, treino e métricas no teste do Fake.br
python scripts/executar_pipeline.py

# Smoke test: 20 iterações e 10 exemplos do conjunto de teste
python scripts/executar_pipeline.py --iters 20 --limit-test 10
```

A pipeline usa `raw/preprocessed/pre-processed.csv` e separa treino, validação e
teste de forma estratificada (70/15/15, seed 42). O conjunto de teste nunca entra
no fine-tuning. Ela gera predições e um relatório com acurácia, precisão, recall
e F1 macro em `results/`, além de registrar uma run de treino e outra de avaliação
no experimento `fakebr-classificacao` do MLflow. Use `--input-csv` para apontar
outro local do CSV e `--test-size`, `--valid-size` ou `--seed` para mudar o split.

## Usar um modelo já treinado

Após o fine-tuning, classifique uma notícia sem treinar novamente:

```bash
python scripts/classificar_noticia.py \
  --adapters-dir models/adapters-fakebr-v1 \
  --texto "Texto da notícia que será classificada."
```

O comando imprime somente `FALSA`, `VERDADEIRA` ou `INDEFINIDO`. Para um texto
longo, use `--arquivo caminho/da/noticia.txt` em vez de `--texto`.

## Verificar uma notícia por link

O comando abaixo extrai o texto da página, busca evidências na web e envia notícia
+ evidências ao seu modelo fine-tunado, devolvendo o veredito e as fontes encontradas:

```bash
python scripts/verificar_link.py "https://exemplo.com/noticia" \
  --adapters-dir models/adapters-fakebr-v1 \
  --output results/verificacao_link.json
```

O modelo usado é o Qwen-base com seu adaptador LoRA. O adaptador atual foi treinado
somente com rótulos Fake.br, então ainda não aprendeu a justificar vereditos com
fontes: as evidências retornadas servem para revisão humana. Para transformar isso
em checagem fundamentada, o próximo fine-tuning precisa de exemplos com alegação,
evidências e veredito esperado.

Cada verificação também cria uma run no experimento `fakebr-verificacao` do
MLflow, com URL, modelo, veredito, quantidade de fontes recuperadas e o relatório
JSON como artefato. Métricas de acurácia e F1 exigem um gabarito e por isso são
calculadas apenas na avaliação do Fake.br.
