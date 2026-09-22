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

Pré-requisitos: Ollama instalado, modelo baixado (`ollama pull qwen2.5:7b`) — ver `Documentacao/docs/Sprint1/config-script.md` para o passo a passo já validado no time.

## Modelos e datasets

Pesos de modelo e datasets grandes **não são versionados aqui** (ver `.gitignore`). Modelos fine-tunados (adaptadores LoRA/QLoRA) são hospedados no [Hugging Face Hub](https://huggingface.co) e referenciados por `HF_MODEL_REPO` no `.env` — mesma convenção documentada no README do [`bot-core`](https://github.com/Challenge-1-Residencia-IA/bot-core#modelos-e-datasets).

Datasets de teste usados na avaliação ficam fora do Git — baixe da fonte original (ver `Documentacao/docs/historias-de-usuario.md` e o dataset da Sprint 1) ou aponte os scripts para o caminho local.

## Resultados

Cada rodada de avaliação salva um resultado versionado em `results/` (ex.: `resultados_v1.json`), para permitir comparar prompts/modelos ao longo do tempo — ver metodologia em `Documentacao/docs/metricas-e-avaliacao.md`.

## Execução reproduzível

Todos os scripts aceitam caminhos explícitos e usam a raiz deste repositório como padrão.

```bash
# 1. Preparar dados de extração (espera as pastas fake/ e true/)
python scripts/preparar_dataset.py --dataset-path data/raw/full_texts

# 2. Treinar o adaptador LoRA
python scripts/treinar_visual.py --data-dir data/processed --adapters-dir models/adapters

# 3. Avaliar predições já produzidas e gerar o relatório canônico
python scripts/metricas_tecnicas.py \
  --predictions_file results/Sprint_1_Resultados_Modelo_Treinado.xlsx \
  --output_json results/relatorio_metricas_tecnicas.json

# 4. Gerar o dashboard a partir do mesmo relatório
python scripts/gerar_dashboard.py \
  --input-json results/relatorio_metricas_tecnicas.json
```

O arquivo de predições precisa conter as colunas listadas no avaliador, incluindo
`Divergencia_Esperada`. A coluna opcional `Resposta_Conclusiva` permite medir
corretamente respostas assertivas emitidas sem evidência.
