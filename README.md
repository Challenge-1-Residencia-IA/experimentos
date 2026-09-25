# Sift AI — perguntas e verificação com evidências

Este repositório reúne experimentos locais para:

- responder perguntas em português com fontes e links;
- verificar alegações e notícias a partir de evidências;
- classificar textos do Fake.br;
- preparar datasets, ajustar adapters LoRA e avaliar os resultados;
- registrar treinos, respostas, justificativas e artefatos no MLflow.

O projeto usa Qwen 2.5 com MLX-LM em Macs Apple Silicon. Ele é experimental:
uma resposta citada ainda precisa de revisão humana, principalmente em decisões
importantes.

## Visão geral

```text
pergunta ou URL
    ↓
busca e filtragem de fontes
    ↓
trechos + links apresentados ao modelo
    ↓
resposta + justificativa + citações [1] [2]
    ↓
validação determinística
    ↓
terminal/JSON + MLflow
```

O modelo não abre a internet sozinho. Os scripts recuperam as fontes e enviam
ao modelo apenas os trechos selecionados. Uma justificativa é uma explicação
observável baseada nessas evidências; não é a cadeia de pensamento interna do
modelo.

## Instalação

Requisitos:

- macOS em Apple Silicon para treino e inferência com MLX;
- Python 3.10 ou superior;
- internet para baixar o modelo, datasets públicos e pesquisar fontes.

```bash
git clone https://github.com/Challenge-1-Residencia-IA/experimentos.git
cd experimentos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

O modelo-base padrão é `mlx-community/Qwen2.5-7B-Instruct-4bit`. Tokens reais,
datasets, pesos, resultados e o banco do MLflow não devem ser versionados.

## Uso pelo menu

```bash
python main.py
```

| Opção | Operação | Resultado |
| ---: | --- | --- |
| 1 | Verificar notícia por link | Veredito, justificativa, fontes, JSON e MLflow |
| 2 | Avaliar adapter de evidências | Métricas, predições e MLflow |
| 3 | Treinar novo adapter | Checkpoints LoRA, `best/` e resumo do treino |
| 4 | Pipeline completo de evidências | Preparo, treino e avaliação opcional |
| 5 | Preparar dataset de evidências | `train.jsonl`, `valid.jsonl`, `test.jsonl` e auditoria |
| 6 | Pipeline completo Fake.br | Preparo, treino e avaliação binária |
| 7 | Classificar uma notícia | `FALSA` ou `VERDADEIRA` |
| 8 | Amostragem com Ollama | Linha de base separada, sem adapter MLX |
| 9 | Abrir MLflow | Interface em `http://127.0.0.1:5000` |
| 10 | Preparar somente Fake.br | Splits estratificados |
| 11 | Avaliar somente Fake.br | Métricas e predições |
| 12 | Fazer uma pergunta | Resposta com evidências ou conversa sem busca |
| 13 | Preparar perguntas | Dataset no formato de resposta e justificativa |
| 14 | Baixar datasets públicos | FaQuAD, Pirá e amostra RAG QA PT-BR |
| 15 | Mesclar datasets de perguntas | Dataset deduplicado pronto para ajuste fino |

Também é possível abrir diretamente uma opção:

```bash
python main.py --opcao 12
```

## Fluxo recomendado: perguntas com fontes

### 1. Baixar fontes públicas

```bash
python scripts/baixar_datasets_perguntas.py
```

Os arquivos são gravados isoladamente em `raw/publicos-perguntas-v1/`. O
comando não altera datasets existentes e não inicia treino. Para continuar um
download interrompido:

```bash
python scripts/baixar_datasets_perguntas.py --resume
```

São usados:

- FaQuAD, com perguntas humanas baseadas em textos;
- Pirá, com QA revisado sobre oceano, costa brasileira e clima;
- uma amostra de 5.000 registros do RAG QA Full-Text PT-BR.

Licenças, links e limitações estão em
[docs/DATASETS_PUBLICOS_PERGUNTAS.md](docs/DATASETS_PUBLICOS_PERGUNTAS.md).

### 2. Preparar e mesclar

```bash
python scripts/preparar_dataset_perguntas_publicas.py \
  --base-data-dir data/perguntas-v2 \
  --public-raw-dir raw/publicos-perguntas-v1 \
  --output-dir data/perguntas-mescladas-v1
```

O preparador:

- converte as fontes para conversas compatíveis com MLX-LM;
- adiciona resposta, justificativa, fonte e citação;
- remove duplicatas;
- mantém um documento em apenas um split público;
- cria `train.jsonl`, `valid.jsonl`, `test.jsonl` e `metadata.json`;
- recusa um diretório de saída que já contenha arquivos.

O dataset local preparado nesta máquina contém 7.015 exemplos de treino, 858
de validação e 747 de teste. Essas contagens podem mudar quando as fontes ou os
limites de download forem alterados.

### 3. Fazer um smoke test de treino

Use sempre um diretório novo:

```bash
python scripts/treinar_visual.py \
  --data-dir data/perguntas-mescladas-v1 \
  --adapters-dir models/adapters-perguntas-v2-smoke \
  --iters 5 \
  --experiment-name sift-perguntas-smoke
```

Esse teste confirma dados, memória, GPU, MLX e MLflow. Ele não é um modelo
final.

### 4. Ajustar o adapter de perguntas

```bash
python scripts/treinar_visual.py \
  --data-dir data/perguntas-mescladas-v1 \
  --adapters-dir models/adapters-perguntas-v2 \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --iters 100 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --num-layers 16 \
  --max-seq-length 1024 \
  --steps-per-eval 25 \
  --save-every 25 \
  --experiment-name sift-perguntas-treino
```

Ao final, prefira `models/adapters-perguntas-v2/best/`, quando disponível. O
treinador protege `adapters.safetensors` existente; `--overwrite` deve ser usado
somente quando a substituição for intencional.

### 5. Perguntar e receber links

```bash
python scripts/responder_pergunta.py \
  "É verdade que mulheres recebem menos que homens no Brasil?" \
  --modo evidencias \
  --adapters-dir models/adapters-perguntas-v2/best \
  --max-resultados 5 \
  --min-fontes 2 \
  --max-tentativas 2 \
  --visual
```

Enquanto o adapter de perguntas não estiver treinado, omita `--adapters-dir` e
use o modelo-base. Adapters cujo caminho indique `evidencias`, `fakebr` ou
`classificacao` são recusados nesse fluxo, porque foram treinados para outro
contrato de saída.

Uma resposta conclusiva precisa:

- ser JSON válido;
- conter resposta e justificativa não vazias;
- declarar pelo menos o mínimo configurado de fontes;
- citar cada índice tanto na resposta quanto na justificativa.

Se a primeira geração falhar apenas no formato ou nas citações, o programa faz
uma segunda tentativa corretiva. Todas as tentativas ficam registradas no
MLflow. Se as fontes forem conflitantes ou insuficientes, `INCONCLUSIVA` é o
resultado esperado.

Para conversa sem pesquisa externa:

```bash
python scripts/responder_pergunta.py \
  "Explique o que é overfitting" \
  --modo conversa \
  --visual
```

## Verificar uma URL específica

O fluxo de URL usa um adapter de evidências, não o adapter de perguntas:

```bash
python scripts/verificar_link.py \
  "https://exemplo.com/noticia" \
  --adapters-dir models/adapters-evidencias-v3/best \
  --output results/verificacao_link.json \
  --visual
```

O script valida o destino e cada redirecionamento, aceitando apenas HTTP/HTTPS,
portas padrão e endereços IP públicos. Depois extrai a página, busca fontes de
outros domínios, valida as citações e salva um rastro auditável. Snippets de
busca podem estar incompletos; abra os links antes de tomar decisões.

O adapter de evidências v3 é experimental. O teste local anterior obteve F1
macro de 0,301 e abstinência correta de 0%, abaixo de um nível adequado para
produção.

## Preparar e avaliar evidências

Coloque o FACTCK.BR em `raw/factckbr/FACTCKBR.tsv` e execute:

```bash
python scripts/preparar_dataset_evidencias.py
python scripts/executar_pipeline_evidencias.py --iters 5
```

Depois do smoke test, use outra pasta de adapter para o treino planejado. Faça a
avaliação reservada somente quando escolher a configuração final:

```bash
python scripts/avaliar_evidencias.py \
  --test-jsonl data/evidencias-por-url-v1/test.jsonl \
  --adapters-dir models/adapters-evidencias-por-url-v1
```

O split é agrupado por URL normalizada para reduzir vazamento entre treino e
teste. O conjunto atual possui uma fonte por exemplo; por isso suas métricas
offline não são diretamente equivalentes à regra operacional de duas fontes na
verificação web.

## Fake.br

Coloque o CSV em `raw/preprocessed/pre-processed.csv`. Para executar preparo,
treino e avaliação:

```bash
python scripts/executar_pipeline.py --iters 20 --limit-test 10
```

Após validar o smoke test, execute a configuração completa em um diretório novo.
Para classificar sem treinar:

```bash
python scripts/classificar_noticia.py \
  --adapters-dir models/adapters-fakebr-v1 \
  --texto "Texto da notícia"
```

Esse classificador responde apenas `FALSA` ou `VERDADEIRA`; ele não foi treinado
para produzir justificativa factual.

## Prompts editáveis

Os prompts de perguntas não estão embutidos no Python:

```text
prompts/perguntas_evidencias_sistema.txt
prompts/perguntas_evidencias_usuario.txt
prompts/perguntas_conversa_sistema.txt
prompts/perguntas_conversa_usuario.txt
```

Preserve `{pergunta}`, `{evidencias}` e `{minimo_fontes}` nos arquivos em que
aparecem. Para experimentar outra versão, copie os quatro arquivos e use
`--prompts-dir`. Use a mesma pasta ao preparar dados e na inferência para evitar
divergência entre treino e uso.

## MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abra `http://127.0.0.1:5000`. Por padrão:

- treinos registram parâmetros, perdas, velocidade, memória e adapters;
- avaliações registram métricas e predições;
- perguntas registram prompt, saída bruta, resposta, justificativa, fontes,
  validação e cada tentativa corretiva;
- verificações registram fontes, descartes, citações e rastro observável.

Perguntas e trechos podem conter informações sensíveis. Configure um backend
adequado antes de usar um servidor MLflow compartilhado.

O significado das métricas está documentado em
[docs/METRICAS_E_GRAFICOS.md](docs/METRICAS_E_GRAFICOS.md).

## Estrutura do repositório

```text
docs/       documentação técnica
models/     manifestos e guia de versionamento; pesos ficam fora do Git
prompts/    templates editáveis usados em treino e inferência
scripts/    preparo, treino, avaliação e uso
tests/      testes das regras determinísticas
data/       datasets preparados locais, ignorados pelo Git
raw/        fontes originais locais, ignoradas pelo Git
results/    relatórios gerados localmente, ignorados pelo Git
```

Componentes principais:

| Arquivo | Responsabilidade |
| --- | --- |
| `main.py` | Menu e encaminhamento para os scripts |
| `scripts/responder_pergunta.py` | Perguntas com busca, fontes, retry e MLflow |
| `scripts/perguntas_utils.py` | Templates, contrato JSON e validação de respostas |
| `scripts/baixar_datasets_perguntas.py` | Download retomável de QA público |
| `scripts/preparar_dataset_perguntas_publicas.py` | Normalização, deduplicação e mesclagem |
| `scripts/verificar_link.py` | Verificação segura de uma URL pública |
| `scripts/evidencias_utils.py` | Contrato e validação de vereditos |
| `scripts/treinar_visual.py` | Fine-tuning LoRA e acompanhamento no MLflow |
| `scripts/avaliar_evidencias.py` | Avaliação reservada do adapter de evidências |
| `scripts/auditoria_evidencias.py` | Métricas e rastro observável das fontes |

## Testes

```bash
python -m unittest discover -s tests -v
python -m py_compile main.py scripts/*.py
```

## Versionamento de modelos

Pesos e checkpoints não são enviados ao Git. Consulte
[models/README.md](models/README.md) para publicar adapters no Hugging Face,
registrar hashes e recuperar revisões fixadas.
