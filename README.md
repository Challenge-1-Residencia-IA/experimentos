# Experimentos de classificação e verificação de fatos

Este repositório reúne scripts para preparar dados, ajustar modelos locais e
avaliar tarefas de classificação de desinformação. O foco atual são modelos
Qwen 2.5 executados com [MLX-LM](https://github.com/ml-explore/mlx-lm) e duas
formas de trabalho:

1. classificar notícias do conjunto Fake.br;
2. verificar alegações usando trechos de evidência e referências.

O projeto é experimental. Os resultados do modelo ajudam a priorizar uma
verificação, mas não substituem a leitura das fontes e a revisão humana.

## Conteúdo

```text
scripts/    preparação, treino, avaliação e inferência
prompts/    prompts de extração e verificação em desenvolvimento
tests/      testes unitários das regras de evidência
results/    relatórios e predições de experimentos
raw/        datasets de entrada locais, não versionados
data/       datasets preparados, não versionados
models/     adaptadores treinados, não versionados
```

Os principais comandos estão agrupados por tarefa:

| Objetivo | Comando principal | Saída mais importante |
| --- | --- | --- |
| Fake.br: preparar, treinar e avaliar | `scripts/executar_pipeline.py` | `results/metricas_fakebr_*.json` |
| Evidências: preparar e treinar | `scripts/executar_pipeline_evidencias.py` | `data/evidencias-por-url-v1/` e `models/` |
| Avaliar modelo de evidências | `scripts/avaliar_evidencias.py` | `results/metricas_evidencias_*.json` |
| Classificar texto com adaptador | `scripts/classificar_noticia.py` | Veredito no terminal |
| Verificar URL com busca web | `scripts/verificar_link.py` | Relatório JSON opcional |

## Requisitos

- Python 3.10 ou superior.
- Para treino ou inferência local com MLX-LM: Mac com Apple Silicon e memória
  suficiente para o modelo escolhido. O modelo padrão é Qwen 2.5 7B quantizado.
- Acesso à internet na primeira execução para instalar dependências e baixar o
  modelo do Hugging Face e, no fluxo de evidências, para obter o FACTCK.BR e
  pesquisar fontes.
- O dataset Fake.br deve ser obtido separadamente. Dados, pesos e adaptadores
  não são incluídos no Git.

## Tutorial: instalar e rodar

Abra o Terminal e entre na pasta do repositório:

```bash
cd /caminho/para/experimentos
```

Crie e ative um ambiente virtual. No macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No Windows, a ativação é `\.venv\\Scripts\\activate` no Prompt de Comando ou
`.venv\\Scripts\\Activate.ps1` no PowerShell. MLX-LM requer Apple Silicon; em
outras plataformas será necessário adaptar o ambiente de treinamento.

Crie o arquivo local de configurações (opcional para padrões locais):

```bash
cp .env.example .env
```

### Menu principal

Para usar o projeto sem precisar memorizar os scripts e seus parâmetros:

```bash
python main.py
```

O menu reúne verificação por link, avaliação, treinamento, preparação de dados,
pipelines completos, dashboards e MLflow. Antes de executar, ele mostra o
comando completo e pede confirmação. Os scripts em `scripts/` continuam
disponíveis como componentes especializados e para automações.

É possível abrir diretamente uma operação, sem selecionar seu número na tabela:

```bash
python main.py --opcao 2
```

### O que cada opção do menu faz

| Opção | Operação | Entrada principal | Resultado |
| ---: | --- | --- | --- |
| 1 | Verificar notícia por link | URL e adapter de evidências | Veredito, fontes, auditoria JSON e execução no MLflow |
| 2 | Avaliar adapter de evidências | `test.jsonl` e adapter | Métricas gerais e por classe, matriz de confusão, CSV, JSON e MLflow |
| 3 | Treinar novo adapter | Dataset JSONL já preparado | Checkpoints LoRA, `/best`, resumo e curvas no MLflow |
| 4 | Pipeline completo de evidências | TSV do FACTCK.BR | Preparo, treino e avaliação opcional |
| 5 | Preparar dataset de evidências | TSV do FACTCK.BR | `train.jsonl`, `valid.jsonl`, `test.jsonl`, auditoria e metadados |
| 6 | Pipeline completo Fake.br | CSV pré-processado | Divisão, treino binário e avaliação |
| 7 | Classificar uma notícia | Texto ou arquivo e adapter Fake.br | `FALSA` ou `VERDADEIRA` |
| 8 | Amostragem com Ollama | CSV Fake.br e modelo instalado no Ollama | Amostra com busca web e relatório JSON; não usa adapter MLX |
| 9 | Métricas técnicas antigas | Planilha ou predições existentes | Oito métricas separadas e relatório estruturado |
| 10 | Gerar dashboard PNG | Relatório JSON | Imagem com resumo das métricas |
| 11 | Preparar extração antiga | Diretórios `fake/` e `true/` | JSONL para o fluxo antigo de extração |
| 12 | Abrir MLflow | `mlflow.db` | Interface local em `http://127.0.0.1:5000` |
| 13 | Preparar somente o Fake.br | CSV pré-processado | Conjuntos estratificados de treino, validação e teste |
| 14 | Avaliar somente o Fake.br | CSV de teste e adapter | Métricas, predições e execução no MLflow |

As opções 1 a 7 trabalham com os modelos MLX locais. A opção 8 é uma linha de
base separada: chama o serviço Ollama em `localhost:11434` e não usa os adapters
salvos em `models/`.

O MLflow usa SQLite em `mlflow.db` e guarda artefatos em `mlartifacts/` por
padrão. Para abrir a interface de experimentos em outro terminal:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abra `http://127.0.0.1:5000`. É possível configurar `MLFLOW_TRACKING_URI` e
`MLFLOW_EXPERIMENT_NAME` no `.env` ou passar as opções correspondentes aos
scripts.

Cada treinamento registra perda de treinamento, perda de validação, velocidade,
memória e parâmetros. Cada avaliação de evidências registra acurácia, precisão,
revocação, F1 macro, F1 por classe, validade de JSON, validade das citações e
abstinência correta. Os arquivos detalhados ficam anexados à execução como
artefatos. Perda menor indica menor erro no texto previsto, mas não substitui as
métricas da tarefa nem a revisão das fontes.

### Opção A: preparar e treinar com evidências

Baixe a fonte FACTCK.BR para o caminho padrão:

```bash
mkdir -p raw/factckbr
curl -L https://raw.githubusercontent.com/jghm-f/FACTCK.BR/master/FACTCKBR.tsv \
  -o raw/factckbr/FACTCKBR.tsv
```

Faça primeiro um preparo separado para conferir contagens e metadados:

```bash
python scripts/preparar_dataset_evidencias.py
```

Revise `data/evidencias-por-url-v1/metadata.json` e os arquivos
`*_audit.jsonl`. O preparo remove registros curtos, duplicados ou cuja evidência
repete quase integralmente a alegação. Divide por URL de checagem para manter
alegações da mesma página no mesmo split. Somente o treino recebe balanceamento
e exemplos sintéticos sem evidência; validação e teste permanecem sem essas
alterações.

Inicie um treino curto para confirmar que ambiente, dados e modelo carregam:

```bash
python scripts/executar_pipeline_evidencias.py --iters 5
```

Quando estiver pronto para o treino planejado:

```bash
python scripts/executar_pipeline_evidencias.py --iters 100
```

O script baixa o Qwen padrão para o cache do Hugging Face na primeira vez,
treina o adaptador e registra o andamento no MLflow. Por segurança, o teste
reservado não é avaliado automaticamente. Use a validação durante ajustes e,
quando escolher a configuração final, rode a avaliação uma única vez:

```bash
python scripts/avaliar_evidencias.py \
  --test-jsonl data/evidencias-por-url-v1/test.jsonl \
  --adapters-dir models/adapters-evidencias-por-url-v1
```

Para apenas testar o comando de avaliação com poucos exemplos, acrescente
`--limit 10`; não use essa amostra como métrica final. As predições CSV e métricas
JSON são salvas em `results/`.

Ao comparar configurações, passe um `--adapters-dir` novo para cada experimento.
Se o diretório já contém `adapters.safetensors`, o treino para para evitar
substituição acidental. Para retomar explicitamente de um checkpoint:

```bash
python scripts/executar_pipeline_evidencias.py \
  --adapters-dir models/adapters-evidencias-por-url-v2 \
  --resume-adapter-file models/adapters-evidencias-por-url-v1/0000100_adapters.safetensors \
  --iters 100
```

O comando acima continua o treino em um novo diretório de saída. `--overwrite`
permite substituir os adaptadores no diretório indicado, então use-o somente
quando essa for a intenção.

### Opção B: pipeline do Fake.br

Coloque o CSV pré-processado em `raw/preprocessed/pre-processed.csv`. O arquivo
deve conter as colunas esperadas pelo script `preparar_fakebr.py` (incluindo
texto e rótulo). Execute um smoke test primeiro:

```bash
python scripts/executar_pipeline.py --iters 20 --limit-test 10
```

Se o fluxo e os dados estiverem corretos, rode a configuração completa:

```bash
python scripts/executar_pipeline.py
```

A pipeline divide os dados de forma estratificada em treino, validação e teste
(padrão 70/15/15, seed 42), treina sem usar o teste e calcula métricas ao final.
Para outra localização de entrada use `--input-csv caminho/arquivo.csv`. Os
parâmetros `--test-size`, `--valid-size` e `--seed` permitem controlar a divisão.

### Opção C: usar um adaptador existente

Para classificar texto diretamente, sem iniciar treino:

```bash
python scripts/classificar_noticia.py \
  --adapters-dir models/adapters-fakebr-v1 \
  --texto "Texto da notícia que será classificada."
```

Ou leia a notícia de um arquivo UTF-8:

```bash
python scripts/classificar_noticia.py \
  --adapters-dir models/adapters-fakebr-v1 \
  --arquivo caminho/para/noticia.txt
```

Para testar o modelo de evidências com uma URL pública, depois de treinar ou
baixar o adaptador correspondente:

```bash
python scripts/verificar_link.py "https://exemplo.com/noticia" \
  --adapters-dir models/adapters-evidencias-por-url-v1 \
  --output results/verificacao_link.json
```

Esse comando extrai texto da página, busca snippets na web, filtra a própria
URL, redes sociais e domínios repetidos, e pede ao modelo um veredito citado.
Snippets podem não refletir o conteúdo completo das fontes: verifique os links e
revise manualmente resultados importantes.

### Painel visual no terminal

Os scripts de teste mantêm a saída CLI simples para automação. Para exibir
barra de progresso, painéis de veredito, tabelas de métricas e fontes, acrescente
`--visual` ao comando. Exemplos:

```bash
python scripts/verificar_link.py "https://exemplo.com/noticia" --adapters-dir models/adapters-evidencias-v2 --visual
python scripts/avaliar_evidencias.py --adapters-dir models/adapters-evidencias-v2 --visual
python scripts/avaliar_fakebr.py --adapters-dir models/adapters-fakebr-v1 --visual
python scripts/classificar_noticia.py --adapters-dir models/adapters-fakebr-v1 --texto "Texto da notícia" --visual
python scripts/amostrar_e_verificar.py --quantidade 5 --visual
python scripts/metricas_tecnicas.py --visual
```

Nos pipelines completos, `--visual` é repassado para a avaliação final. O
treinamento já possui seu próprio painel de progresso.

## O que cada etapa faz

O fluxo de evidências segue esta sequência:

```text
FACTCK.BR
   ↓ preparação e separação por URL
train.jsonl / valid.jsonl / test.jsonl
   ↓ ajuste fino LoRA com MLX-LM
adapter final + checkpoints + adapter /best
   ↓ avaliação no teste reservado
métricas + predições + matriz de confusão + MLflow
   ↓ uso em notícia real
extração da página + busca web + veredito citado + auditoria
```

### Preparação de evidências

`preparar_dataset_evidencias.py` transforma o TSV do FACTCK.BR em três arquivos
de conversas compatíveis com MLX-LM (`train.jsonl`, `valid.jsonl` e `test.jsonl`).
Também grava versões de auditoria, com URL, editora, rótulo de origem e marcação
de dados sintéticos. O arquivo `metadata.json` registra contagens e limitações.

O split é agrupado por URL normalizada: query, fragmento e `www.` não criam um
grupo diferente. Isso reduz vazamento entre treino e avaliação quando uma
checagem tem mais de uma alegação. É necessário haver ao menos sete URLs
distintas.

### Treino

`treinar_visual.py` valida os JSONL antes de iniciar o MLX-LM, mostra a perda de
treinamento e o progresso, e registra parâmetros e métricas no MLflow. Ao final, salva
`training_summary.json` e tenta copiar o checkpoint de menor perda de validação
para `models/.../best/`. Menor perda não garante melhor qualidade de verificação;
compare as métricas de tarefa e exemplos revisados.

### Avaliação e resultados

`avaliar_evidencias.py` calcula acurácia, precisão, recall e F1 macro, além de
validade do JSON, citações válidas e abstinência em casos inconclusivos. Cada
execução cria arquivos com horário UTC em `results/`. Avaliar muitas vezes o
teste reservado durante ajustes transforma o teste em parte do processo de
seleção; deixe a medição final para depois de escolher o modelo.

A avaliação também cria uma execução no MLflow, no mesmo experimento informado
por `--experiment-name`. O painel recebe métricas gerais, precisão, revocação e
F1 por classe, critérios mínimos de protótipo, CSV de predições e relatório JSON.

## Comandos úteis

Ver opções de qualquer script:

```bash
python scripts/executar_pipeline_evidencias.py --help
python scripts/treinar_visual.py --help
python scripts/avaliar_evidencias.py --help
```

Executar testes unitários:

```bash
python -m unittest discover -s tests -v
```

## Configuração e dados locais

`.env.example` documenta as configurações locais. Copie-o para `.env`; o arquivo
real é ignorado pelo Git e não deve conter credenciais versionadas.

| Variável | Finalidade | Obrigatória? |
| --- | --- | --- |
| `HF_TOKEN` | Autenticação no Hugging Face para modelos privados ou limitados | Não para o Qwen público padrão |
| `HF_MODEL_REPO` | Modelo-base usado pelos fluxos MLX | Não; possui padrão local |
| `OLLAMA_MODEL` | Modelo da comparação auxiliar via Ollama | Somente para a opção 8 |
| `MLFLOW_TRACKING_URI` | Banco SQLite local ou endereço de um servidor MLflow | Não; usa `sqlite:///mlflow.db` |
| `MLFLOW_EXPERIMENT_NAME` | Experimento padrão compartilhado por treino e avaliação | Não |
| `MLFLOW_DISABLE_AGENT_HINT` | Oculta sugestões automáticas do MLflow no terminal | Não |

Datasets, modelos, checkpoints, `.env`, banco MLflow e artefatos locais são
ignorados pelo Git. Consulte as fontes indicadas em `Documentacao/docs/` quando
disponíveis no ambiente do projeto. O FACTCK.BR contém checagens jornalísticas,
não necessariamente fontes primárias, e seus rótulos foram normalizados para
quatro classes: `FALSA`, `VERDADEIRA`, `FORA_DE_CONTEXTO` e `INCONCLUSIVA`.

## Organização do código

`main.py` é a entrada recomendada para uso manual. Ele coleta parâmetros, mostra
o comando e encaminha a operação ao componente correspondente. Essa separação
evita que treino, avaliação e inferência sejam implementados duas vezes.

| Componente | Responsabilidade |
| --- | --- |
| `scripts/evidencias_utils.py` | Prompt compartilhado, contrato JSON e validação do veredito |
| `scripts/ui_terminal.py` | Painéis, tabelas e cores reutilizados pelos comandos visuais |
| `scripts/preparar_dataset_evidencias.py` | Limpeza, agrupamento por URL, divisão e balanceamento |
| `scripts/treinar_visual.py` | Ajuste fino LoRA, checkpoints, seleção do `/best` e MLflow |
| `scripts/avaliar_evidencias.py` | Inferência no teste reservado, métricas, critérios e MLflow |
| `scripts/verificar_link.py` | Extração de página, busca, filtragem de fontes e auditoria |
| `scripts/executar_pipeline_evidencias.py` | Orquestra preparo, treino e avaliação de evidências |
| `scripts/preparar_fakebr.py` | Prepara os conjuntos da classificação binária |
| `scripts/avaliar_fakebr.py` | Avalia a classificação binária e registra no MLflow |
| `scripts/executar_pipeline.py` | Orquestra todo o fluxo Fake.br |
| `scripts/amostrar_e_verificar.py` | Comparação auxiliar usando Ollama sem adapter MLX |
| `scripts/metricas_tecnicas.py` | Mantém a avaliação técnica da planilha legada |
| `scripts/gerar_dashboard*.py` | Produz imagens a partir dos relatórios existentes |

Os scripts continuam executáveis individualmente para automações e reprodução
de experimentos. O menu não contém uma segunda implementação da IA: ele chama
esses componentes com os parâmetros escolhidos.
