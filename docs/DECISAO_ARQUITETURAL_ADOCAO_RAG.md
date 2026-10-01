# Decisão arquitetural: adoção de RAG para perguntas com evidências

## Status

**Aprovada.** O projeto encerra a linha de fine-tuning voltada à geração de
respostas para perguntas e passa a concentrar o desenvolvimento em Retrieval
Augmented Generation (RAG).

Esta decisão substitui a orientação anterior, que tratava o fine-tuning como
temporariamente pausado. Os experimentos e artefatos existentes permanecem
como histórico técnico, mas não serão evoluídos nem promovidos para produção.

## Contexto

O NexoVerdict IA precisa responder perguntas com base em evidências
verificáveis, indicar as fontes utilizadas e reconhecer quando o material
disponível não é suficiente para uma resposta conclusiva. Para esse tipo de
uso, a qualidade depende principalmente de recuperar conteúdo relevante e
apresentá-lo ao modelo com metadados que permitam rastrear sua origem.

O fine-tuning avaliado no projeto exigiu preparação de datasets, treinamento,
validação de checkpoints e investigação de falhas numéricas. Esse ciclo
consumiu muito tempo de execução e análise, enquanto os resultados práticos
foram pequenos e não produziram um adapter de perguntas confiável.

Além disso, o ajuste dos pesos do modelo não resolve sozinho os principais
requisitos do produto: acesso a informações atualizadas, recuperação de fontes
independentes, citações rastreáveis e atualização rápida da base documental.

## Decisão

O projeto adotará RAG como arquitetura principal para perguntas com
evidências. O fluxo deverá:

1. coletar e normalizar documentos de fontes conhecidas;
2. dividir o conteúdo em trechos adequados à recuperação;
3. indexar os trechos com embeddings e metadados de origem;
4. recuperar os trechos mais relevantes para cada pergunta;
5. fornecer pergunta e evidências ao modelo-base;
6. exigir resposta estruturada, citações e abstenção quando faltar evidência;
7. registrar consultas, fontes recuperadas e métricas no MLflow.

Não serão realizados novos treinamentos, ajustes de hiperparâmetros ou
avaliações de adapters para o fluxo de perguntas durante a implementação do
RAG. Os adapters de classificação e de outras tarefas permanecem separados e
não fazem parte desta decisão.

## Motivos

### Relação entre esforço e resultado

O treinamento local apresentou alto custo de tempo para preparar dados,
executar validações e diagnosticar checkpoints. Mesmo após esse investimento,
o smoke test não entregou respostas utilizáveis e o treino mais longo falhou
numericamente. O ganho potencial não justificou continuar esse ciclo.

### Evidências precisam ser recuperadas no momento da pergunta

O conhecimento incorporado aos pesos fica limitado ao dataset e ao momento do
treino. Com RAG, documentos podem ser adicionados, corrigidos ou removidos sem
treinar novamente o modelo, e cada resposta pode manter vínculo explícito com
os trechos que a sustentam.

### Avaliação alinhada ao produto

RAG permite medir separadamente a qualidade da recuperação e a qualidade da
resposta. Essa separação facilita identificar se uma falha ocorreu porque a
fonte correta não foi encontrada ou porque o modelo interpretou mal uma fonte
adequada.

### Menor tempo de iteração

Alterações no particionamento, nos embeddings, nos filtros, no reranking e no
prompt podem ser avaliadas sem gerar novos pesos. Isso reduz o ciclo entre uma
hipótese e sua validação prática.

## Evidências dos experimentos de fine-tuning

Foram executadas duas tentativas relevantes:

1. Um smoke test de cinco iterações em `data/perguntas-mescladas-v2` produziu
   pesos, mas a avaliação de cinco exemplos teve 0% de JSON válido e respostas
   compostas por `!`.
2. Um treino de 300 iterações, salvo localmente com o nome incorreto
   `adapters-evidencias-v4`, apresentou perda `NaN` desde a primeira validação.

No segundo experimento, muitos prompts atingiam o limite de 1.024 tokens antes
do início da resposta. Como o treino usava `--mask-prompt`, a resposta era
removida por truncamento e alguns lotes ficavam sem tokens-alvo. A auditoria do
dataset de treino v2 encontrou:

| Limite | Prompts que já atingiam o limite |
| ---: | ---: |
| 1.024 tokens | 960 de 5.522 |
| 1.536 tokens | 132 de 5.522 |
| 2.048 tokens | 10 de 5.522 |

O pipeline recebeu proteções contra esse problema e foi criado o dataset
`data/perguntas-mescladas-v3`. Essas correções preservam o aprendizado técnico
do experimento, mas não alteram a relação desfavorável entre custo e benefício.
Os pesos e checkpoints gerados nessas tentativas não devem ser publicados,
retomados ou tratados como modelos válidos.

## Consequências

### Benefícios esperados

- respostas sustentadas por trechos recuperados e fontes rastreáveis;
- atualização do conhecimento sem novo treinamento;
- ciclos de experimentação menores;
- avaliação independente de recuperação, geração, citações e abstenção;
- uso do modelo-base sem manter um adapter específico para perguntas.

### Custos e riscos

- será necessário construir e manter a ingestão e o índice documental;
- a resposta continuará limitada pela qualidade e cobertura das fontes;
- recuperação ruim pode fornecer contexto incompleto ou irrelevante;
- embeddings e índice precisam ser atualizados quando os documentos mudarem;
- citações devem ser validadas para impedir referências sem suporte no texto.

## Plano inicial para o RAG

1. Definir as fontes aceitas e os metadados obrigatórios de cada documento.
2. Criar um pipeline reproduzível de ingestão, limpeza, particionamento e
   indexação.
3. Implementar busca semântica com filtros e retornar texto, fonte e posição.
4. Integrar os trechos recuperados ao fluxo de perguntas existente.
5. Exigir saída estruturada com resposta, citações e indicação de evidência
   insuficiente.
6. Registrar no MLflow a pergunta, os trechos recuperados, a resposta, a
   latência e as métricas de avaliação.
7. Avaliar busca e geração em um conjunto reservado de perguntas.

## Critérios de sucesso

A primeira versão será considerada viável quando:

- recuperar uma fonte relevante entre os primeiros resultados para a maioria
  das perguntas do conjunto de avaliação;
- produzir JSON válido de forma consistente;
- associar cada afirmação factual às fontes recuperadas;
- abster-se quando as evidências forem ausentes ou insuficientes;
- apresentar tempo de resposta adequado ao uso interativo;
- permitir atualização da base sem treinamento do modelo.

## Quando revisar esta decisão

A decisão poderá ser revisada se avaliações do RAG mostrarem uma limitação
recorrente de comportamento que não possa ser resolvida por recuperação,
reranking, prompt ou validação, e se houver dataset curado, métricas claras e
recursos suficientes para demonstrar que um novo fine-tuning oferece ganho
material. Até que essas condições existam, o investimento do projeto seguirá
direcionado ao RAG.
