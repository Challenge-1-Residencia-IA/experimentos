# Auditoria e primeira rodada de melhorias — 2026-10-09

Branch: `feat/melhoria-modelo`. Base da revisão: suíte inicial com 58 testes aprovados.
Esta rodada melhora o pipeline RAG e sua avaliação; não altera pesos nem treina um adaptador.

## Escopo da revisão

Inventário dos módulos versionados de `dados`, `rag`, `scripts`, testes, interface,
prompts, configurações e relatórios. Leitura detalhada do caminho de resposta,
busca web, validação, indexação Chroma/Neon, métricas, preparação de casos e
fragmentação. Scripts históricos de treinamento/publicação e notebooks foram
inventariados; não foram todos executados. Isso não equivale a certificar cada
linha ou validar o sistema inteiro em produção.

## Correções implementadas

- A resposta exibe somente a passagem literal conferida. Texto inventado antes
  ou depois das aspas deixa de chegar ao usuário como evidência validada.
- Uma evidência com duas passagens entre aspas é rejeitada; a segunda passagem
  não pode passar sem conferência.
- A verificação literal compara título e corpo separadamente, impedindo citações
  artificiais formadas pela junção dos dois campos.
- Desmentidos na frase que contém a passagem são considerados, reduzindo o risco
  de o modelo recortar a negação e apresentar o restante como apoio.
- A chave do cache de resposta inclui a restrição de inconclusividade e a versão
  do contrato. Uma resposta anterior não substitui essa restrição.
- Páginas, buscas e checagens usam cache de cinco minutos, limitado a 128 entradas
  por categoria e protegido para acesso concorrente. Falhas da API não ficam em
  cache. A resposta completa continua com TTL de dez minutos.
- URLs repetidas e grupos já consultados são removidos antes de novos downloads.
- Contexto de respostas conclusivas descreve os trechos validados, sem usar a
  explicação genérica de uma resposta inconclusiva.
- Métricas contam referências efetivamente resolvidas. Resposta sem citação não
  recebe artificialmente 100% de citações válidas. Essa métrica é estrutural,
  não uma medida de veracidade ou de sustentação semântica.
- Avaliação passa a reportar F1 por classe/macro, concordância incluindo erros,
  conclusões incorretas e quantidade de casos sem gabarito.
- Contrato atualizado para `analise-evidencias-v7`.

## Achados que ainda precisam de experimentos

1. **Gabarito de avaliação:** os 1.791 casos em
   `data/avaliacao/casos_dataset_colega.jsonl` têm zero valores preenchidos de
   `avaliacao_revisada`; 1.440 têm URLs esperadas. Esse conjunto permite avaliar
   recuperação, mas não afirmar acurácia dos vereditos. O relatório do golden set
   registra 65 itens e pendências de anotação. Não atribuir rótulos automaticamente
   para depois usá-los como prova de qualidade.
2. **Corpus local:** 1.075 documentos, dos quais 1.073 são evidências e dois são
   educativos. Nenhum tem `conclusive_eligible=true` explícito. A indexação também
   não preserva esse campo. Portanto, no fluxo local atual, duas fontes qualificadas
   não se tornam disponíveis para uma conclusão. Resolver exige definir a política
   de qualificação, preservar o campo em todos os backends e avaliar o resultado;
   não basta marcar todas as notícias como confiáveis.
3. **Relevância:** correspondência lexical rígida pode perder paráfrases e nomes
   alternativos e confundir eventos com nomes parecidos. Próximo experimento:
   comparar o baseline com busca híbrida e reranker num conjunto fixo, incluindo
   negativos difíceis e alegações sem evidência.
4. **Temporalidade:** publicar uma data e expirar cache não valida a atualidade
   do fato. Mortes, cargos, leis e estatísticas precisam de avaliação temporal
   específica. Fontes distintas também podem reproduzir a mesma reportagem.
5. **Fragmentação:** o divisor recursivo merece teste de preservação de pontuação,
   limites e overlap. Mudá-lo exige versionar a política de fragmentação e
   reindexar, pois hoje a atualização depende principalmente do hash do conteúdo.
6. **Benchmark antigo:** os resultados AVeriTeC documentados são de uma versão
   anterior, com outro contrato. Não representam desempenho desta rodada.
7. **Empacotamento:** `pyproject.toml` distribui apenas `dados`; a execução RAG
   depende do checkout com `rag`, prompts e políticas disponíveis. Uma versão
   instalável completa precisa incluir esses recursos.

## Como validar e comparar

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m scripts.avaliar_rag_mlflow \
  data/avaliacao/casos_dataset_colega.jsonl \
  --retrieval-only --run-name melhoria-modelo-recuperacao
```

Para avaliar respostas completas, fornecer um JSONL com `pergunta`,
`avaliacao_revisada` no contrato atual e `urls_esperadas`, ligar o Ollama e
executar o mesmo avaliador sem `--retrieval-only`. Manter casos de desenvolvimento
e teste separados por alegação/URL e não ajustar regras repetidamente no teste.
Comparar o mesmo corpus, índice e modelo, acompanhando erros, conclusões
incorretas, F1, abstenção, recuperação e tempo.

Os testes novos são regressões sintéticas determinísticas, com geração e busca
simuladas. Não demonstram aumento de acurácia do Qwen. O endpoint local do Ollama
não respondeu durante esta revisão; não foi concluído benchmark real de geração
nem teste de qualidade com busca externa nesta rodada.
