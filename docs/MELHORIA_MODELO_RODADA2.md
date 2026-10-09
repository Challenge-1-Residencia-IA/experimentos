# Melhoria do RAG — rodada 2

Data: 09/10/2026. Base de comparação: `b56c011`.

## Mudanças

- Divisão em janelas literais do texto limpo, com sobreposição efetiva e limite de tamanho. Evita reconstruir frases ou perder palavras nas fronteiras.
- Preservação de `conclusive_eligible` na ingestão e nos índices Chroma, Neon TCP e Neon HTTPS. Apenas o booleano `true` qualifica uma fonte; strings e valores ausentes não qualificam.
- Mudanças de qualificação invalidam a indexação incremental, inclusive quando o conteúdo não mudou.
- Versão do divisor registrada nos metadados e parâmetros de avaliação/MLflow. Índices antigos são reconstruídos na próxima indexação; a assinatura da atualização automática também considera essa versão.
- Teste reproduzível de geração real com fontes sintéticas, sem contaminar o corpus.

## Validação

Suite local: **85 testes passaram**, com dois avisos de dependências (Chroma e MLflow). `git diff --check` sem erros.

Auditoria dos 1.075 documentos locais, com limite de 180 palavras e sobreposição de 30:

| Critério | Antes | Depois |
| --- | ---: | ---: |
| Total de trechos | 3.792 | 4.354 |
| Trechos que não eram passagens literais do texto limpo | 88 | 0 |
| Trechos acima de 180 palavras | 104 | 0 |

A comparação é contra o texto após limpeza, não contra os bytes originais. A sobreposição aumenta a quantidade de trechos e, portanto, o custo de embeddings e armazenamento.

Geração com `qwen2.5:7b` local: **4/4 cenários controlados** produziram a avaliação esperada:

| Cenário | Resultado | Tempo observado |
| --- | --- | ---: |
| Duas fontes favoráveis de origens diferentes | Evidências suficientes favoráveis | 7,564 s |
| Duas fontes contrárias de origens diferentes | Evidências suficientes contrárias | 5,180 s |
| Duas fontes da mesma origem | Evidências insuficientes | 5,066 s |
| Nenhuma evidência | Evidências insuficientes | 0,002 s |

O último cenário usa o retorno seguro sem precisar gerar texto com o modelo. Esses tempos não incluem busca web nem recuperação vetorial e não são um benchmark de produção. Os quatro casos são sintéticos: **não representam acurácia geral nem comprovam ausência de alucinações**.

Relatório local completo: `results/geracao-controlada-rodada2.json`, com respostas, diagnósticos, hash do prompt e digest do modelo.

## Reproduzir

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.avaliar_geracao_controlada
```

O segundo comando requer Ollama ativo com `qwen2.5:7b` instalado. Não consulta a web e não publica resultados remotamente.

## Ativação e limites

O banco remoto não foi modificado nesta rodada. A nova divisão passa a valer para a recuperação após executar a indexação habitual ou a atualização automática habilitada. A primeira reconstrução pode demorar mais que uma atualização incremental normal.

Não foi atribuída qualificação automática aos documentos existentes: preservar esse campo não significa comprovar que uma fonte está correta. A revisão das fontes, a avaliação temporal, a recuperação de evidências realmente pertinentes e um conjunto de casos com respostas revisadas continuam necessários para medir a qualidade em situações reais.

Esta rodada melhora o pipeline de RAG; não altera nem treina os pesos do modelo.
