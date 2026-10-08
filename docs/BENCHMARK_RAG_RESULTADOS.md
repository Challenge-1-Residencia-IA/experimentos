# Resultados e próximos passos do benchmark RAG

Esta análise usa a execução **concluída** `43322b1a08ba4387927d13f48e94c2b9`,
nome `averitec-ptbr-10x3-seed42`, salva no experimento MLflow `RAG-test`.
Ela é a linha de base para comparar mudanças posteriores. Veja
[como o benchmark foi feito](BENCHMARK_RAG_COMO_FOI_FEITO.md) antes de
interpretar os números.

## Resultado geral

| Medida | Resultado |
| --- | ---: |
| Casos avaliados | 30; 10 por classe |
| Respostas válidas | 21/30 (70%) |
| Vereditos corretos | 9/30 (30%) |
| F1 macro | 0,284 |
| URL esperada entre as seis primeiras | 28/30 (93,3%) |
| MRR | 0,764 |
| Tempo médio por caso | 21,7 s |
| Evidências citadas por resposta válida | 1,52 |

Dos 21 casos com resposta válida, 9 tiveram o veredito esperado (42,9%).
Essa taxa condicional não substitui a acurácia de 30%, que inclui os erros.
O benchmark durou aproximadamente 10 minutos e 56 segundos.

## Vereditos por classe

| Referência | Prevista `FALSA` | Prevista `VERDADEIRA` | Prevista `ENGANOSA` | Prevista `INCONCLUSIVA` | Erro |
| --- | ---: | ---: | ---: | ---: | ---: |
| `FALSA` | 0 | 0 | 1 | 5 | 4 |
| `VERDADEIRA` | 0 | 2 | 1 | 5 | 2 |
| `INCONCLUSIVA` | 0 | 0 | 0 | 7 | 3 |

O sistema se abstém com frequência: entre as 21 respostas válidas, 17 são
`INCONCLUSIVA`. Nenhuma alegação falsa recebeu o veredito `FALSA`. O recall
foi 0 para `FALSA`, 0,2 para `VERDADEIRA` e 0,7 para `INCONCLUSIVA`.

## Diagnóstico dos erros

1. **Citações inválidas:** nove casos terminaram com
   `ValueError: A resposta contém citação ausente de fontes_usadas`. O modelo
   mencionou uma referência no texto, mas não a listou em `fontes_usadas`.
   A validação rejeitou a resposta inteira. A expressão de validação reconhece
   referências como `[1]`, mas não identifica corretamente o formato `[1,2]`;
   isso também pode reduzir o número de origens consideradas pelo validador.
2. **URL encontrada não significa evidência correta:** a URL esperada apareceu
   em 28 casos, mas o trecho mais relevante nem sempre foi selecionado ou
   interpretado corretamente. No caso `averitec-dev-86`, a resposta usou um
   trecho de Facebook pertencente a outra alegação, além de um trecho
   pertinente ao caso. O índice reúne evidências de todos os casos.
3. **Regra de duas origens citadas:** em três respostas válidas o validador
   substituiu a justificativa por falta de duas origens independentes e mudou
   o veredito para `INCONCLUSIVA`. No caso `averitec-dev-484`, a análise
   mencionou `[1,2]`, mas a resposta final ficou sem fontes listadas. A regra
   precisa ser aplicada às evidências que realmente sustentam a conclusão,
   após tratar corretamente os formatos de citação.
4. **Interpretação da alegação:** no caso `averitec-dev-453`, o modelo tratou
   o anúncio de uma suspensão do financiamento da OMS como se a alegação
   dissesse que todos os pagamentos haviam parado. Retornou `ENGANOSA` para
   uma alegação rotulada `VERDADEIRA`. No caso `averitec-dev-149`, respondeu
   `ENGANOSA` para uma alegação rotulada `FALSA`; aqui também convém revisar
   manualmente a fronteira entre esses dois rótulos.
5. **Qualidade do conjunto:** algumas respostas anotadas descrevem páginas
   indisponíveis ou ausência de evidência, e a tradução automática produz
   frases pouco naturais. Isso limita o que o modelo pode concluir mesmo
   quando a URL esperada é recuperada.

A verificação automática confere formato, existência das referências e
diversidade declarada de origens. Ela não comprova que o conteúdo citado
sustenta a afirmação. Os exemplos acima são observações do relatório e
merecem revisão humana antes de alterar rótulos ou políticas de veredito.

## Melhorias em ordem de prioridade

1. **Corrigir citações e preservar a resposta para auditoria.** Aceitar
   formatos como `[1]`, `[1,2]` e `[1, 2]`, exigir consistência com
   `fontes_usadas` e registrar a saída bruta do Qwen quando houver erro.
   Repetir os mesmos 30 casos e medir a queda nos nove erros de validação.
2. **Medir relevância por caso e reduzir contaminação.** Registrar a qual
   alegação cada evidência pertence; calcular se o trecho recuperado vem do
   caso certo e se responde à pergunta específica. Comparar a busca atual
   com uma etapa de reordenação das evidências antes de gerar a resposta.
3. **Separar geração e regra de segurança nas métricas.** Guardar o veredito
   proposto pelo Qwen e o veredito após validação, junto com a razão da
   alteração. Manter a exigência de independência quando ela for necessária,
   mas avaliar seu efeito por classe e por tipo de alegação.
4. **Revisar interpretação e suporte factual.** Criar uma amostra revisada
   por pessoas que marque, para cada resposta, se as citações são pertinentes,
   sustentam o veredito e explicam como verificar a informação. Ajustar
   prompt e seleção de evidências a partir desses erros concretos.
5. **Adicionar uma avaliação de pesquisa ao vivo.** Integrar a busca na web
   à pergunta e montar casos brasileiros recentes com fontes primárias,
   datas, links acessíveis e rótulos revisados. Medir descoberta de fontes,
   leitura do conteúdo, pertinência e resposta separadamente. Este benchmark
   atual não cobre essa etapa.

Após cada mudança, execute o mesmo benchmark com semente 42 e compare com esta
linha de base no MLflow. Depois avalie em um conjunto distinto: mudanças
escolhidas a partir destes 30 casos podem melhorar a nota aqui sem melhorar
o comportamento em alegações novas.
