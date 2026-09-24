# Guia de métricas, números e gráficos

Este documento explica os números produzidos pelo treinamento, pelas avaliações,
pela verificação de links e pelos dashboards do projeto. Ele também informa o
que pode e o que não pode ser concluído a partir de cada indicador.

## 1. Como ler os números

Há três formatos principais:

- **Proporção:** valor entre `0` e `1`. Por exemplo, `0.82` significa `82%`.
- **Contagem:** número absoluto, como `fontes_citadas = 2` ou `suporte = 40`.
- **Indicador binário:** `0` significa não/falso e `1` significa sim/verdadeiro.

Nas tabelas visuais do terminal, valores `float` entre `0` e `1` são exibidos
automaticamente como porcentagem. Por isso, `0.82` pode aparecer como `82.0%`.
Inteiros e valores fora desse intervalo são exibidos como números absolutos.

Em geral, acurácia, precisão, revocação, F1 e taxas positivas são melhores quando
se aproximam de `1`. As exceções mais importantes são perda de treinamento,
perda de validação, taxa de alucinação, respostas conclusivas sem evidência e
taxa de descarte, que normalmente são melhores quando diminuem.

Nenhuma métrica isolada comprova que uma notícia é verdadeira. Para comparar
dois adapters de forma justa, use exatamente o mesmo conjunto de teste, modelo-
base, prompt, limite de tokens e regras do validador. Na verificação web, registre
também a data, porque resultados de busca e páginas podem mudar.

## 2. Métricas do treinamento LoRA

Estas métricas aparecem como séries por iteração no MLflow e no painel do
terminal durante `treinar_visual.py`.

| Métrica | Significado | Como interpretar |
| --- | --- | --- |
| `perda_treinamento` | Erro médio do modelo nos exemplos usados para ajustar os pesos do adapter. | Menor costuma ser melhor. Uma queda mostra que o modelo está aprendendo o conjunto de treino, mas não garante generalização. |
| `perda_validacao` | Erro nos exemplos de validação, que não atualizam os pesos. | É o principal número usado para escolher o diretório `best`. O menor ponto gera o melhor checkpoint segundo essa métrica. |
| `iteracoes_por_segundo` | Velocidade de processamento do treinamento. | Mede desempenho da máquina, não qualidade do modelo. Quedas podem indicar pressão de memória, carga do sistema ou sequências maiores. |
| `pico_memoria_gb` | Maior uso de memória reportado pelo MLX até aquele ponto. | Serve para capacidade e estabilidade. Não informa se o modelo está aprendendo melhor. |
| `iteracao` ou `step` | Quantidade de passos de otimização já executados. | É o eixo horizontal dos gráficos de treinamento no MLflow. |

### Como interpretar as curvas de perda

- **Treino e validação caindo:** aprendizado saudável naquele intervalo.
- **Treino caindo e validação subindo:** sinal clássico de sobreajuste; o modelo
  memoriza o treino e piora fora dele.
- **As duas praticamente paradas:** pode haver convergência, taxa de aprendizado
  baixa, dados insuficientes ou sinal de treino fraco.
- **Oscilações pequenas:** são normais, principalmente com lotes pequenos.
- **Picos grandes ou `NaN`:** podem indicar taxa de aprendizado alta, dados
  problemáticos ou instabilidade numérica.

O `best` é selecionado pela menor `perda_validacao`. Ele não é automaticamente o
melhor em F1, citações ou verificação factual; sempre execute a avaliação da
tarefa no `best` e no adapter final. Valores de loss só são diretamente
comparáveis quando dados, tokenização, mascaramento, tamanho de sequência e
configuração do treinamento são os mesmos.

## 3. Métricas supervisionadas de classificação

Estas métricas exigem um conjunto de teste com gabarito. Elas aparecem nas
avaliações de evidências e Fake.br.

| Métrica | Cálculo simplificado | O que responde | Direção desejada |
| --- | --- | --- | --- |
| `accuracy` / `acuracia` | acertos / total | Quantas amostras foram classificadas corretamente? | Maior |
| `acuracia_balanceada` | média do recall de cada classe | O modelo acerta as classes sem deixar a classe majoritária dominar o resultado? | Maior |
| `precision` / `precisao` | TP / (TP + FP) | Quando o modelo escolhe uma classe, com que frequência está correto? | Maior |
| `recall` / `revocacao` | TP / (TP + FN) | De todos os exemplos reais da classe, quantos foram encontrados? | Maior |
| `f1` | média harmônica entre precisão e recall | Há equilíbrio entre falsos positivos e falsos negativos? | Maior |
| `suporte` | quantidade real de exemplos da classe | Quantos casos sustentam a métrica daquela classe? | Não é nota |
| `coeficiente_matthews` | correlação entre gabarito e predição | Qualidade global da classificação, inclusive com classes desbalanceadas. | Mais próximo de `1` |

No coeficiente de Matthews, `1` representa predição perfeita, `0` desempenho sem
correlação e `-1` oposição perfeita ao gabarito. O valor pode ser negativo.

### Macro, weighted e por classe

- **Macro:** calcula a métrica separadamente para cada classe e tira a média sem
  considerar o tamanho das classes. É útil para não esconder uma classe ruim.
- **Weighted:** faz a média ponderada pelo suporte. Representa melhor o volume do
  conjunto, mas uma classe grande pode esconder falhas nas classes pequenas.
- **Por classe:** mostra onde o modelo erra. Para notícias, observe especialmente
  classes de risco, e não apenas o valor global.

Exemplo: precisão alta e recall baixo para `FALSA` significa que, quando acusa
uma notícia de falsa, o modelo costuma acertar, mas deixa muitas notícias falsas
passarem. Recall alto e precisão baixa significa que encontra muitas falsas,
porém também acusa notícias verdadeiras indevidamente.

### Métricas adicionais do Fake.br

| Métrica | Significado |
| --- | --- |
| `indefinidas` | Quantidade de respostas que não puderam ser interpretadas como `fake` ou `true`. |
| `taxa_respostas_interpretaveis` | Fração de respostas que respeitou uma das duas classes esperadas. Quanto mais próximo de `1`, melhor o cumprimento do formato. |
| `precisao_fake`, `revocacao_fake`, `f1_fake` | Desempenho considerando `fake` como a classe de interesse. |
| `precisao_true`, `revocacao_true`, `f1_true` | Desempenho considerando `true` como a classe de interesse. |

O classificador Fake.br avalia padrão textual e só produz a classe. Ele não
recebe pesquisa web nem oferece justificativa factual auditável. Portanto, um
alto F1 nesse conjunto não significa que ele consiga conferir uma notícia atual.

## 4. Matriz de confusão

A matriz de confusão cruza gabarito e resposta:

- **Linhas:** classe verdadeira, isto é, o gabarito.
- **Colunas:** classe prevista pelo modelo.
- **Diagonal principal:** acertos.
- **Fora da diagonal:** tipos específicos de erro.

Uma célula na linha `VERDADEIRA` e coluna `FALSA`, por exemplo, conta notícias
verdadeiras acusadas como falsas. Não olhe apenas para a cor mais forte: classes
com mais suporte naturalmente produzem células maiores. Compare também os
percentuais por linha e o F1 de cada classe.

## 5. Métricas de formato, abstinência e qualidade mínima

Produzidas por `avaliar_evidencias.py`:

| Métrica | Significado | Interpretação |
| --- | --- | --- |
| `taxa_json_valido` | Fração das respostas que continha JSON analisável. | Próximo de `1` indica formato estável; não mede verdade factual. |
| `taxa_citacoes_validas` | Fração com índices de fontes existentes e válidos. | Próximo de `1` indica que o modelo não inventa números de fontes. Não garante que a fonte sustente a frase. |
| `taxa_abstencao_correta` | Entre os casos cujo gabarito é `INCONCLUSIVA`, fração em que o modelo também se absteve. | Maior é melhor. Mede prudência quando faltam evidências. |
| `proporcao_criterios_atendidos` | Critérios de qualidade aprovados / critérios avaliados. | Resumo administrativo; confira quais critérios falharam. |
| `aprovado_prototipo` | `1` somente quando todos os critérios configurados passam. | É um portão técnico para protótipo com revisão humana, não aprovação para produção autônoma. |

Os limites atualmente codificados para o protótipo são:

- F1 macro maior ou igual a `0.60`;
- JSON válido maior ou igual a `0.95`;
- citações válidas maior ou igual a `0.95`;
- abstinência correta maior ou igual a `0.60`;
- F1 de cada classe presente maior ou igual a `0.50`.

Esses limites são escolhas do projeto, não padrões universais. Devem ser revistos
com base no risco do uso, no tamanho do teste e no custo de cada tipo de erro.

## 6. Métricas de recuperação e auditoria de evidências

Na verificação de link, não existe gabarito automático para aquela URL. Portanto,
o sistema registra qualidade do processo e rastreabilidade, mas não pode calcular
acurácia ou F1 de uma notícia isolada.

### Recuperação das fontes

| Métrica | Tipo | Significado |
| --- | --- | --- |
| `resultados_brutos_da_busca` | Contagem | Resultados devolvidos pela busca antes dos filtros. |
| `fontes_descartadas_pelos_filtros` | Contagem | Resultados removidos por URL inválida, mesmo site da notícia, rede social, URL repetida ou domínio repetido. |
| `fontes_nao_apresentadas_por_limite` | Contagem | Fontes válidas que ficaram de fora por causa de `--max-resultados`. |
| `taxa_descarte_da_busca` | Proporção | Fontes descartadas / resultados brutos. Alta pode indicar busca ruidosa, mas também filtros rigorosos. |
| `fontes_recuperadas` | Contagem | Fontes efetivamente apresentadas ao modelo. |
| `dominios_unicos` | Contagem | Quantidade de domínios diferentes entre as fontes apresentadas. |
| `diversidade_de_dominios` | Proporção | Domínios únicos / fontes recuperadas. `1` significa que cada fonte veio de um domínio diferente. |
| `fontes_prioritarias_recuperadas` | Contagem | Fontes recuperadas cujo domínio pertence à lista heurística prioritária. |
| `taxa_fontes_prioritarias_recuperadas` | Proporção | Fontes prioritárias / fontes recuperadas. |
| `extracao_html_direta` | Binário | `1` quando o texto foi extraído diretamente da página; `0` quando outro método foi usado ou não informado. |
| `caracteres_noticia` | Contagem | Tamanho do texto extraído e enviado ao fluxo. Muito baixo pode indicar bloqueio ou extração incompleta. |
| `minimo_fontes_exigido` | Contagem/configuração | Número mínimo de fontes válidas exigido para manter um veredito conclusivo. Não é uma nota de desempenho. |

O bloco `auditoria_da_recuperacao` também contém números usados para explicar o
funil da busca:

| Campo | Significado |
| --- | --- |
| `resultados_brutos` | Total retornado pela busca antes da limpeza. |
| `fontes_aceitas_apos_filtros` | Total restante depois dos filtros, antes de aplicar o limite de fontes do prompt. |
| `fontes_descartadas` | Soma de todos os descartes. |
| `fontes_apresentadas_ao_modelo` | Total efetivamente incluído no prompt. |
| `dominios_apresentados` | Lista de domínios incluídos; é uma lista para auditoria, não uma métrica. |

Em `descartes_por_motivo`, cada número é a quantidade removida por uma regra:

- `sem_url_valida`: resultado sem endereço utilizável;
- `mesmo_site_da_noticia`: autocitação ou outra página do portal analisado;
- `rede_social`: resultado de domínio social bloqueado pelo filtro;
- `url_repetida`: mesmo endereço já aceito;
- `dominio_repetido`: outra página de um domínio já representado.

Uma fonte “prioritária” apenas pertence a uma lista de domínios definida no
código. Isso é uma heurística de procedência, não uma garantia de que o trecho é
correto, atual ou relevante para a alegação.

### Uso das fontes e justificativa

| Métrica | Tipo | Significado |
| --- | --- | --- |
| `fontes_citadas` | Contagem | Quantas fontes válidas o modelo declarou usar. |
| `cobertura_fontes_recuperadas` | Proporção | Fontes citadas / fontes apresentadas. Cobertura alta não é objetivo isolado: citar fonte irrelevante não melhora a resposta. |
| `requisito_minimo_fontes_atendido` | Binário | `1` quando uma resposta conclusiva cita o mínimo configurado, ou quando o resultado final é inconclusivo. |
| `fontes_prioritarias_citadas` | Contagem | Quantas fontes citadas pertencem à lista de domínios prioritários. |
| `taxa_fontes_prioritarias_citadas` | Proporção | Fontes prioritárias citadas / total de fontes citadas. |
| `justificativa_presente` | Binário | Indica se existe texto de justificativa. Não avalia se o texto está correto. |
| `palavras_justificativa` | Contagem | Tamanho da justificativa em palavras. Mais palavras não significam maior qualidade. |
| `fontes_mencionadas_na_justificativa` | Contagem | Quantas fontes citadas também aparecem explicitamente como `[N]` na justificativa. |
| `taxa_fontes_citadas_mencionadas_na_justificativa` | Proporção | Fontes citadas mencionadas no texto / fontes citadas. |
| `sobreposicao_justificativa_evidencias` | Proporção | Fração dos termos relevantes da justificativa encontrada nos títulos e trechos citados. |
| `sobreposicao_alegacao_evidencias` | Proporção | Fração dos termos relevantes da alegação encontrada nos títulos e trechos citados. |
| `veredito_conclusivo` | Binário | `1` para veredito final diferente de `INCONCLUSIVA`; `0` para abstinência. |
| `veredito_alterado_pelo_validador` | Binário | `1` quando uma regra determinística mudou ou rebaixou a saída original. |
| `resposta_json_valida` | Binário | Versão por caso da validade do JSON. Mede estrutura, não conteúdo factual. |
| `citacoes_validas` | Binário | Versão por caso da validade dos índices citados. Não mede relevância semântica. |

As sobreposições são lexicais: comparam palavras, não significado. Um valor alto
pode ocorrer porque uma fonte repete a alegação sem confirmá-la; um valor baixo
pode ocorrer quando a fonte confirma a informação usando sinônimos. Elas ajudam
na auditoria, mas não são métricas de veracidade nem de confiança probabilística.

### Índice de auditabilidade

`indice_auditabilidade_heuristico` é a média de cinco condições binárias:

1. resposta em JSON válido;
2. citações com índices válidos;
3. justificativa presente;
4. requisito mínimo de fontes atendido;
5. resposta inconclusiva ou sobreposição justificativa–evidências de pelo menos
   `0.20`.

Assim, os valores possíveis por caso são `0`, `0.2`, `0.4`, `0.6`, `0.8` e `1`.
Ele mede o quanto a resposta é inspecionável segundo essas regras, não se o
veredito está correto. Uma resposta pode ter auditabilidade `1` e ainda estar
factualmente errada.

### Médias no conjunto de avaliação

Os nomes iniciados por `media_` são médias das métricas por amostra. Os nomes
iniciados por `taxa_` são proporções de amostras que atendem uma condição:

- `media_fontes_citadas`;
- `media_cobertura_fontes`;
- `taxa_requisito_minimo_fontes_atendido`;
- `taxa_justificativa_presente`;
- `media_sobreposicao_justificativa_evidencias`;
- `media_sobreposicao_alegacao_evidencias`;
- `media_indice_auditabilidade_heuristico`;
- `taxa_veredito_alterado_pelo_validador`.

Uma taxa alta de alterações pelo validador mostra que a proteção está atuando,
mas também que o modelo ainda não cumpre sozinho o contrato de saída. Analise os
campos `ajustes_aplicados`, `veredito_original_modelo` e `veredito` no CSV.

## 7. Rastro da decisão

O campo `rastro_da_decisao` documenta o caminho observável da resposta:

- `entrada`: alegação e método de extração;
- `recuperacao`: quantidade, domínios e descartes da busca;
- `regra_de_decisao`: mínimo de fontes e exigência de citações válidas;
- `saida_declarada_pelo_modelo`: veredito, justificativa e fontes antes da
  validação;
- `validacao`: formato, citações, ajustes e veredito final;
- `fontes_citadas_com_trechos`: material que pode ser revisado por uma pessoa;
- `metricas`: indicadores calculados para aquele caso.

Esse rastro não é cadeia de pensamento privada, explicação causal completa ou
prova de verdade. Ele permite conferir entradas, declarações e regras aplicadas.

## 8. As oito métricas técnicas legadas

O fluxo `metricas_tecnicas.py` usa uma planilha/CSV estruturado e gera o dashboard
de métricas técnicas. Suas métricas não devem ser confundidas com a avaliação
mais recente de evidências.

| Número | Métrica | Significado | Limitação principal |
| --- | --- | --- | --- |
| 1 | Precisão macro | Média não ponderada da precisão das cinco classes. | Depende do gabarito e das classes presentes. |
| 2 | Recall macro | Média não ponderada do recall das cinco classes. | Pode cair muito por uma única classe fraca. |
| 3 | F1 macro | Média não ponderada do F1 das classes. | Não mostra qual classe causou o problema. |
| 4 | Taxa de identificação correta das afirmações | Casos em que alguma alegação extraída teve Jaccard lexical de pelo menos `0.25` com o gabarito. | Sinônimos e paráfrases podem ser penalizados. |
| 5 | Qualidade das fontes | Fontes em domínios marcados como confiáveis / total recuperado. | Avalia domínio, não conteúdo ou relevância. |
| 6 | Taxa de alucinação | Frases da justificativa com menos de `0.20` de sobreposição lexical com o contexto / total de frases. | É lexical; não detecta corretamente todas as alucinações. Menor é melhor. |
| 7 | Respostas conclusivas sem evidência | Casos sem evidência nos quais foi emitida resposta conclusiva / total de casos. | Depende da marcação correta de ausência de evidência. Menor é melhor. |
| 8 | Precisão de divergências | Divergências corretas / todas as divergências apontadas. | Deve ser lida junto do recall e F1 de divergências. |

O relatório também inclui `recall_divergencias`, `f1_divergencias`, contagens de
TP/FP/FN/TN, métricas weighted, suporte e detalhes por caso.

### Contadores que formam as métricas legadas

Esses valores são denominadores ou diagnósticos, não notas isoladas:

| Grupo | Contadores | Significado |
| --- | --- | --- |
| Extração | `total_casos`, `total_afirmacoes_esperadas`, `total_afirmacoes_extraidas`, `afirmacoes_corretamente_identificadas` | Volume avaliado, volume extraído e quantidade de casos que passou o limiar lexical. |
| Recuperação | `total_fontes_recuperadas`, `fontes_alta_qualidade`, `fontes_baixa_qualidade_ou_ruido` | Volume total e classificação heurística dos domínios. Fontes fora das duas listas podem não entrar nas duas últimas contagens. |
| Ausência de evidência | `casos_sem_evidencia`, `respostas_assertivas_sem_evidencia` | Casos sem material e subconjunto em que o sistema mesmo assim concluiu. |
| Divergência | `true_positives_divergencia`, `false_positives_divergencia`, `false_negatives_divergencia`, `true_negatives_divergencia` | Acertos, alarmes falsos, divergências perdidas e concordâncias corretas. |
| Geração | `total_justificativas_avaliadas`, `total_afirmacoes_sintetizadas`, `afirmacoes_suportadas_evidencia`, `afirmacoes_alucinadas` | Quantidade de textos, frases avaliadas e classificação lexical das frases. |
| Classificação | `total_amostras` | Denominador das métricas globais de classe. |

## 9. O que cada gráfico significa

### Gráficos do MLflow — treinamento

Ao abrir uma execução de ajuste fino, o MLflow desenha linhas ao longo do `step`:

- **Perda de treinamento:** deve cair sem ser usada sozinha para escolher o
  modelo.
- **Perda de validação:** o menor ponto indica o checkpoint copiado para `best`.
- **Iterações por segundo:** monitora velocidade, não qualidade.
- **Pico de memória:** monitora recursos e risco operacional.

Ao comparar runs, sobreponha apenas execuções equivalentes. Uma curva com menos
loss em outro dataset não prova que o adapter é superior.

### Gráficos e comparações do MLflow — avaliação

As métricas de avaliação são registradas uma vez por execução, sem uma série de
steps. Ao selecionar várias runs, o MLflow pode mostrá-las em barras, tabela ou
gráfico comparativo. Cada ponto/barra representa um adapter avaliado, não uma
etapa do treinamento. Use os parâmetros e tags da run para confirmar adapter,
dataset, quantidade de amostras e limites antes de comparar.

Os campos `modelo`, `diretorio_adaptadores`, `conjunto_teste`, `quantidade_amostras`
e `limite_*` são parâmetros de contexto. Eles explicam a execução, mas não são
métricas de qualidade.

### Painel “As 8 métricas definidas”

É um gráfico de barras horizontais com os oito indicadores legados. Para a maior
parte das barras, valores altos são desejáveis. Nas barras **Taxa de alucinação**
e **Respostas conclusivas sem evidência**, valores baixos são melhores. As cores
são apenas regras visuais simples; não substituem critérios de aprovação e não
invertem automaticamente o significado das métricas de erro.

### Painel “F1 por classe”

Cada barra mostra o equilíbrio entre precisão e recall daquela classe. Uma barra
baixa identifica uma classe que precisa de mais dados, exemplos melhores ou
ajuste de prompt/treino. Leia junto do suporte: uma classe com poucos exemplos
gera uma estimativa instável.

### Painel “Matriz de confusão”

Mostra quais classes são confundidas entre si. Linhas são o gabarito e colunas
são as predições. A diagonal contém acertos; células fora dela mostram erros.

### Painel “Diagnóstico gerado a partir do JSON”

É um resumo textual das métricas do relatório: amostras, acurácia, extração,
procedência das fontes, cobertura, divergências e alucinação lexical. Não é uma
análise independente; apenas apresenta números já calculados.

### Dashboard de verificação amostral

| Painel | O que mostra | Cuidado ao interpretar |
| --- | --- | --- |
| Vereditos do modelo | Contagem de `VERDADEIRA`, `FALSA` e `INCONCLUSIVA`. | Distribuição não é acurácia. Muitas conclusões podem significar baixa prudência. |
| Confiança declarada | Contagem de `ALTA`, `MÉDIA`, `BAIXA` ou ausente, conforme texto gerado. | É autodeclarada, não calibrada e não equivale à probabilidade de acerto. |
| Fontes recuperadas por mensagem | Número de resultados encontrados para cada amostra. | Quantidade não mede independência, relevância nem correção. |
| Resumo da execução | Fontes com URL, taxa conclusiva e concordância com o corpus entre respostas binárias. | A concordância exclui inconclusivas e não substitui métricas completas com gabarito. |

## 10. Leitura recomendada para decidir entre adapters

Use esta ordem:

1. confirme que o teste é o mesmo e que não vazou para o treinamento;
2. confira suporte e distribuição das classes;
3. compare F1 macro, F1 por classe e matriz de confusão;
4. confira abstinência correta e erros de alto risco;
5. confira JSON, citações, mínimo de fontes e alterações do validador;
6. abra o CSV e revise manualmente uma amostra de acertos e erros;
7. use loss de validação como informação complementar, não como veredito final.

Para uma decisão mais confiável, registre também intervalo de confiança ou rode
bootstrap quando o conjunto de teste for pequeno. Diferenças mínimas, como
`0.72` contra `0.73`, podem ser apenas variação da amostra.

## 11. Onde encontrar cada resultado

| Resultado | Local |
| --- | --- |
| Histórico e melhor loss de validação | `models/<adapter>/training_summary.json` |
| Métricas de treinamento | MLflow, execução `ajuste-fino-lora-*` |
| Avaliação de evidências | `results/metricas_evidencias_*.json` |
| Predições e rastros por caso | `results/predicoes_evidencias_*.csv` |
| Avaliação Fake.br | `results/metricas_fakebr_*.json` |
| Predições Fake.br | `results/predicoes_fakebr_*.csv` |
| Verificação individual por link | caminho informado em `--output` |
| Verificação amostral | `results/verificacao_amostral.json` |
| Métricas técnicas legadas | `results/relatorio_metricas_tecnicas.json` |
| Dashboards em imagem | arquivos `dashboard_*.png` em `results/` |
