# Métricas e gráficos do projeto

Este guia explica os números registrados pelo treino, pelas avaliações, pelas
perguntas e pela verificação de links. Nenhuma métrica isolada comprova que uma
notícia ou resposta seja verdadeira.

## Convenções

- Proporções variam de `0` a `1`; `0.82` equivale a 82%.
- Contagens são números absolutos, como `fontes_citadas = 2`.
- Indicadores binários usam `0` para não e `1` para sim.
- Métricas só devem ser comparadas entre execuções com o mesmo dataset, modelo-
  base, prompt, limites e regras de validação.

## Treinamento LoRA

O `treinar_visual.py` registra séries por iteração:

| Métrica | Significado |
| --- | --- |
| `perda_treinamento` | Erro do modelo nos exemplos usados para atualizar o adapter. |
| `perda_validacao` | Erro em exemplos que não atualizam os pesos. |
| `iteracoes_por_segundo` | Velocidade da máquina, não qualidade do modelo. |
| `pico_memoria_gb` | Maior uso de memória reportado pelo MLX. |

Treino e validação caindo sugerem aprendizado naquele intervalo. Treino caindo
com validação subindo sugere sobreajuste. O diretório `best/` é escolhido pela
menor perda de validação disponível, mas deve ser confirmado nas métricas da
tarefa: loss menor não garante melhor F1, citações ou correção factual.

## Classificação supervisionada

As avaliações de evidências e Fake.br usam um conjunto com gabarito:

| Métrica | Pergunta respondida |
| --- | --- |
| Acurácia | Qual fração de todas as previsões está correta? |
| Acurácia balanceada | O modelo atende às classes sem favorecer a majoritária? |
| Precisão | Quando prevê uma classe, com que frequência acerta? |
| Recall/Revocação | Quantos exemplos reais daquela classe foram encontrados? |
| F1 | Qual o equilíbrio entre precisão e recall? |
| F1 macro | Como está a média das classes sem ponderar pelo tamanho? |
| Suporte | Quantos exemplos reais sustentam a métrica da classe? |
| Matthews | Qual a correlação global entre gabarito e previsão? |

Precisão alta e recall baixo para `FALSA` significa que as acusações costumam
estar corretas, mas muitas notícias falsas passam. Recall alto e precisão baixa
significa que o modelo encontra muitas falsas e também acusa verdadeiras.

### Matriz de confusão

As linhas são as classes reais e as colunas são as previsões. A diagonal contém
acertos; células fora dela mostram quais classes são confundidas. Leia a matriz
junto do suporte e do F1 por classe.

### Fake.br

`indefinidas` conta saídas que não puderam ser interpretadas como `fake` ou
`true`. `taxa_respostas_interpretaveis` mede cumprimento do formato. Esse fluxo
classifica padrões textuais e não recebe pesquisa web; F1 alto no Fake.br não
prova capacidade de verificar fatos atuais.

## Formato, citações e abstinência

Produzidas por `avaliar_evidencias.py` e pelos fluxos online:

| Métrica | Significado |
| --- | --- |
| `taxa_json_valido` | Fração das respostas com todos os campos JSON esperados. |
| `taxa_citacoes_validas` | Fração cujos índices apontam para fontes existentes. |
| `taxa_abstencao_correta` | Casos inconclusivos em que o modelo também se absteve. |
| `fontes_citadas` | Quantidade de fontes declaradas pelo modelo. |
| `requisito_minimo_fontes_atendido` | Se a conclusão respeitou o mínimo configurado. |
| `veredito_alterado_pelo_validador` | Se uma regra determinística mudou a saída. |

JSON e citações válidas medem estrutura, não veracidade. Uma fonte pode existir
e ainda ser irrelevante, desatualizada ou incorreta.

Os critérios atuais do protótipo de evidências são:

- F1 macro ≥ 0,60;
- JSON válido ≥ 0,95;
- citações válidas ≥ 0,95;
- abstinência correta ≥ 0,60;
- F1 de cada classe presente ≥ 0,50.

São limites internos para um protótipo com revisão humana, não certificação para
produção autônoma.

## Recuperação e auditoria das fontes

Uma URL ou pergunta real não possui gabarito automático. Nesse caso o sistema
registra rastreabilidade do processo:

| Métrica | Significado |
| --- | --- |
| `resultados_brutos_da_busca` | Itens devolvidos antes dos filtros. |
| `fontes_descartadas_pelos_filtros` | Itens removidos por URL, domínio ou regra de segurança. |
| `fontes_recuperadas` | Fontes apresentadas ao modelo. |
| `dominios_unicos` | Diversidade aproximada das origens. |
| `fontes_nao_apresentadas_por_limite` | Fontes válidas cortadas pelo limite do prompt. |
| `extracao_html_direta` | Se o texto veio diretamente da página. |
| `caracteres_noticia` | Tamanho da notícia enviada ao fluxo. |

Domínios distintos não garantem independência editorial: veículos podem repetir
o mesmo relatório ou pertencer ao mesmo grupo. Uma fonte “prioritária” também é
apenas uma heurística de procedência.

### Justificativa

| Métrica | Significado |
| --- | --- |
| `justificativa_presente` | Existe texto explicativo? |
| `fontes_mencionadas_na_justificativa` | Quantas citações `[N]` aparecem na explicação? |
| `sobreposicao_justificativa_evidencias` | Fração lexical compartilhada com os trechos citados. |
| `sobreposicao_alegacao_evidencias` | Fração lexical compartilhada entre alegação e fontes. |
| `indice_auditabilidade_heuristico` | Resumo de formato, citações, justificativa e sustentação lexical. |

Sobreposição compara palavras, não significado. Uma página pode repetir uma
alegação falsa e obter sobreposição alta; uma confirmação com sinônimos pode
obter sobreposição baixa.

O índice de auditabilidade é a média de cinco condições: JSON válido, índices
válidos, justificativa presente, mínimo de fontes atendido e abstinência ou
sobreposição mínima. Ele mede facilidade de inspeção, não correção factual.

## Perguntas e tentativas corretivas

Cada pergunta registra:

- quantidade de fontes recuperadas e citadas;
- JSON e citações válidas;
- resposta conclusiva ou abstida;
- prompt final, saída bruta e diagnóstico do validador;
- todas as tentativas de correção de formato.

Uma segunda tentativa bem-sucedida significa que o contrato foi corrigido. Ela
não transforma evidência fraca em evidência forte. O resultado deve continuar
`INCONCLUSIVA` quando as fontes não respondem à pergunta ou entram em conflito.

## Rastro observável

`rastro_da_decisao` documenta entrada, recuperação, regras, saída declarada,
ajustes do validador e trechos citados. Esse rastro não é cadeia de pensamento
privada nem prova de verdade; ele reúne elementos que uma pessoa consegue
conferir.

## Gráficos no MLflow

Durante o treino, os gráficos são séries ao longo dos steps. Na avaliação, cada
barra ou ponto normalmente representa uma execução completa. Antes de comparar
runs, confira os parâmetros `modelo`, `diretorio_adaptadores`, dataset,
quantidade de amostras, prompt e limites.

Ordem recomendada para escolher um adapter:

1. confirme que treino e teste não se misturam;
2. confira suporte e distribuição das classes;
3. compare F1 macro, F1 por classe e matriz de confusão;
4. confira abstinência e erros de maior risco;
5. confira JSON, citações e alterações do validador;
6. revise manualmente acertos e erros;
7. use loss de validação apenas como informação complementar.

## Onde encontrar os resultados

| Resultado | Local |
| --- | --- |
| Histórico e melhor loss | `models/<adapter>/training_summary.json` |
| Métricas do treino | MLflow, execução `ajuste-fino-lora-*` |
| Avaliação de evidências | `results/metricas_evidencias_*.json` |
| Predições de evidências | `results/predicoes_evidencias_*.csv` |
| Avaliação Fake.br | `results/metricas_fakebr_*.json` |
| Perguntas e justificativas | experimento `nexoverdict-perguntas` no MLflow |
| Verificação individual | caminho informado em `--output` e MLflow |

`results/` é gerado localmente e não é versionado.
