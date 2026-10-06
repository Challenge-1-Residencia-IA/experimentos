# Glossário de métricas

Referência rápida para os notebooks de `notebooks/`.

## Qualidade dos dados

| Termo | Definição |
|---|---|
| NaN / NaT | Valor ausente (número / data). **Ausente ≠ zero.** |
| Duplicata | Mesmo texto repetido. Se cair em treino e teste, infla a nota do modelo. |
| Boilerplate | Texto de site capturado pelo scraping (rodapé, chamada). Identifica o veículo, não o conteúdo. |

## Estatística descritiva

| Termo | Definição |
|---|---|
| `mean` | Média. Sensível a valores extremos. |
| `50%` / mediana | Valor do meio. Representa o caso típico melhor que a média. |
| `25%` / `75%` | Quartis: 25% / 75% dos valores estão abaixo deles. |
| `std` | Desvio padrão: o quanto os valores se espalham. |
| Boxplot | Linha = mediana; caixa = 25% a 75%. |

## Comparação entre classes

### Delta de Cliff

Tamanho da diferença entre fake e true, de -1 a +1.

| δ | Significado |
|---|---|
| +1 | todo fake tem valor maior que todo true |
| 0 | sem diferença |
| -1 | todo true tem valor maior que todo fake |

| \|δ\| | Efeito |
|---|---|
| < 0,147 | desprezível |
| 0,147 – 0,33 | pequeno |
| 0,33 – 0,474 | médio |
| ≥ 0,474 | grande |

Faixas de Romano et al. (2006), convertidas dos limiares do d de Cohen (0,2 / 0,5 / 0,8). São convenção, não propriedade da técnica: servem para ordenar e descrever efeitos, não para decidir sozinhas o que é relevante.

### p-valor (Mann-Whitney U)

Chance de a diferença ser acaso. p < 0,05 = significativo. Com milhares de exemplos quase tudo dá p ≈ 0, então **o δ é o que diz se a diferença importa.**

## Texto

| Termo | Definição |
|---|---|
| Stopwords | Palavras comuns sem conteúdo ("de", "que"). Removidas nas análises de vocabulário. |
| n-grama | Sequência de n palavras. Bigrama: "redes sociais". |
| TF-IDF | Peso alto para palavra frequente **neste** texto e rara **nos outros**. |
| chi² | **Força** da associação termo ↔ classe. **Não indica o lado.** |
| `prop_fake` / `prop_true` | % de textos de cada classe que contêm o termo. O maior indica o lado. |
| Canonicalização | Padronizar o texto (sem acento, número, pontuação) antes de comparar, para variações de formatação contarem como iguais. |
| Similaridade de cosseno | Semelhança entre dois textos vetorizados: 0 = nada em comum; 1 = mesmas palavras nas mesmas proporções. |
| Quase-duplicata | Textos quase iguais (ex.: similaridade ≥ 0,9). Em bases grandes, detectados com MinHash + LSH. |
| Índice de Jaccard | Itens em comum ÷ itens no total de duas listas. 1 = idênticas; 0 = nada em comum. |

## Estilometria

| Métrica | Mede | Valor alto indica |
|---|---|---|
| `maiusculas_por_100` | palavras em MAIÚSCULA / 100 palavras | tom apelativo |
| `pron_1_2_sing_por_100` | "eu", "você" / 100 palavras | tom pessoal |
| `emotiveness` | (adjetivos + advérbios) / (substantivos + verbos) | texto carregado |
| `pausality` | pontuação / sentenças | frases mais pausadas |
| `pct_erros_ortograficos` | palavras com erro / total | menos revisão |
| `comp_medio_sentenca` | palavras por sentença | frases longas |
| `comp_medio_palavra` | letras por palavra | vocabulário formal |
| `ttr_normalizado` | palavras distintas / total | vocabulário variado |

Contagens brutas crescem com o tamanho do texto. Por isso as métricas acima são taxas.

## Classificação

| Termo | Definição |
|---|---|
| Acurácia | % de acertos. Com classes 50/50, 0,5 = chute. |
| F1 | Equilíbrio entre precisão (dos marcados como fake, quantos são) e recall (das fakes, quantas achou). |
| Baseline | Modelo simples de referência. Um modelo real precisa superá-lo com folga. |
| Atalho (shortcut) | Padrão que separa as classes sem ser a causa real (tamanho, veículo, época). |
| Mascaramento | Apagar do texto possíveis atalhos (veículos, datas, números, boilerplate) e medir se o modelo ainda acerta. Queda grande = dependia deles. |
| Classificabilidade por origem | Treinar um modelo para prever de qual dataset veio o texto. Acerto alto = datasets com "sotaques" próprios; juntá-los cria atalho. |
| Generalização cruzada | Treinar num dataset e testar em outro. Queda grande em relação ao teste interno = o modelo aprendeu o dataset, não a tarefa. |
| Chute de referência | Acurácia de sempre responder a classe mais frequente. Com classes 50/50 é 0,5; com tamanhos diferentes, é a proporção da maior. |
| GroupKFold | Validação cruzada que mantém os itens de um grupo (o par fake/true) do mesmo lado. |
| Vazamento | Informação do teste usada no treino. Deixa a nota melhor do que o modelo é. |

## Referências

- Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006). *Appropriate statistics for ordinal level data: Should we really be using t-test and Cohen's d for evaluating group differences on the NSSE and other surveys?* Annual meeting of the Florida Association of Institutional Research.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2ª ed.). Lawrence Erlbaum Associates.
