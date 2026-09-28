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
| GroupKFold | Validação cruzada que mantém os itens de um grupo (o par fake/true) do mesmo lado. |
| Vazamento | Informação do teste usada no treino. Deixa a nota melhor do que o modelo é. |
