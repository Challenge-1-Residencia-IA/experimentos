# Golden set v1

Gerado por `python -m dados.golden_set` (`make golden-set`) a partir de `golden_set_v1_anotacao.xlsx`. O arquivo de dados fica em `data/processed/` e não é versionado no git.

## Composição

- Itens: **65** (65 de faketrue).
- Afirmações anotadas: 151 (129 fato verificável, 12 opinião, 6 previsão).
- Temas: 25 política, 22 saúde, 15 outro, 2 ciência.
- Técnicas presentes: urgencia em 26, pedido_compartilhamento em 11, autoridade_anonima em 13, apelo_emocional em 25.
- Sinalizações: sensivel_ao_tempo em 26, precisa_contexto em 32, tema_de_risco em 39.

## Concordância entre anotadores

Rodada com as abas `av_FilipeLacerda` (gabarito) e `av_FelipeRodrigues`, 10 itens em comum. Kappa de Cohen por campo; leitura pelas faixas de Landis e Koch (1977).

| campo                       |   itens |   concordancia_bruta |   kappa | leitura          |
|:----------------------------|--------:|---------------------:|--------:|:-----------------|
| natureza_1                  |      10 |                  0.9 |    0    | leve             |
| tema                        |      10 |                  0.8 |    0.71 | substancial      |
| tec_urgencia                |      10 |                  0.7 |    0.4  | razoável         |
| tec_pedido_compartilhamento |      10 |                  1   |    1    | quase perfeita   |
| tec_autoridade_anonima      |      10 |                  0.5 |    0    | leve             |
| tec_apelo_emocional         |      10 |                  0.6 |    0.31 | razoável         |
| sensivel_ao_tempo           |      10 |                  0.4 |   -0.2  | pior que o acaso |
| precisa_contexto            |      10 |                  0.5 |    0    | leve             |
| tema_de_risco               |      10 |                  0.7 |    0.35 | razoável         |

## Limitações

- Os itens da rodada de concordância foram discutidos entre os anotadores antes da anotação ajustada, o que tende a elevar o kappa.
- Campos em que quase todos os itens têm o mesmo valor (como `natureza_1`) podem ter kappa baixo com concordância bruta alta.
- A maior parte dos itens tem um único anotador.
- Não há casos escritos à mão para sátira, opinião e previsão: as métricas dessas naturezas não são confiáveis nesta versão.

## Pendências

- **GS017**: tema inválido (nan); tec_urgencia vazio ou inválido; tec_pedido_compartilhamento vazio ou inválido; tec_autoridade_anonima vazio ou inválido; tec_apelo_emocional vazio ou inválido; sensivel_ao_tempo vazio ou inválido; precisa_contexto vazio ou inválido; tema_de_risco vazio ou inválido
- **GS076**: afirmacao_1 sem natureza válida (nan); afirmacao_2 sem natureza válida (nan); afirmacao_3 sem natureza válida (nan)
- **GS077**: afirmacao_2 sem natureza válida (nan)
