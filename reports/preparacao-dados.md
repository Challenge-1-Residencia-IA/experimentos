# Preparação dos dados

Gerado por `python -m dados.preparar` (`make data`). Os arquivos de dados ficam em `data/` e não são versionados.

## Parâmetros

| Parâmetro | Valor |
|---|---|
| Semente aleatória | 42 |
| Limiar de boilerplate (textos com a frase) | 20 |
| Limiar de quase-duplicata (similaridade de cosseno) | 0.9 |
| Checagens mantidas fora da base | 20% |
| Golden set v1: mensagens / artigos | 80 / 20 |
| Mínimo de palavras para golden set e few-shot | 10 |

## Camada genérica (`data/interim/textos.parquet`)

Textos por dataset e classe:

| dataset   |   fake |   true |
|:----------|-------:|-------:|
| fakebr    |   3600 |   3600 |
| faketrue  |   1791 |   1791 |

Natureza dos textos:

| dataset   |   checagem |   desinformacao |   noticia |
|:----------|-----------:|----------------:|----------:|
| fakebr    |          0 |            3600 |      3600 |
| faketrue  |       1231 |            1791 |       560 |

Sinalizações (textos marcados, nada é descartado nesta camada):

| dataset   |   id_inconsistente |   sem_acento |   duplicado_interno |   quase_duplicata_outro_dataset |
|:----------|-------------------:|-------------:|--------------------:|--------------------------------:|
| fakebr    |                  4 |            4 |                   2 |                               6 |
| faketrue  |                  0 |          589 |                  71 |                               7 |

Variações de boilerplate removidas em `texto_limpo`: 30, além dos padrões de rodapé.

## Visões (`data/processed/`)

| Arquivo | Linhas | Conteúdo |
|---|---|---|
| `base_checagens.parquet` | 1075 | Textos verdadeiros distintos do FakeTrue.Br que formam a base de evidências |
| `avaliacao_busca.parquet` | 1791 | Boatos do FakeTrue.Br com a checagem esperada; 1440 com a checagem na base e 351 sem |
| `golden_set_candidatos.parquet` | 100 | 80 mensagens (uma por checagem, 63 com checagem na base) e 20 artigos, para anotação |
| `fewshot_candidatos.parquet` | 3575 | Textos de desinformação do Fake.br fora do golden set, para exemplos de prompt |

Regras aplicadas:

- Textos de desinformação nunca entram na base de evidências (RD-01).
- Golden set e few-shot não compartilham grupos, e excluem quase-duplicatas entre os datasets (RD-05).
- Parte das checagens fica fora da base, para avaliar o caso "sem evidência suficiente" (IA-07).
