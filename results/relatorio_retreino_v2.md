# Relatório do retreino baseado em evidências — adaptador v2

Data: 2026-09-22

Modelo-base: `mlx-community/Qwen2.5-7B-Instruct-4bit`
Adaptador: `models/adapters-evidencias-v2`

## Objetivo

Substituir a classificação binária baseada apenas no texto da notícia por uma
verificação local de **alegação + evidências recuperadas**, com saída estruturada,
citações e possibilidade de abstinência (`INCONCLUSIVA`).

## Mudanças implementadas

- Foi criado um contrato único de prompt e de resposta JSON para treino,
  avaliação e inferência: `FALSA`, `VERDADEIRA`, `FORA_DE_CONTEXTO` e
  `INCONCLUSIVA`.
- O preparo do FACTCK.BR passou a preservar validação e teste sem exemplos
  sintéticos. O balanceamento acontece apenas no treino; foram mantidos 600
  exemplos de treino (150 por classe), 85 de validação e 85 de teste.
- O treino LoRA passou a expor seed, learning rate, tamanho de sequência,
  número de camadas, checkpoint, retomada explícita e validação do JSONL antes
  de usar GPU. O prompt de entrada é mascarado na loss por padrão.
- Foi criado o avaliador de evidências, que mede F1 macro, JSON válido,
  citações válidas e abstinência correta.
- A verificação por link remove a própria URL, redes sociais e domínios/URLs
  repetidos antes de enviar fontes ao modelo.
- Um veredito conclusivo sem citação válida é rebaixado para `INCONCLUSIVA`.
- A saída por link agora inclui `auditoria_do_veredito`: regra necessária para
  o rótulo emitido, justificativa do modelo, fontes citadas e alertas.

## Treino do v2

Configuração:

```text
iterações: 100
learning rate: 1e-5
batch size: 1
camadas LoRA: 16
max sequence length: 1024
seed: 42
mask prompt: ativo
validação/checkpoint: a cada 25 iterações
```

Curva de validação registrada:

| Iteração | Val loss |
|---:|---:|
| 1 | 0,751 |
| 25 | 0,046 |
| 50 | 0,043 |
| 75 | 0,056 |
| 100 | 0,037 |

O menor `val_loss` foi o da iteração 100. Por isso, `v2/best` é uma cópia do
checkpoint final `0000100`; os arquivos possuem o mesmo SHA-256. Não há ganho
em usar `v2/best` em vez de `v2` neste experimento específico.

## Avaliação offline

Conjunto de teste: 85 exemplos. O relatório está em
`results/metricas_evidencias_20260922T184848Z.json`.

| Métrica | Resultado |
|---|---:|
| Acurácia | 80,0% |
| Precisão macro | 51,8% |
| Recall macro | 39,2% |
| F1 macro | 35,6% |
| JSON válido | 100% |
| Citações válidas | 100% |
| Abstinência correta | 0% |

Resultados por classe:

| Classe | Suporte | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| FALSA | 72 | 92,9% | 90,3% | 91,5% |
| VERDADEIRA | 6 | 100,0% | 16,7% | 28,6% |
| FORA_DE_CONTEXTO | 4 | 14,3% | 50,0% | 22,2% |
| INCONCLUSIVA | 3 | 0,0% | 0,0% | 0,0% |

A acurácia não deve ser usada isoladamente: o teste contém 72 casos `FALSA`.
O F1 macro mostra que o modelo ainda não tem desempenho suficiente nas classes
minoritárias, especialmente em abstinência.

## Testes por link

| Arquivo | Caso | Resultado | Análise |
|---|---|---|---|
| `teste_link_v2.json` e `_2` | Guia de hobbies | `VERDADEIRA` | Caso fora do escopo factual: a página é um guia, não uma alegação verificável. Falta a classe `NÃO_VERIFICÁVEL`/opinião. |
| `teste_link_v2_1.json` | Queda do helicóptero com Rick | `FALSA` | Erro: as fontes recuperadas confirmavam o evento. O adaptador copiou um trecho confirmatório e emitiu rótulo contraditório. |
| `teste_link_v2_3.json` | Artigo satírico do The Onion | `FORA_DE_CONTEXTO` | Erro de categoria. A busca achou notícias reais sobre snipers, mas não confirmou a alegação satírica. Seria necessário `SÁTIRA` ou `INCONCLUSIVA`. |
| `teste_link_v2_4.json` | Discurso de Lula na ONU | `VERDADEIRA` | Correto: a fonte citada confirmou a atribuição da fala. Isso não valida a opinião política contida na fala; valida que ela foi feita. |
| `teste_link_v2_5.json` | Queda do helicóptero com Rick, repetição auditada | `VERDADEIRA` | Correto: a fonte citada confirmou objetivamente que os destroços foram encontrados e que houve mortes. |

Os três primeiros arquivos foram produzidos antes da regra explícita de
contradição/confirmação e antes da auditoria. A repetição do caso do helicóptero
após a atualização do prompt (`_5`) trocou `FALSA` por `VERDADEIRA` sem mudar
os pesos do adaptador. Isso demonstra que o prompt influencia o resultado, mas
não é uma garantia de robustez.

## Diagnóstico

O pipeline técnico funciona: baixa a página, filtra evidências, gera JSON e
cita índices válidos. A fraqueza é a decisão factual do modelo.

As causas principais são:

1. Poucos exemplos naturais de `VERDADEIRA`, `FORA_DE_CONTEXTO` e
   `INCONCLUSIVA` no FACTCK.BR após o filtro.
2. O texto de checagens jornalísticas pode conter padrões linguísticos que
   revelam o rótulo; isso permite aprender estilo em vez de relação entre
   alegação e prova.
3. A evidência recuperada no uso real é um snippet de busca, diferente do
   texto de checagem usado no treino.
4. Não há classes para sátira nem para conteúdo que não apresenta uma alegação
   factual.
5. Uma única fonte independente é suficiente para o fluxo atual; isso é útil
   para cobertura, mas insuficiente para muitos fatos sensíveis.

## Próximos passos recomendados

1. Avaliar o `v3` no mesmo `test.jsonl` e comparar o F1 macro com 35,6%.
2. Não promover o modelo para produção enquanto o recall de `VERDADEIRA` e
   `INCONCLUSIVA` permanecer tão baixo.
3. Adicionar exemplos humanos com fontes primárias/independentes para as
   classes minoritárias; incluir `SÁTIRA` e `NÃO_VERIFICÁVEL` se fizerem parte
   do produto.
4. Criar exemplos adversariais em que as fontes parecem relacionadas, mas não
   sustentam a alegação.
5. Para temas de alto impacto, exigir duas fontes independentes com
   `--min-fontes-independentes 2` e manter revisão humana.

## Reprodução

```bash
source .venv/bin/activate

python scripts/avaliar_evidencias.py \
  --adapters-dir models/adapters-evidencias-v2

python scripts/verificar_link.py "URL_DA_NOTICIA" \
  --adapters-dir models/adapters-evidencias-v2 \
  --output results/teste_link.json

python -m unittest discover -s tests -v
```

Na validação deste commit, todos os 7 testes automatizados passaram.
