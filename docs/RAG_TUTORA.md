# Tutora RAG: coleta, indexação e avaliação

## Objetivo

A tutora ensina como investigar uma alegação e apresenta um veredito somente
quando encontra evidências suficientes. Materiais educativos orientam o método;
documentos do tipo `evidencia` sustentam afirmações factuais. Exemplos de
resposta e rótulos de avaliação ficam separados do índice.

## Instalação

Use Python 3.11 ou 3.12. O projeto mantém as dependências do protótipo RAG em
`requirements-rag.txt` para evitar instalar as ferramentas de fine-tuning.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-rag.txt
cp .env.example .env
```

Para gerar respostas, instale e execute o Ollama com um modelo local:

```bash
ollama pull qwen2.5:7b
```

O modelo é configurado por `OLLAMA_MODEL`; a API usa
`http://localhost:11434/api/chat` por padrão e pode ser alterada por
`OLLAMA_URL`. Na primeira indexação, o modelo de embeddings
`intfloat/multilingual-e5-small` será baixado pelo Sentence Transformers.

## Fontes

Edite `sources.json` para adicionar documentos com `path` local ou `url`
HTTP(S). Também é possível adicionar feeds RSS em
`feeds`; cada item publicado no feed vira uma URL candidata. Só configure
fontes cujo acesso automatizado e indexação sejam permitidos.

Cada entrada exige `kind` (`evidencia` ou `educativo`), `source` e
`origin_group`. O último identifica a origem factual comum: reproduções da
mesma apuração devem compartilhar o mesmo grupo. Registre `license`,
`published_at` e, se necessário, `title`. O arquivo de exemplo usa somente
dois guias próprios e não coleta sites externos. Para URLs e feeds remotos,
confirme as condições de uso e defina `usage_approved: true`; sem isso o
coletor recusa o download.

O `sources.json` atual contém dois guias próprios e quatro fichas factuais
locais baseadas em divulgações da Agência de Notícias IBGE. Cada ficha usa
`path` para o texto indexado e `reference_url` para o link oficial mostrado
na resposta. A [licença da Agência de Notícias
IBGE](https://agenciadenoticias.ibge.gov.br/agencia-sala-de-imprensa/2080-agencia-de-noticias/an-artigos-diversos/9277-licenca.html)
permite reprodução com atribuição. Os quatro documentos compartilham
`origin_group: "ibge"`: são dados da mesma instituição, não confirmações
independentes. Atualize as fichas quando os números ou períodos mudarem.

```json
{
  "url": "https://exemplo.org/relatorio.pdf",
  "kind": "evidencia",
  "source": "Órgão responsável",
  "origin_group": "orgao-responsavel",
  "license": "condições de uso verificadas",
  "usage_approved": true,
  "published_at": "2026-09-20"
}
```

PDFs com texto, HTML, TXT e Markdown são aceitos. PDFs digitalizados precisam
de OCR antes da coleta. A extração automática de HTML pode exigir ajuste
específico quando o site monta o conteúdo com JavaScript ou mistura texto
principal e navegação.

Para descobrir checagens, configure `FACT_CHECK_API_KEY` em `.env` e execute
`python -m scripts.descobrir_checagens "tema"`. O comando salva URLs candidatas
em `data/candidatas_checagens.jsonl`; ele não indexa o conteúdo dessas páginas.
Revise a fonte, a licença e a origem das evidências antes de adicioná-las ao
arquivo de fontes. A avaliação de um veículo não substitui uma checagem
independente da alegação.

## Execução

Abra o menu interativo na raiz do repositório com `python main.py`. Ele mostra
o estado local e reúne coleta, indexação, conversa, descoberta de checagens e
avaliação. Após a instalação, a sequência inicial é usar a opção 1 com
`sources.json`, a opção 2 para indexar e a opção 3 para conversar.

Os comandos individuais continuam disponíveis na raiz do repositório:

```bash
python -m scripts.coletar_rag --config sources.json
python -m scripts.indexar_rag
python -m scripts.perguntar_rag "Como posso verificar esta notícia?"
```

O coletor grava `data/documents.jsonl`, com URL, título, origem, data de coleta,
licença e hash do conteúdo. Também preserva em `data/raw/` uma cópia de cada
arquivo baixado, identificada pelo hash dos bytes originais. A indexação guarda vetores no
`vector_store/chroma/`, em coleções separadas. Esses diretórios são locais e
ignorados pelo Git. Reexecutar coleta e indexação atualiza documentos
alterados e pula documentos iguais. A coleta não apaga automaticamente itens
removidos do arquivo de fontes. Para retirar uma fonte, remova a linha
correspondente de `data/documents.jsonl` e execute a indexação novamente;
os trechos ausentes do corpus serão eliminados do índice.

Uma conclusão `VERDADEIRA`, `FALSA` ou `ENGANOSA` exige duas origens
independentes citadas. O código valida IDs de citação, formato e diversidade de
origens declarada. Ele não comprova automaticamente que o trecho realmente
sustenta a conclusão; essa avaliação exige revisão humana.

## Dataset próprio de avaliação

Crie um JSONL separado com perguntas revisadas e URLs que deveriam aparecer
na recuperação. Guarde rótulos e justificativas revisados no mesmo conjunto,
sem indexá-los. Exemplo de linha:

```json
{"pergunta":"Qual é o dado oficial sobre o tema?","urls_esperadas":["https://exemplo.org/relatorio.pdf"],"veredito_revisado":"INCONCLUSIVA","justificativa_revisada":"O documento não cobre a alegação completa."}
```

Para medir busca:

```bash
python -m scripts.avaliar_recuperacao_rag data/avaliacao/casos.jsonl --k 5
```

O comando informa a posição da primeira URL esperada, `hit@5` e MRR.
Reserve casos de períodos e temas diferentes; revise rótulos e evidências por
pessoas antes de usar o conjunto para comparar versões da tutora.

## Avaliar a tutora no MLflow

O menu oferece a opção 7 para avaliar respostas, a opção 8 para o benchmark
AVeriTeC e a opção 9 para abrir o MLflow. Inicie com
`.venv/bin/python main.py`; `--listar` mostra as opções e `--opcao 9` abre
diretamente o painel. Os comandos individuais continuam disponíveis:

```bash
python -m scripts.avaliar_rag_mlflow data/avaliacao/casos.jsonl --run-name minha-primeira-avaliacao
.venv/bin/python -m mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"
```

Abra `http://127.0.0.1:5000` no navegador. Cada execução aparece no experimento
`NexoVerdict-RAG` (ou no nome definido por `MLFLOW_EXPERIMENT_NAME`) com parâmetros do modelo e da busca, hashes do conjunto de
casos, corpus e prompt, revisão do Git, métricas agregadas e o artefato
`avaliacao/relatorio.json`. Esse relatório guarda pergunta, URLs recuperadas,
posição da primeira URL esperada, resposta, fontes citadas, latência e erro de
cada caso. Use apenas casos que possam ser armazenados localmente no MLflow.

O arquivo JSONL aceita `id`, `pergunta`, `urls_esperadas`,
`veredito_revisado` e `justificativa_revisada`. Os dois últimos campos devem
ser definidos por revisão humana com fontes identificadas. Quando não há
`urls_esperadas`, o MLflow omite `hit_at_6` e MRR; quando não há
`veredito_revisado`, omite a acurácia. A avaliação mede:

- `taxa_respostas_validas`: a consulta terminou com JSON e citações válidos;
- `hit_at_6` e `mrr`: recuperação das URLs esperadas entre as evidências;
- `acuracia_veredito` e `taxa_abstencao_correta`: comparação com rótulos revisados;
- `latencia_media_s`, `documentos_citados_media` e
  `evidencias_citadas_media`: tempo e quantidade de documentos citados,
  distinguindo evidências de guias educativos.

Essas métricas não provam que uma citação sustenta uma afirmação nem que a
explicação ensina bem. Revise manualmente amostras do relatório, incluindo
casos errados, e registre erros de suporte factual, clareza didática e falsas
equivalências antes de comparar versões.

Um caso ilustrativo em `docs/casos_avaliacao_exemplo.jsonl` usa o boato do
"clone de Lula" e duas checagens publicadas. Ele permite demonstrar a
avaliação, mas a base inicial contém apenas guias educativos. Assim, uma
execução inicial deve expor a ausência dessas evidências e a divergência do
veredito, sem ser interpretada como medida geral de qualidade. Antes de
indexar qualquer conteúdo externo, revise as condições de uso e registre a
origem da apuração. Monte um conjunto maior, variado e revisado para medir
desempenho real.

## Benchmark reproduzível com AVeriTeC PT-BR

Veja [como o benchmark foi feito](BENCHMARK_RAG_COMO_FOI_FEITO.md) e a
[análise da execução concluída](BENCHMARK_RAG_RESULTADOS.md).

Para um teste com mais casos, execute na raiz do projeto:

```bash
python -m scripts.testar_benchmark_averitec --per-class 10
.venv/bin/python -m mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"
```

O comando baixa um snapshot verificado por SHA-256 do [AVeriTeC traduzido
para PT-BR](https://github.com/lucasfrag/automated-fact-checking-in-pt-br/tree/main/averitec/pt-br/dataset)
e cria 30 casos: 10 `FALSA`, 10 `VERDADEIRA` e 10 `INCONCLUSIVA`.
Dados e índice ficam isolados em `data/benchmark_averitec/`; a base normal
`data/documents.jsonl` não é alterada. Só as perguntas e respostas de
evidência, com URLs, entram no índice. Vereditos e justificativas ficam
somente no arquivo de avaliação. O manifesto registra snapshot, seleção,
quantidades e exclusões. Use `--seed` e `--per-class` para mudar a amostra;
`--prepare-only` prepara sem executar o Qwen.

O painel mostra `hit_at_6`, MRR, acurácia, `f1_macro`, recall por classe,
abstenção correta, respostas válidas, evidências citadas e latência. O
artefato por caso permite revisar os erros. Este é um teste de RAG com
evidências previamente anotadas: mede se o sistema as encontra e usa, não
mede a descoberta autônoma de fontes na web. O conjunto de desenvolvimento
tem alegações antigas, foi traduzido automaticamente e sua licença original
é [CC BY-NC 4.0](https://github.com/MichSchli/AVeriTeC#license). Use os
resultados como benchmark local de pesquisa e complemente depois com casos
brasileiros recentes revisados por pessoas. `Conflicting
Evidence/Cherrypicking` fica fora da amostra porque não corresponde de forma
segura a um único veredito da tutora.
