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
Essa chave deve ser uma chave de API de um projeto Google Cloud com a
**Fact Check Tools API** ativada. Se a API responder HTTP 400 ou 403, o comando
mostra a mensagem devolvida pelo Google para ajudar a identificar chave
inválida, API não ativada ou restrições da chave. A opção 3 do menu, para
conversar e analisar links com pesquisa na web, não usa essa chave.
Revise a fonte, a licença e a origem das evidências antes de adicioná-las ao
arquivo de fontes. A avaliação de um veículo não substitui uma checagem
independente da alegação.

## Execução

Abra o menu interativo na raiz do repositório com `python main.py`. Ele mostra
o estado local e reúne coleta, indexação, conversa, descoberta de checagens e
avaliação. Para incluir os guias educativos locais, use a opção 1 com
`sources.json` e a opção 2 para indexar. A opção 3 já pode pesquisar na web
com o índice vazio; ela abre uma conversa contínua até digitar `sair`.
Ao abrir essa opção, o programa também coleta e indexa automaticamente as
fontes de `sources.json` se passaram 24 horas desde a última atualização ou
se a configuração mudou. Se a coleta falhar, a conversa continua com as
fontes disponíveis e a próxima abertura tenta novamente. Essa atualização
acontece enquanto o menu é usado; não há um serviço rodando com o programa
fechado. O arquivo `data/rag_auto_refresh.json` guarda o horário da última
atualização concluída. As opções 1 e 2 continuam disponíveis para uma
atualização imediata.

O `sources.json` inclui o [feed RSS da Agência de Notícias IBGE](https://agenciadenoticias.ibge.gov.br/agencia-rss),
com até 10 publicações por atualização. A [licença da agência](https://agenciadenoticias.ibge.gov.br/agencia-sala-de-imprensa/2080-agencia-de-noticias/an-artigos-diversos/9277-licenca.html)
permite reprodução com atribuição. Todos esses textos compartilham a mesma
origem factual `ibge`, portanto não contam como confirmações independentes.
Para incluir outros feeds, verifique antes as condições de acesso e uso.
Atualizar o índice acrescenta conhecimento recuperável ao RAG; o Qwen não
aprende novos pesos com essa operação.

### Pesquisa na web e links

A conversa pesquisa com [`ddgs`](https://github.com/deedy5/ddgs) e lê cada
página permitida com [`trafilatura`](https://trafilatura.readthedocs.io/en/latest/quickstart.html).
Não é necessária uma chave de busca. O navegador local do usuário ou o Ollama
não fazem a pesquisa: o código Python seleciona e envia ao Qwen somente
trechos das páginas efetivamente lidas. Resultados que retornam apenas título
e resumo não viram evidência. PDFs textuais também são aceitos.

Ao enviar somente um link, o Qwen extrai uma afirmação factual central do
conteúdo e a pesquisa procura outras fontes. Se houver várias alegações na
página, envie também a afirmação específica que deseja verificar. A página
enviada é mostrada como origem da alegação, sem contar como prova de si mesma.
Perguntas educativas podem receber `NAO_APLICAVEL` no campo de veredito.

O arquivo `web_sources.json` lista domínios que podem contribuir para um
veredito conclusivo. A lista é uma triagem inicial, não uma certificação de
verdade. A validação ainda exige duas origens qualificadas, distintas e
citadas. Outros sites podem aparecer como contexto, mas sozinhos não permitem
um veredito conclusivo. Domínios em `blocked_automated_domains` não são
baixados; revise as condições de acesso antes de alterar essa lista. Quando
um link não puder ser lido, a tutora informa a falha ou a quantidade de
páginas ignoradas. A coleta pontual da conversa não adiciona páginas ao
índice permanente.

Na opção 3, a pesquisa faz uma segunda consulta com termos de checagem quando
a primeira não traz duas origens qualificadas. Ela combina os trechos das
páginas lidas com evidências pertinentes do índice local. Materiais educativos
continuam separados das evidências factuais. O resultado mostra quantas
páginas foram lidas, quantas origens web qualificadas foram encontradas e
quantas evidências locais entraram na análise. Se o veredito for
`INCONCLUSIVA`, a tela informa se faltaram fontes ou se os trechos não
sustentaram uma conclusão. O domínio de uma fonte é apenas uma triagem:
duas páginas que repetem a mesma apuração não devem contar como confirmação
independente, mesmo quando aparecem em veículos diferentes.

Use, por exemplo:

```bash
python main.py --opcao 3
python -m scripts.perguntar_rag "É verdade que a alegação X aconteceu?"
python -m scripts.perguntar_rag "https://exemplo.org/noticia"
```

Buscadores e publicadores externos podem limitar requisições, bloquear
leitura automática ou exigir JavaScript. O programa não oferece garantia de
busca ilimitada. Quando não obtém evidência suficiente, responde
`INCONCLUSIVA` e mostra o limite da consulta. Antes de usar vereditos como
produto público, revise manualmente pertinência dos trechos, independência
factual das fontes e datas das publicações.

Os comandos individuais continuam disponíveis na raiz do repositório:

```bash
python -m scripts.coletar_rag --config sources.json
python -m scripts.indexar_rag
python -m scripts.perguntar_rag "Como posso verificar esta notícia?"
python -m scripts.perguntar_rag --local "Pergunta sobre o índice local"
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

## Integração de Datasets Externos e Fragmentação Inteligente (Chunking)

Para expandir o corpus com bases factuais abertas e gratuitas de checagem em português:

- **FakeRecogna (Hugging Face / recogna-nlp/FakeRecogna):** 11.902 amostras com checagens de agências (Aos Fatos, Boatos.org, E-farsas, UOL Confere, AFP Checamos, Projeto Comprova) e notícias factuais de portais (G1, UOL, Ministério da Saúde). Licença MIT.
- **X-FACT (Hugging Face / utahnlp/x-fact - split PT):** Declarações reais brasileiras com evidências recuperadas via buscas no Google. Licença MIT.
- **AVeriTeC PT-BR:** Pares de perguntas e respostas de evidência com URLs e vereditos. Licença CC BY-NC 4.0.
- **Fake.br Corpus (USP):** Notícias completas verdadeiras e falsas alinhadas para treinamento e validação.

### Como executar a integração

Use o script `scripts/integrar_datasets.py` ou a opção `10` do menu interativo (`main.py`):

```bash
# Integrar 50 itens de cada dataset público e fragmentar textos grandes
.venv/bin/python scripts/integrar_datasets.py --dataset todos --limit 50 --inject-context

# Atualizar o banco vetorial com os novos trechos
.venv/bin/python scripts/indexar_rag.py
```

### Divisor Recursivo Semântico (`rag/splitter.py`)

O módulo `rag/splitter.py` implementa `RecursiveTextSplitter`:
1. Respeita a estrutura hierárquica (títulos Markdown, quebras de parágrafo, sentenças com pontuação terminal e palavras), evitando que orações ou números fiquem cortados.
2. Suporta injeção de contexto (`[Documento: <Título> | Fonte: <Origem>]`) no início de cada chunk para enriquecer a similaridade semântica dos embeddings.
3. Remove ruídos automáticos de web (botões de compartilhamento, scripts e rodapés de sites).
