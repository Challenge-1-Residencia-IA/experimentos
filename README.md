# Tutora RAG de educação midiática

Este repositório experimenta uma tutora que ensina a verificar informações.
Ela recupera materiais educativos e evidências, explica como conferir uma
alegação e apresenta uma análise das evidências sem emitir veredito.

Cada resposta mostra afirmação, evidências favoráveis e contrárias, contexto,
pontos de atenção, avaliação das evidências e limitações. A avaliação descreve
o estado das fontes consultadas; não classifica a alegação como verdadeira ou falsa.
O prompt e o esquema JSON têm versão `analise-evidencias-v1`. Cada consulta
registra no MLflow a pergunta, os documentos recuperados, a saída original do
modelo, a resposta validada, o modelo e os arquivos do prompt e do esquema.
Para registrar apenas a versão atual, sem consultar o modelo, execute
`.venv/bin/python -m scripts.versionar_resposta_mlflow`.

## Início rápido

Use Python 3.11 ou 3.12, Ollama e o modelo `qwen2.5:7b`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-rag.txt
cp .env.example .env
ollama pull qwen2.5:7b
python main.py
```

No menu, a opção **3** abre uma conversa contínua: digite uma alegação, uma
pergunta ou cole um link. A tutora pesquisa com `ddgs`, lê páginas com
`trafilatura`, recupera trechos pertinentes e usa o Qwen para explicar a
checagem. Digite `sair` para voltar ao menu. A busca não exige chave de API,
mas pode sofrer limites ou bloqueios impostos por sites externos.

Ao abrir a opção **3**, o programa também verifica se passou um dia desde a
última atualização da base ou se `sources.json` mudou. Nesse caso, coleta as
fontes configuradas e atualiza o índice automaticamente. O `sources.json`
inclui um feed RSS da Agência de Notícias IBGE, limitado aos 10 itens
disponíveis no feed. A pesquisa da conversa continua buscando outras fontes
atuais na web. Para forçar uma atualização imediata, use as opções **1** e
**2**. Isso atualiza os documentos e embeddings do RAG; não altera os pesos
do Qwen nem é treinamento do modelo.

O menu também permite coletar documentos, atualizar o banco, descobrir
checagens e avaliar a recuperação. Os comandos individuais continuam disponíveis:

```bash
python -m scripts.coletar_rag --config sources.json
python -m scripts.indexar_rag
python -m scripts.perguntar_rag "Como posso verificar uma notícia?"
python -m scripts.perguntar_rag "É verdade que ...?"
python -m scripts.perguntar_rag "https://exemplo.org/noticia"
python -m scripts.avaliar_rag_mlflow docs/casos_avaliacao_exemplo.jsonl
python -m scripts.testar_benchmark_averitec --per-class 10
```

O `sources.json` inicial reúne dois guias educativos e quatro fichas factuais
com links para publicações do IBGE. Use as opções 1 e 2 para coletar e indexar
novas fontes. Esse conjunto cobre apenas os temas das fichas; para outras
alegações, a pesquisa durante a conversa busca páginas atuais. As fontes
locais continuam úteis para ensinar o método e cobrir informações já revisadas.
O arquivo `web_sources.json` controla quais domínios podem contribuir para
um veredito conclusivo e quais não são coletados automaticamente.

Para acompanhar avaliações, use a opção 9 do menu ou execute
`.venv/bin/python -m mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"`
e abra `http://127.0.0.1:5000`. O exemplo de
avaliação mostra a lacuna factual da base inicial; as métricas de veredito e
recuperação só ficam úteis com casos revisados e evidências indexadas.
O benchmark AVeriTeC usa dados e índice separados da base principal; veja
[docs/RAG_TUTORA.md](docs/RAG_TUTORA.md) para o protocolo e as limitações.

O guia completo de fontes, formatos, atualização incremental e avaliação está
em [docs/RAG_TUTORA.md](docs/RAG_TUTORA.md). Dados e índice vetorial são
mantidos localmente em `data/` e `vector_store/`.

`requirements.txt` preserva as dependências dos experimentos antigos de
fine-tuning. O código do RAG usa `requirements-rag.txt`.
