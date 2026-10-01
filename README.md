# Tutora RAG de educação midiática

Este repositório experimenta uma tutora que ensina a verificar informações.
Ela recupera materiais educativos e evidências, explica como conferir uma
alegação e só apresenta um veredito quando há suporte suficiente nas fontes.

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

O menu permite coletar documentos, atualizar o banco, conversar com a tutora,
descobrir checagens e avaliar a recuperação. Os comandos individuais também
continuam disponíveis:

```bash
python -m scripts.coletar_rag --config sources.json
python -m scripts.indexar_rag
python -m scripts.perguntar_rag "Como posso verificar uma notícia?"
python -m scripts.avaliar_rag_mlflow docs/casos_avaliacao_exemplo.jsonl
python -m scripts.testar_benchmark_averitec --per-class 10
```

O `sources.json` inicial reúne dois guias educativos e quatro fichas factuais
com links para publicações do IBGE. Use as opções 1 e 2 para coletar e indexar
novas fontes. Esse conjunto cobre apenas os temas das fichas; para outras
alegações, acrescente evidências específicas e atualizadas.

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
