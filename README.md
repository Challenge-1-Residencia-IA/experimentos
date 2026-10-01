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
python -m scripts.coletar_rag --config sources.example.json
python -m scripts.indexar_rag
python -m scripts.perguntar_rag "Como posso verificar uma notícia?"
```

O exemplo indexa dois guias educativos próprios. Para verificar fatos, inclua
documentos do tipo `evidencia` no arquivo de fontes e repita a coleta e a
indexação. Até lá, a tutora deve responder com veredito `INCONCLUSIVA`.

O guia completo de fontes, formatos, atualização incremental e avaliação está
em [docs/RAG_TUTORA.md](docs/RAG_TUTORA.md). Dados e índice vetorial são
mantidos localmente em `data/` e `vector_store/`.

`requirements.txt` preserva as dependências dos experimentos antigos de
fine-tuning. O código do RAG usa `requirements-rag.txt`.
