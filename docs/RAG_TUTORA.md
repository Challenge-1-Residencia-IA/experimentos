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

Copie `sources.example.json` para `sources.json` e adicione documentos com
`path` local ou `url` HTTP(S). Também é possível adicionar feeds RSS em
`feeds`; cada item publicado no feed vira uma URL candidata. Só configure
fontes cujo acesso automatizado e indexação sejam permitidos.

Cada entrada exige `kind` (`evidencia` ou `educativo`), `source` e
`origin_group`. O último identifica a origem factual comum: reproduções da
mesma apuração devem compartilhar o mesmo grupo. Registre `license`,
`published_at` e, se necessário, `title`. O arquivo de exemplo usa somente
dois guias próprios e não coleta sites externos. Para URLs e feeds remotos,
confirme as condições de uso e defina `usage_approved: true`; sem isso o
coletor recusa o download.

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
`sources.example.json`, a opção 2 para indexar e a opção 3 para conversar.

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
