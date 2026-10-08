# Banco vetorial compartilhado no Neon

O RAG usa Chroma local por padrão. Neon é opcional e guarda os trechos e
embeddings em PostgreSQL com a extensão `pgvector`. O MLflow/DagsHub continua
separado: configurar Neon não envia nem sincroniza experimentos.

## Primeira configuração

1. Instale as dependências atualizadas com
   `python -m pip install -r requirements-rag.txt`.
2. Copie `.env.example` para `.env` se ainda não existir. No **`.env` local**,
   configure `VECTOR_BACKEND=neon`, `NEON_DATABASE_URL` com a URL completa
   fornecida pelo Neon e, opcionalmente, `NEON_VECTOR_NAMESPACE=nexoverdict-rag`.
   Se a rede bloquear a porta 5432, defina `NEON_TRANSPORT=https`; `tcp` é o
   padrão e continua usando `psycopg`. O transporte HTTPS usa o pacote
   comunitário `neon-serverless==0.1.0`, não um driver Python oficial do Neon.
   Não coloque a URL com senha em mensagens, commits ou `.env.example`.
3. No computador que possui `data/documents.jsonl`, execute
   `python main.py --opcao 2`. Esse comando habilita `pgvector` se necessário,
   cria a tabela própria `nexoverdict_rag_chunks` e indexa o corpus. A carga
   envia texto dos trechos, metadados e embeddings para o Neon; confirme que
   esse é o banco e o corpus que deseja usar antes de executar.
4. Em cada computador que só fará consultas, configure o mesmo backend, URL e
   namespace no `.env` local, instale dependências e Ollama, e use
   `python main.py --opcao 3`. Não é preciso copiar `vector_store/chroma/`.

O modelo de embeddings continua sendo `intfloat/multilingual-e5-small` (384
dimensões). Os prefixos `passage:` e `query:` e a distância de cosseno são
mantidos. As categorias `evidencia` e `educativo` permanecem separadas.

## Atualizações e segurança

Com Neon, uma conversa **não** reindexa automaticamente a base compartilhada.
Escolha um computador responsável por coletar fontes (opção 1) e atualizar o
índice (opção 2). A indexação é incremental e cada documento é atualizado em
uma transação; se houver falha, basta executar novamente para retomar. Ela
não apaga documentos remotos que não estejam no corpus local, evitando perda
de dados quando outra máquina usa um corpus menor. Documentos removidos do
corpus precisam de revisão e limpeza
administrativa separada.

O namespace isola este RAG de outras aplicações no mesmo banco. Se a conexão
falhar, o sistema informa o erro em vez de consultar silenciosamente um índice
Chroma desatualizado. O `.env` é ignorado pelo Git.

Os transportes TCP e HTTPS usam a **mesma tabela e namespace**. O HTTPS envia
consultas SQL parametrizadas para o endpoint SQL-over-HTTP do Neon. Na
indexação, exclusão e inserções de cada documento são enviadas em uma única
transação HTTP; uma falha não deve deixar aquele documento parcialmente
atualizado. O HTTPS evita a porta 5432, mas requer que a rede permita HTTPS
até o endpoint Neon. Se o acesso ao serviço for proibido pela política da
rede, confirme o uso com a administração da rede.

A tabela `analises`, criada por outra integração, é independente do índice
vetorial `nexoverdict_rag_chunks`. Este fluxo não grava histórico nessa tabela
nem transforma textos do Fake.br em evidência factual.

O registro de modelo no DagsHub ainda empacota um índice Chroma local e não
converte automaticamente a versão registrada para Neon. Consulte o RAG em
Neon pela opção 3 ou por `rag.tutor.ask_online()` com `VECTOR_BACKEND=neon`.
