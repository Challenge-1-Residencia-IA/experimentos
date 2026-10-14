# Executar o NexoVerdict

Requisitos: Git, Python 3.11 ou 3.12, Ollama e uma máquina capaz de executar
`qwen2.5:7b`. Os comandos abaixo são para PowerShell, a partir da raiz do
projeto. A descrição do sistema está em [DOCUMENTACAO.md](DOCUMENTACAO.md).

## 1. Instalar e configurar

```powershell
git clone git@github.com:Challenge-1-Residencia-IA/experimentos.git
cd experimentos
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-rag.txt -r bot-test/requirements.txt
python -m playwright install chromium
Copy-Item .env.example .env
Copy-Item bot-test/.env.example bot-test/.env
ollama pull qwen2.5:7b
```

Se já tiver `.env`, preserve-o em vez de copiar por cima. No Linux/macOS,
ative com `source .venv/bin/activate` e use `cp` no lugar de `Copy-Item`.

Na raiz, configure `.env`:

```dotenv
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_URL=http://localhost:11434/api/chat
MLFLOW_EXPERIMENT_NAME=NexoVerdict-RAG
```

Em `bot-test/.env`, configure um contato de teste com nome único:

```dotenv
RAG_BRIDGE_URL=http://127.0.0.1:8765
RAG_TIMEOUT_SECONDS=200
BOT_TARGET_CHAT=Nome exato do contato
BOT_TARGET_CHATS=[]
BOT_ENABLE_SEND=false
BOT_ALLOWED_CONTACTS=[]
BOT_QUEUE_SIZE=20
```

## 2. Preparar corpus e índice

Se já possui `data/documents.jsonl` com evidências e materiais educativos,
execute apenas a indexação ao final desta seção. Caso contrário, prepare os
guias e o corpus factual. Para recuperar os datasets do armazenamento DVC:

```powershell
python -m pip install pandas pyarrow scikit-learn "dvc[http]"
dvc remote modify --local dagshub user SEU_USUARIO
dvc remote modify --local dagshub password SEU_TOKEN
dvc pull data/raw/Fake.br-Corpus.dvc data/raw/FakeTrue.Br.dvc
python -m scripts.coletar_rag --config sources.json
python -m dados.preparar
python -m scripts.converter_dataset_colega_rag
python -m scripts.indexar_rag
```

O acesso DVC depende de permissão no armazenamento configurado em `.dvc/config`.
Alternativamente, disponibilize os dados autorizados nos caminhos
`data/raw/Fake.br-Corpus/` e `data/raw/FakeTrue.Br/FakeTrueBr_corpus.csv`,
e execute os quatro comandos Python acima. Outra opção é cadastrar suas fontes
factuais autorizadas em `sources.json`, com `kind: "evidencia"`, junto dos
materiais `educativo`, e executar coleta e indexação.

A ponte exige `evidence_chunks` e `educational_chunks` maiores que zero.
Os guias educativos sozinhos não bastam. A primeira indexação precisa de
internet para baixar os embeddings; a ponte depois usa esse cache em modo offline.
Atualizou o corpus? Execute novamente `python -m scripts.indexar_rag`.

## 3. Iniciar o sistema completo

Abra três terminais na raiz do projeto. Ative `.venv` nos terminais Python.

Terminal 1, caso o aplicativo Ollama ainda não esteja atendendo:

```powershell
ollama serve
```

Terminal 2:

```powershell
.\.venv\Scripts\Activate.ps1
python -m scripts.rag_bridge --port 8765 --timeout 180
```

Terminal 3:

```powershell
.\.venv\Scripts\Activate.ps1
Invoke-RestMethod http://127.0.0.1:8765/health
python bot-test/botfinal.py
```

Só execute o bot quando `/health` indicar `ready: true`. A ponte pode levar
até 120 segundos para inicializar. Se retornar 503, espere e consulte novamente;
se permanecer indisponível, confira Ollama, nome do modelo, cache e as duas
coleções do índice antes de reiniciar a ponte.

Escaneie o QR no Chromium. Mantenha o contato visível na lateral ou abra-o
manualmente. Aguarde `histórico inicial ignorado; pronto para mensagem nova`,
depois envie uma mensagem nova de outra conta. A prévia aparece no console;
`BOT_ENABLE_SEND=false` impede envio. O número `alvo` segue a ordem da lista
configurada; para um único `BOT_TARGET_CHAT`, é 1.

## 4. Habilitar envio após validar as prévias

Em `bot-test/.env`:

```dotenv
BOT_ENABLE_SEND=true
BOT_ALLOWED_CONTACTS=["Nome exato do contato"]
```

Reinicie o bot e envie uma mensagem nova. Para vários contatos, configure
`BOT_TARGET_CHATS=["Contato A","Contato B"]` e autorize todos em
`BOT_ALLOWED_CONTACTS`. Use nomes únicos e mantenha os contatos visíveis.
Não navegue manualmente no Chromium durante a operação. Não execute duas
instâncias usando o mesmo perfil ou banco.

## 5. Encerrar ou reiniciar

Use Ctrl+C primeiro no bot, depois na ponte. Pare o Ollama se necessário.
Respostas pendentes não são restauradas para envio no reinício. Após timeout da
ponte, reinicie ponte e bot em dry-run. Se um envio ficar sem confirmação,
confira o WhatsApp antes de agir; não há reenvio automático. Preserve
`bot-test/data/deliveries.db`, pois ele impede duplicatas.

## Usar pelo terminal, sem WhatsApp

Com corpus, índice e Ollama preparados:

```powershell
python main.py
python -m scripts.perguntar_rag "Pergunta ou alegação pública"
python -m scripts.perguntar_rag --local "Pergunta usando somente o índice"
```

O menu oferece as ferramentas adicionais. Para consultar os registros locais:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abra `http://127.0.0.1:5000`. Consultas da ponte WhatsApp não são registradas no
MLflow. A integração reconstruída ainda depende de validação no PC do Qwen.
