# Versionamento no DagsHub

Este projeto usa o MLflow hospedado pelo DagsHub para compartilhar versões do
RAG, métricas e respostas. O Qwen roda no Ollama da máquina que faz a consulta.
A versão registrada guarda código, prompt, esquema JSON, política de fontes,
índice Chroma, corpus, dependências e hashes. Os pesos do Qwen não são enviados:
o pacote exige o mesmo digest do modelo instalado no Ollama.

## Preparar o repositório da equipe

Crie um repositório no DagsHub ou [conecte o repositório GitHub existente](https://dagshub.com/docs/integration_guide/github/).
Adicione os colegas como colaboradores. O endpoint MLflow é
`https://dagshub.com/ORGANIZACAO/REPOSITORIO.mlflow`.

No `.env` ignorado pelo Git, defina:

```dotenv
DAGSHUB_REPO=ORGANIZACAO/REPOSITORIO
MLFLOW_EXPERIMENT_NAME=NexoVerdict-RAG
MLFLOW_TRACKING_USERNAME=SEU_USUARIO_DAGSHUB
MLFLOW_TRACKING_PASSWORD=SEU_TOKEN_DAGSHUB
```

Cada integrante deve usar sua própria credencial. Não inclua tokens em commits,
comandos compartilhados ou mensagens. O DagsHub também permite configurar a
autenticação com `dagshub.init`; este projeto usa diretamente as variáveis de
autenticação [documentadas para MLflow](https://dagshub.com/docs/integration_guide/mlflow_tracking/).

## Teste temporário no MLflow local

Para experimentar sem enviar a consulta ao DagsHub:

```sh
.venv/bin/python -m scripts.perguntar_rag \
  "É verdade que esta alegação aconteceu?" \
  --local --mlflow-local /tmp/nexoverdict-mlflow-teste
```

O diretório informado recebe `mlflow.db` e `artifacts/`. Para inspecionar:

```sh
.venv/bin/python -m mlflow ui \
  --backend-store-uri sqlite:////tmp/nexoverdict-mlflow-teste/mlflow.db \
  --port 5000
```

Abra `http://127.0.0.1:5000`. Remover esse diretório descarta apenas o teste.
Sem `--mlflow-local`, as consultas são registradas automaticamente no
`mlflow.db` da raiz do projeto. O DagsHub é usado somente pelas ações de
publicação e pelo carregamento de versões do Model Registry.

## Publicar e consultar versões

```sh
.venv/bin/python -m scripts.versionar_resposta_mlflow
.venv/bin/python -m scripts.registrar_modelo_rag_mlflow
.venv/bin/python -m scripts.publicar_dataset_rag_mlflow
```

O primeiro comando cria uma execução `versao-analise-evidencias-v1` no experimento da
equipe. O `contract_sha256` diferencia alterações no prompt, esquema ou
validador. A execução também guarda os arquivos da versão, o commit Git, o
modelo Ollama, o modelo de embeddings e o hash do corpus presente na máquina.

O segundo comando cria uma nova versão de `NexoVerdict-RAG-Analise-Evidencias`
no **Model Registry** e uma execução de origem. Ele exige o corpus e o índice
já criados, o Ollama ativo e `qwen2.5:7b` instalado. A versão fixa o digest do
Qwen, a revisão do `intfloat/multilingual-e5-small`, hashes do índice e corpus,
além do código executável. O repositório DagsHub deve continuar privado: o
corpus reúne materiais com licenças MIT, Academic Use e CC BY-NC 4.0. Respeite
as condições de cada fonte antes de compartilhar os artefatos fora da equipe.

O Qwen e o modelo de embeddings executam localmente. No computador atual, os
pesos do Qwen ficam em `~/.ollama/models/blobs/`; os embeddings ficam no cache
`~/.cache/huggingface/hub/`; `data/documents.jsonl` e `vector_store/chroma/`
guardam a base local. Em outra máquina, instale Ollama, execute
`ollama pull qwen2.5:7b`, instale `requirements-rag.txt` e configure `.env`.
O pacote recusa a execução se o digest do Qwen instalado for diferente do
registrado.

O terceiro comando publica `rag-datasets-v1` como uma execução separada. Ela
contém o corpus principal, os arquivos do benchmark AVeriTeC e um manifesto
com hashes, tamanhos, quantidades e licenças. Não publica HTML bruto, caches,
credenciais ou arquivos temporários. Mantenha o repositório privado por causa
das condições Academic Use e CC BY-NC 4.0 presentes no conjunto.

Para carregar a versão 1 registrada e consultar o fluxo com pesquisa web:

```sh
.venv/bin/python -m scripts.testar_modelo_dagshub \
  "É verdade que esta alegação aconteceu?" --version 1 --modo web
```

O comando baixa a versão registrada, confirma o digest do Qwen instalado e
executa o pacote. Para um teste sem pesquisa atual na internet, troque por
`--modo local`.

O mesmo teste também pode ser feito diretamente em Python:

```python
import mlflow
import pandas as pd
from rag.dagshub_tracking import configure_tracking

configure_tracking()
model = mlflow.pyfunc.load_model("models:/NexoVerdict-RAG-Analise-Evidencias/1")
resposta_json = model.predict(pd.DataFrame({"pergunta": ["Sua alegação aqui"]}))[0]
```

Use a coluna opcional `modo="local"` para consultar somente a base indexada.
No modo padrão, a busca web depende de disponibilidade das fontes. A resposta
é uma string JSON validada; alegações sem suporte podem resultar em avaliação
inconclusiva, e saídas que violam a validação geram erro.

Para avaliar e vincular métricas a uma versão com contrato e corpus iguais:

```sh
.venv/bin/python -m scripts.avaliar_rag_mlflow docs/casos_avaliacao_exemplo.jsonl
```

O relatório fica em `avaliacao/relatorio.json` na execução, com métricas como
taxa de respostas válidas, latência, `hit_at_6` e `mrr`. Avaliações com um só
caso servem apenas para conferir o fluxo; não medem qualidade geral. O vínculo
ao registro aparece nas tags `registered_model_name` e
`registered_model_version` quando os hashes conferem.

Consultas e avaliações usam o MLflow local e registram o mesmo hash. A opção
**9** abre a interface local; a opção **14** mostra a URL do DagsHub, onde
ficam versões explicitamente publicadas. O DagsHub escolhe o armazenamento
remoto desses artefatos.

Os registros já presentes em `mlflow.db` não são migrados automaticamente.
No plano gratuito, o DagsHub limita execuções rastreadas em projetos privados;
como cada consulta gera uma execução, confira a [página de preços](https://dagshub.com/pricing)
antes de usar esse modo em alto volume.

Para sincronizar os runs locais já existentes sem repetir as chamadas ao Qwen:

```sh
.venv/bin/python -m scripts.sincronizar_runs_mlflow_dagshub --dry-run
.venv/bin/python -m scripts.sincronizar_runs_mlflow_dagshub
```

O sincronizador copia métricas, histórico das métricas, parâmetros, tags e um
resumo rastreável com o ID original. Ele não copia os artefatos locais, pois os
relatórios e rastreios podem conter trechos do FakeTrue.Br cuja licença ainda
precisa ser confirmada. Runs já sincronizados são reconhecidos pela tag
`source_local_run_id` e não são duplicados.
