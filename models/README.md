# Versionamento de modelos e adapters

Os pesos não são armazenados neste repositório Git. O Git guarda este guia e as
referências necessárias para localizar e reproduzir cada versão; os arquivos
pesados ficam em um repositório de modelos, como o Hugging Face Hub.

Os adapters atuais foram publicados no repositório privado
`Filipelac/sift-ai-adapters-evidencias`, na revisão
`5bae4349300e2c4c20de12b2a7b0081e47ca45c4` e na tag
`adapters-2026-09-25`. A estrutura remota é:

```text
v2-best/   melhor checkpoint do v2
v3/        adapter final do v3 e checkpoints 25, 50, 75 e 100
v3-best/   melhor checkpoint do v3, correspondente à iteração 50
```

## Publicação recomendada

Publique cada versão final do adapter em um repositório separado ou em uma
revisão identificável do mesmo repositório. Use tags ou revisões imutáveis, por
exemplo `v3`, e registre o identificador da revisão (commit SHA) no manifesto.
Evite depender de `main` ou de uma tag que possa ser movida.

Envie somente o adapter selecionado para uso, normalmente o diretório `best/`
produzido pelo treino. O arquivo principal esperado pelos scripts deste projeto
é `adapters.safetensors`. Inclua também os pequenos arquivos de configuração e
tokenizer que a versão do MLX-LM gerar e que forem necessários para carregar o
adapter. Não envie checkpoints intermediários, caches do Hugging Face, datasets
ou credenciais, a menos que haja uma necessidade explícita e autorização para
compartilhá-los.

Na página do repositório do modelo no Hub, anote:

- identificador do modelo-base e sua revisão;
- revisão exata do adapter;
- versão do MLX-LM e parâmetros relevantes do treino;
- identificador e versão do dataset usado;
- caminho do relatório de avaliação correspondente;
- SHA-256 dos arquivos publicados.

O projeto já registra vários parâmetros no MLflow e salva um
`training_summary.json` junto ao adapter. Preserve esse resumo com a versão
publicada e mantenha o relatório de avaliação em `results/`.

## Manifesto local

Para cada adapter publicado, mantenha um manifesto versionado no Git. O arquivo
`models/adapters-evidencias-v3.json` registra o adapter local e seus hashes. O
repositório e a revisão no Hub ficam como pendentes até que o login e a
publicação sejam concluídos:

```json
{
  "nome": "adapters-evidencias-v3",
  "hub": {
    "repo_id": null,
    "revisao": null,
    "visibilidade": "private",
    "status_publicacao": "pendente_de_autenticacao"
  },
  "modelo_base": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "revisao_modelo_base": null,
  "framework": "mlx-lm",
  "dataset": "FACTCK.BR",
  "relatorio_avaliacao": "results/metricas_evidencias_DATA_UTC.json",
  "sha256": {
    "adapters.safetensors": "48de45a6f4cf029a3c32e3a28619775d9b010e9288e885990c4de8cc6d355027",
    "adapter_config.json": "a59fb282ea15a3c64deab9561f5059adfbb28e384cae02e9218bda421b26f242"
  },
  "sha256_dataset_local": {
    "train.jsonl": "da58f4ae89f579f2486daa26a28eae52668bf9c954b89218a2da0da2f1febe5c",
    "valid.jsonl": "047933ec105c4c4bbb01fb63e5aa5097648a2b1c1b795c83f2173baea04f079d",
    "test.jsonl": "4b99366fa0206f482d5c733747263f556f76a36c6aaa2630a0f86a0d64b2ec2e",
    "metadata.json": "be9b4810334ed2aff87e13b87b37f9e5ad5e39e64fcc7c33f8faa7f569233d69"
  }
}
```

Não registre um manifesto com valores de exemplo como se fossem dados reais.
O SHA-256 do adapter v3 foi calculado sobre os arquivos locais em `best/`; confira
os hashes após o upload e o download. A revisão do modelo-base ainda não está
fixada nos dados do treino, então o manifesto a deixa explicitamente como
desconhecida. Para adapters compostos por vários arquivos, registre o hash de
cada arquivo relevante.

## Recuperar um adapter

Depois de publicar, autentique o CLI, crie um repositório privado e envie o
adapter final. Use o identificador da sua conta no lugar de `USUARIO`:

```bash
hf auth login
hf auth whoami
hf repos create USUARIO/adapters-evidencias-v3 --repo-type model --private
hf upload USUARIO/adapters-evidencias-v3 models/adapters-evidencias-v3/best . --repo-type model
hf upload USUARIO/adapters-evidencias-v3 models/adapters-evidencias-v3-card.md README.md --repo-type model
hf repos tag create USUARIO/adapters-evidencias-v3 v3
```

Registre no manifesto o identificador retornado, o commit exato e o hash dos
arquivos enviados. Para recuperar uma revisão fixada, use `hf download` e a
revisão registrada:

```bash
hf download USUARIO/adapters-evidencias-v3 \
  --revision REVISAO_FIXADA \
  --local-dir models/adapters-evidencias-v3/best
```

Para recuperar a publicação atual em sua estrutura local:

```bash
hf download Filipelac/sift-ai-adapters-evidencias \
  --revision adapters-2026-09-25 \
  --include 'v2-best/**' \
  --local-dir models/huggingface

hf download Filipelac/sift-ai-adapters-evidencias \
  --revision adapters-2026-09-25 \
  --include 'v3/**' \
  --include 'v3-best/**' \
  --local-dir models/huggingface
```

Os scripts aceitam `--adapters-dir models/adapters-evidencias-v3`; se os pesos
estiverem na subpasta `best`, passe o caminho diretamente:

```bash
python scripts/avaliar_evidencias.py \
  --adapters-dir models/adapters-evidencias-v3/best
```

Para um repositório privado, configure `HF_TOKEN` localmente em `.env` ou no
ambiente de execução. Nunca inclua o token neste repositório.

## Reproduzir o treino

Além dos pesos publicados, registre os parâmetros de treino, a revisão do
modelo-base, a versão do dataset e a revisão do código usada. O identificador
do modelo-base sozinho não fixa necessariamente os mesmos arquivos ao longo do
tempo; prefira também registrar sua revisão exata. O conjunto de teste deve
continuar reservado para avaliação final, conforme as instruções do README
principal.

Os diretórios locais `models/`, os adapters e os checkpoints permanecem
ignorados pelo Git. Este README e os manifestos pequenos são exceções para que
as referências de versão possam ser compartilhadas sem adicionar os pesos ao
histórico do código.
