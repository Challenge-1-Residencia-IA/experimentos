# Como o benchmark RAG foi feito

Este documento registra a avaliação da tutora com o AVeriTeC PT-BR. A análise
da execução concluída está em [Resultados do benchmark](BENCHMARK_RAG_RESULTADOS.md).

## Objetivo e escopo

O teste verifica se o RAG local recupera evidências anotadas e se o Qwen 2.5
7B produz uma resposta educativa com veredito e fontes. **Ele não mede pesquisa
ao vivo na internet**: o índice contém respostas e URLs previamente anotadas
no conjunto AVeriTeC.

## Preparação dos dados

O script `scripts/preparar_benchmark_averitec.py` baixa o arquivo `dev.json`
traduzido para PT-BR e confere seu SHA-256 antes de usá-lo. Com a semente 42,
seleciona 10 casos de cada classe: `FALSA`, `VERDADEIRA` e `INCONCLUSIVA`.
Exclui alegações curtas ou repetidas, casos sem fontes suficientes e o rótulo
ambíguo `Conflicting Evidence/Cherrypicking`.

As perguntas e respostas anotadas que têm URL são convertidas em documentos
de evidência. Rótulos e justificativas de referência ficam fora do índice
vetorial, no arquivo de casos. O índice reúne evidências de **todos os casos
selecionados**, o que permite recuperar também trechos de outros casos.

Na execução analisada, foram gerados 30 casos e 86 documentos de evidência.
Os arquivos ficam em `data/benchmark_averitec/`:

| Arquivo | Conteúdo |
| --- | --- |
| `dev.json` | Snapshot baixado do conjunto traduzido |
| `manifest.json` | Origem, hash, semente, contagens e exclusões |
| `casos.jsonl` | Alegações, rótulos, URLs esperadas e justificativas |
| `documents.jsonl` | Perguntas e respostas anotadas usadas como evidência |
| `chroma/` | Índice vetorial separado da base da tutora |

## Execução

Na raiz do projeto, com as dependências instaladas e o Ollama servindo
`qwen2.5:7b`:

```bash
.venv/bin/python -m scripts.testar_benchmark_averitec --per-class 10 --seed 42
```

O comando prepara os casos, indexa os documentos com
`intfloat/multilingual-e5-small`, consulta o Qwen uma vez por caso e registra
a avaliação no MLflow. Para preparar os dados sem chamar o modelo:

```bash
.venv/bin/python -m scripts.testar_benchmark_averitec --per-class 10 --seed 42 --prepare-only
```

Para abrir o painel:

```bash
.venv/bin/python -m mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"
```

O relatório detalhado é salvo como artefato
`avaliacao/relatorio.json` da execução. Ele contém, para cada caso, a pergunta,
o rótulo de referência, as URLs recuperadas, a posição da primeira URL
esperada, o veredito, as fontes citadas, a latência e eventuais erros. O MLflow
guarda ainda modelo, parâmetros da busca, hashes dos casos, corpus e prompt,
e a revisão do Git. A revisão do Git não incorpora alterações locais não
commitadas; os hashes ajudam a distinguir esses arquivos.

## Como ler as métricas

- `hit_at_6`: pelo menos uma URL esperada apareceu entre as seis evidências
  recuperadas. **Não** confirma que o trecho é pertinente ou sustenta a
  conclusão.
- `mrr`: média do inverso da posição da primeira URL esperada.
- `taxa_respostas_validas`: proporção de casos sem erro de consulta ou de
  validação do formato e das referências.
- `acuracia_veredito`: proporção de acertos entre **todos** os casos rotulados;
  casos com erro contam como incorretos.
- `f1_macro` e `recall_*`: desempenho por classe, incluindo o efeito dos
  casos sem resposta válida.
- `latencia_media_s`: tempo médio de recuperação, geração e validação por caso.

O conjunto é traduzido automaticamente e traz alegações de outros países e
períodos. Use-o como comparação local, não como estimativa de desempenho com
alegações brasileiras recentes. Consulte a [origem do AVeriTeC PT-BR](https://github.com/lucasfrag/automated-fact-checking-in-pt-br/tree/main/averitec/pt-br/dataset)
e a [licença CC BY-NC 4.0](https://github.com/MichSchli/AVeriTeC#license).
