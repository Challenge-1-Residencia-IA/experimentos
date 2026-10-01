---
base_model: mlx-community/Qwen2.5-7B-Instruct-4bit
library_name: mlx
pipeline_tag: text-generation
---

# NexoVerdict IA — adapter de verificação por evidências v3

Adapter LoRA experimental para o fluxo MLX-LM deste projeto. Ele recebe uma
alegação e snippets de evidência e produz uma resposta estruturada com veredito,
justificativa e índices de fontes.

## Modelo e artefatos

- Modelo-base: `mlx-community/Qwen2.5-7B-Instruct-4bit`.
- Adapter: checkpoint de menor perda de validação, iteração 50 de 100.
- Framework: MLX-LM.
- Dataset de treino: FACTCK.BR preparado localmente. Os dados não são incluídos
  neste repositório; consulte o manifesto do código para hashes dos splits.
- Parâmetros, hashes e resultado completo: `models/adapters-evidencias-v3.json`
  no repositório de código.

## Resultado conhecido

No teste reservado local de 81 exemplos, obteve F1 macro `0.301`, acurácia
`0.716` e taxa de abstinência correta `0.0`. O conjunto é desbalanceado e contém
somente dois casos `INCONCLUSIVA`. O resultado não demonstra confiabilidade nas
classes minoritárias; o adapter não atingiu os critérios do protótipo.

As citações válidas e o JSON bem formado medem o formato, não a correção factual
nem se a fonte realmente sustenta o veredito. Use apenas como experimento e
mantenha revisão humana.

## Carregamento neste projeto

Baixe esta revisão no diretório local e passe o caminho do adapter ao avaliador:

```bash
python scripts/avaliar_evidencias.py \
  --adapters-dir models/adapters-evidencias-v3/best
```

O verificador de links exige por padrão duas fontes de domínios distintos. Esse
critério operacional é mais estrito que o contrato histórico dos exemplos de
treino, que contém uma fonte por amostra.
