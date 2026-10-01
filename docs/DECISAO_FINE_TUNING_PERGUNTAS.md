# Decisão: fine-tuning de perguntas pausado

## Decisão atual

O fine-tuning para perguntas com evidências foi interrompido e não será
promovido para uso no NexoVerdict IA. O custo de execução no hardware local foi
alto para o resultado obtido, e não há um adapter de perguntas validado nesta
versão do projeto.

O fluxo de perguntas continua disponível com o modelo-base, busca de fontes,
prompt externo, validação de JSON, citações e registro no MLflow. Um adapter de
perguntas só deve ser retomado depois de uma revisão do dataset e de critérios
de avaliação mais fortes.

## O que foi tentado

1. Um smoke test de cinco iterações em `data/perguntas-mescladas-v2`.
   Ele produziu pesos, mas a avaliação em cinco exemplos retornou 0% de JSON
   válido e saídas compostas por `!`; portanto não foi considerado release.
2. Um treino de 300 iterações, salvo localmente sob o nome incorreto
   `adapters-evidencias-v4`, embora tivesse sido iniciado com um dataset de
   perguntas.
3. A configuração usava `--mask-prompt`, sequência máxima de 1024 e validação
   em 25 lotes. A primeira validação já retornou `nan`; as perdas de treino e
   validação seguintes permaneceram `nan`.

Esses pesos e checkpoints locais não devem ser usados, avaliados como modelo
final, publicados ou retomados.

## Causa identificada

No dataset `perguntas-mescladas-v2`, muitos prompts ocupavam todo o limite de
1024 tokens antes de a resposta do assistant começar. Com `--mask-prompt`,
somente a resposta contribui para a perda. Quando ela era totalmente truncada,
o lote ficava sem tokens-alvo e a perda se tornava `NaN`.

A auditoria com o tokenizer do Qwen encontrou, no conjunto de treino v2:

| Limite | Prompts que já atingiam o limite |
| ---: | ---: |
| 1024 | 960 de 5.522 |
| 1536 | 132 de 5.522 |
| 2048 | 10 de 5.522 |

Além disso, `--val-batches -1` fazia o MLX avaliar o conjunto inteiro antes da
primeira iteração e deixava o painel aparentemente parado por muitos minutos.

## Correções feitas no código e nos dados

- O preparador agora recorta o contexto público ao redor da resposta e reduz o
  tamanho de evidências distratoras.
- Foi gerado `data/perguntas-mescladas-v3`, sem substituir os datasets
  anteriores.
- Nesse conjunto v3, nenhum prompt ultrapassa 1536 tokens antes da resposta.
- O treinador usa 25 lotes de validação como padrão, em vez de `-1`.
- O painel informa que a validação inicial ocorre antes da barra de treino.
- A saída bruta do MLX é salva em `training_output.log`.
- O treinador interrompe automaticamente quando detecta perda `NaN` ou infinita.

Essas correções tornam uma retomada futura mais segura, mas não mudam a
decisão atual de não investir mais tempo em fine-tuning de perguntas.

## Limitações que impedem uma promoção

- Os datasets públicos de QA normalmente trazem uma fonte de apoio, enquanto o
  uso em produção exige duas fontes independentes para uma resposta conclusiva.
- Os exemplos realmente multifonte precisam ser revisados e curados; a
  infraestrutura para recebê-los existe, mas eles ainda não são suficientes.
- A avaliação do smoke test mostrou falha no contrato de JSON, não apenas baixa
  qualidade factual.
- O tempo local de treinamento e validação não se justificou diante desses
  sinais iniciais.

## Caminho recomendado sem fine-tuning

Use o modelo-base no modo `evidencias`. A aplicação recupera links, apresenta
trechos ao modelo, exige JSON estruturado, valida citações e faz nova tentativa
quando a saída não cumpre o contrato. Para conversa geral, use o modo
`conversa`, também com o modelo-base.

Os adapters de evidências e de classificação publicados no Hugging Face seguem
separados, pois têm contratos de saída diferentes. Não misture seus pesos com
um futuro adapter de perguntas.

## Condições para reconsiderar no futuro

Só retome o ajuste fino se houver:

1. um conjunto curado de perguntas com duas ou mais fontes independentes;
2. validação de comprimento por tokens antes do treino;
3. um smoke test com perdas finitas e JSON válido;
4. uma avaliação reservada com métricas de resposta, citações e abstenção;
5. orçamento de tempo suficiente para comparar checkpoints e revisar erros.
