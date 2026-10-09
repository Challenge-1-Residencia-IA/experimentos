# Rodada 4 — alertas automáticos de alegações

O último teste registrado no MLflow, run `40eaf925f05845d28454b1515c3e3927`, perguntou: “Pablo Vittar rouba ônibus mata Neymar e resgata Lula na prisão, isso é verdade?”. A análise descartou sete páginas sem pertinência direta e retornou uma mensagem genérica de insuficiência de fontes.

Agora o sistema sinaliza sequências curtas que combinam roubo, morte/violência e resgate/fuga. Sem trechos validados, fornece automaticamente uma explicação de forte desconfiança, pede a publicação original e recomenda não compartilhar a história como notícia. Os links consultados continuam disponíveis. Alegações isoladas de morte ou crime recebem um alerta de gravidade, sem serem chamadas de falsas ou absurdas.

A regra é uma heurística de linguagem explícita, sem nomes de pessoas fixados no código. Não cobre todo conteúdo absurdo nem determina sua verdade. Hipóteses, ficção explicitamente indicada e negação desativam o alerta conservadoramente. Se existirem citações validadas, sua síntese é preservada e o alerta aparece nos pontos de atenção.

Implementação em `rag/claim_alerts.py`, integrada ao validador comum. A versão da resposta passa a `analise-evidencias-v9` para evitar reutilizar respostas antigas em cache. Não há nova chamada ao modelo, nem interrupção da pesquisa web por causa do alerta.

Regressões: último caso, notícia grave plausível, ficção/negação e preservação dos links. Esse mecanismo é um primeiro detector explicável; não equivale ao entendimento geral de um modelo maior e requer avaliação com exemplos reais para expandir sua cobertura.
