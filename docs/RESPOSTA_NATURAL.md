# Resposta natural — versão v11

A conversa agora apresenta uma análise em parágrafos e as fontes disponíveis. Foram removidos da interface os blocos vazios de evidências favoráveis/contrárias, contexto, avaliação e limitações. Os campos estruturados permanecem no JSON para avaliação e rastreabilidade.

Após selecionar e validar as citações, uma etapa de redação pede ao Qwen uma explicação específica: o que o material diz, sua relação com a pergunta e o que ainda falta investigar. Essa etapa também funciona sem evidências factuais, para explicar opiniões, hipóteses ou a insuficiência da pesquisa a partir da mensagem. Ela não usa conhecimentos externos como evidência nem transforma improbabilidade em falsidade.

O novo campo público é `resposta_usuario`. A interface, o modo web, o comando de consulta local e o modelo empacotado usam essa redação. O aviso de IA e as tabelas de fontes permanecem visíveis. Respostas antigas ainda podem ser exibidas usando `sintese` ou `veredito`.

A redação é orientada a citar somente fontes já usadas na análise. Referências inexistentes, ausência de citações quando há evidências, texto inválido e links criados na narrativa são rejeitados. Nesse caso, a interface recebe uma explicação extrativa com as citações conferidas. Essas verificações não garantem suporte semântico de todas as frases: a fidelidade da interpretação continua exigindo avaliação.

A redação também recebe um campo semântico `natureza` (opinião, sátira, hipótese, alegação factual, misto ou indeterminado). Quando o próprio modelo identifica uma mensagem exclusivamente subjetiva/fictícia, uma proteção substitui eventuais exigências indevidas de confirmação factual por uma explicação apropriada. Essa classificação ainda pode errar e não substitui a avaliação do detector lexical existente.

O prompt de redação, entrada, saída bruta e motivo de fallback são registrados em `saida_modelo.redacao_resposta` no rastreio MLflow. Os diagnósticos distinguem tentativas de extração e de redação. O cache reutiliza a resposta inteira e a versão v11 invalida entradas antigas.

Há uma chamada adicional ao Qwen por pergunta nova; o tempo de geração aumenta. Sem evidências, ocorre apenas a chamada de redação. Essa mudança não corrige por si só todas as limitações de relevância e classificação lexical identificadas na auditoria.

Para testar, reinicie a aplicação e execute:

```bash
.venv/bin/python main.py --opcao 3
```

Compare uma opinião sem palavras-chave (por exemplo, “Esse debate foi horrível”), uma paródia explícita e uma alegação acompanhada de fontes. Observe se a resposta explica a mensagem concretamente, cita as fontes ao usar fatos e diz qual informação falta, sem empilhar seções vazias.
