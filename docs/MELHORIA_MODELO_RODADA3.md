# Rodada 3 — análise mais direta, sem veredito binário

## Comportamento

A resposta descreve explicitamente a direção dos trechos validados. Mesmo com uma fonte ainda não qualificada, pode dizer: “Os trechos selecionados [1] contradizem a afirmação”. Depois explica que falta corroborar o material. A ausência de qualificação não apaga o achado, mas continua impedindo a classificação como evidência suficiente.

Quando há trechos nos dois sentidos, a resposta expõe a divergência e suas referências. Duas fontes qualificadas favoráveis não apagam um trecho contrário de uma fonte ainda não qualificada. A contagem de fontes não funciona como votação sobre a verdade.

Sem trechos utilizáveis, o retorno continua informando que não há evidência suficiente e preservando os links encontrados. Citações inventadas e recortes que omitem um desmentido continuam sendo rejeitados.

Todas as respostas produzidas incluem `aviso_ia`: “Sou uma IA e posso cometer erros.” A interface exibe esse aviso e substitui o título “Veredito” por “O que as fontes indicam”.

## Contrato e compatibilidade

- Versão `analise-evidencias-v8`, também usada para invalidar respostas antigas em cache.
- Novo campo `sintese`, produzido a partir dos trechos validados e com referências numéricas.
- O campo legado `veredito` permanece como alias da síntese para consumidores existentes. Não contém rótulo verdadeiro/falso.
- O prompt orienta a extrair evidências diretas mesmo com apenas uma fonte ou sem qualificação, preservando evidências em sentidos opostos.
- Não há chamada adicional ao modelo para redigir a síntese; o código a monta após validar as citações.

## Validação reproduzível

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.avaliar_geracao_controlada --output results/geracao-controlada-rodada3.json
```

A suíte passou com 90 testes. As novas regressões cobrem direção favorável/contrária sem qualificação, conflito que antes podia ser apagado, ausência de evidências e apresentação do aviso.

O Qwen `qwen2.5:7b` passou nos 6/6 cenários controlados: fontes favoráveis, contrárias, mesma origem, nenhuma evidência, contrária sem qualificação e fontes conflitantes. A extração com geração levou de 3,642 a 6,313 segundos por cenário; o retorno sem evidências levou 0,002 segundo. São tempos locais observados, sem busca web.

O teste de geração usa fontes sintéticas e Qwen local; não mede recuperação web nem acurácia em notícias reais. A conferência literal das citações não garante interpretação semântica correta. A qualidade das fontes, sua data e sua independência continuam relevantes.

As alterações desta rodada entram em vigor ao reiniciar a aplicação e não dependem de reindexação. A reindexação continua necessária para ativar a nova divisão de documentos da rodada 2.
