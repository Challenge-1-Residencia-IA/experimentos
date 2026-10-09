# Recuperação de padrões e prompt adaptativo

A base `knowledge/padroes_discurso.json` contém fichas educativas próprias: opinião, sátira/humor, ficção/hipótese, sensacionalismo, rumor e alegação factual. Cada ficha contém descrição, exemplo, marcadores de recuperação e orientação para o prompt.

`rag/discourse.py` recupera fichas por correspondência lexical normalizada. Os sinais observados ficam registrados na resposta, juntamente com IDs das fichas e hash da base. Vários tipos podem coexistir. Sem marcadores, o resultado é indeterminado. A classificação indica sinais de escrita, não intenção comprovada do autor.

O prompt de extração recebe as fichas recuperadas e orientações específicas de cada tipo. A resposta exibe esses sinais e, sem evidências validadas, adapta a explicação para opinião, hipótese, humor, rumor ou sensacionalismo. Os exemplos educativos não entram como fontes factuais. O prompt efetivamente utilizado e o perfil recuperado ficam no rastreio da consulta no MLflow.

Essa base de padrões é local e separada do banco vetorial factual. Não precisa de reindexação no Neon/Chroma. A recuperação atual é lexical, não semântica; não foi treinada no Fake.br e não mede probabilidade de fake news. Ampliar a cobertura exige exemplos revisados e avaliação de falsos positivos, especialmente ironia, negação e textos que misturam gêneros.

O contrato passa a `analise-evidencias-v10`. O cache considera também o hash da base. A adaptação não exige chamada adicional ao modelo. Reinicie a aplicação para carregar o código atualizado.
