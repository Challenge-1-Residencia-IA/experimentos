# Guia de anotação do golden set

Como preencher a planilha de anotação do golden set. O golden set é o conjunto fixo de mensagens, com gabarito escrito por pessoas, usado para medir os comportamentos da IA definidos em [Requisitos](https://challenge-1-residencia-ia.github.io/Documentacao/requisitos/) (EN01).

## Princípios

1. **Anote a mensagem, não a checagem.** A coluna `texto` traz o boato como ele circulou. A página da `fonte_esperada` pode ser lida para entender o caso, mas a conclusão dela não vai para a planilha.
2. **Registre o que está no texto, nunca um veredito.** Nenhum campo diz se a mensagem é verdadeira ou falsa, assim como o bot não dá veredito.
3. **Não deduza intenção.** Se algo não aparece escrito na mensagem, a resposta é "não".
4. **Use só os valores das listas suspensas.** Nada de "nao", "-", "outro" em campo que não tem essa opção. Campo que não se aplica fica vazio.
5. **Anotação humana.** Nenhum campo é preenchido por modelo de linguagem: o gabarito existe para avaliar um.
6. **Na dúvida, escreva em `observacoes`** e siga em frente. As dúvidas viram regras novas neste guia.

## Ordem de preenchimento de cada linha

1. Ler o `texto` inteiro.
2. `manter` (se "não", passar para a próxima linha).
3. Afirmações e natureza.
4. Técnicas de manipulação.
5. Tema, sensível ao tempo, precisa de contexto, tema de risco.
6. Anotador e observações.

---

## `manter`

Decide se a mensagem entra no golden set.

| Marque **não** quando | Exemplo |
|---|---|
| O texto está quebrado ou cortado a ponto de não dar para entender | Só "versão 2: a polícia…", sem o resto |
| É praticamente igual a outra mensagem já mantida | Duas versões do mesmo boato |
| Não há nenhuma afirmação | "olha isso!!!" |
| Depende de uma imagem ou vídeo que não está descrito no texto | "vejam o vídeo abaixo" e nada mais |

## Afirmações (`afirmacao_1` a `afirmacao_3`)

O que a mensagem afirma e que pode ser analisado. Mede se o bot separa as afirmações da mensagem (IA-01).

**Regras:**

1. **Uma frase curta, com as suas palavras.** Não copie trechos longos, não use caixa alta, não comece com "a notícia diz que" ou "o usuário afirma que".
2. **Em ordem de importância.** A `afirmacao_1` é o núcleo do boato, a ideia que faz a mensagem circular.
3. **A mesma ideia conta uma vez só.** Duas frases que dizem a mesma coisa viram uma afirmação.
4. **Só uma afirmação? Preencha só a `afirmacao_1`** e deixe as outras vazias.
5. **Mais de 3?** Registre as 3 principais e anote em `observacoes`.
6. **Chamadas e pedidos não são afirmações.** "Aviso importante", "repassem", "façam as contas" vão nas técnicas.

| ✅ Assim | ❌ Não assim |
|---|---|
| "O TSE entregou o código-fonte das urnas à Smartmatic" (GS048) | "BOMBA!!! O TSE entregou os códigos fonte das urnas eletrônicas pra 05 venezuelanos da Smartmatic um pouco antes…" |
| "A ivermectina tem eficácia maior que as vacinas contra a covid-19" (GS054) | Afirmação 1: "Ivermectina supera vacinas"; afirmação 2: "Estudo revela 83% de eficácia" (mesma ideia em duas) |
| "Homem beijou uma criança em passeio de barco na Ilha de Marajó" (GS053) | "O usuário que está dando essa opinião critica duramente a mídia" (descreve a mensagem em vez de registrar o que ela afirma) |

## Natureza (`natureza_1` a `natureza_3`)

O tipo de cada afirmação, que define como o bot deve tratá-la (IA-01). Taxonomia de [Fluxo e interação](https://challenge-1-residencia-ia.github.io/Documentacao/fluxo-e-interacao/).

**Regra central: a natureza responde "dá para checar?", nunca "é verdade?".** Uma afirmação falsa, absurda ou perigosa continua sendo fato verificável se dá para checá-la.

| Natureza | Pergunta | Exemplo |
|---|---|---|
| Fato verificável | Dá para checar agora se é ou se aconteceu? | "Mel com canela cura a maioria das doenças" (GS057): é falso, mas é **checável** |
| Opinião | É julgamento, gosto ou xingamento, sem nada a checar? | "A mídia brasileira é nojenta" (GS053) |
| Previsão | Fala de algo que ainda não tinha acontecido quando a mensagem foi escrita? | "Vai começar a circular no WhatsApp um vídeo que hackeia o celular" (GS007) |
| Sátira | Foi produzida como piada, por uma fonte de humor? | Notícia do Sensacionalista compartilhada como real |
| Fora de contexto | O conteúdo é real, mas a data, o lugar ou a situação foram trocados? | Foto real de uma enchente antiga apresentada como de hoje |

**Casos de fronteira:**

- **Fato e opinião na mesma frase:** separe. "Boulos defende um serial killer, isso é a esquerda" (GS050) tem um fato ("Boulos classificou Lázaro como vítima da sociedade") e uma opinião ("a esquerda defende criminosos").
- **Afirmação disfarçada de pergunta:** "Você sabia que o mel cura doenças?" é fato verificável.
- **Previsão cujo prazo já passou:** continua sendo previsão. O desfecho vai em `sensivel_ao_tempo` ("sim") e em `observacoes` ("prazo já passou").
- **Boato absurdo sem origem humorística** ("chip na vacina") é fato verificável, não sátira.

## Técnicas de manipulação (`tec_*`)

Como a mensagem tenta convencer ou fazer repassar. Mede se o bot aponta esses recursos ao usuário (IA-02). Cada coluna é independente: a mensagem pode ter várias técnicas ou nenhuma, e **"não" deve ser a resposta mais comum**. Uma coluna marcada "sim" em todas as linhas não serve para avaliar nada.

| Coluna | Marque **sim** quando o texto tem… | Marque **não** quando… |
|---|---|---|
| `tec_urgencia` | Pressão para agir rápido: "AVISO IMPORTANTE", "URGENTE", "antes que apaguem", "atenção!" | Uma data dita só como informação |
| `tec_pedido_compartilhamento` | Pedido explícito para espalhar: "repassem", "compartilhe", "mande para seus grupos" | Só chama atenção: "olha isso", "vejam" |
| `tec_autoridade_anonima` | Fonte vaga, sem nome nem onde conferir: "a polícia afirmou", "estudiosos dizem", "um estudo do Reino Unido" (GS054) | A fonte tem nome identificável: "o ex-secretário de TI do TSE, Giuseppe Dutra" (GS048) |
| `tec_apelo_emocional` | Linguagem **explícita** de medo, raiva, nojo ou indignação: "os adoradores do mal atacam novamente" (GS050), "nojento demais" (GS053) | O tom é informativo, mesmo que o assunto seja grave: "Facebook desliga inteligência artificial…" (GS046) |
| `tec_outra` | Outro recurso, descrito em poucas palavras: "imita o visual do G1", "alarmismo com exclamações", "foto como prova" | Vazio se não houver (não escreva "não" nem "-") |

**Regra do apelo emocional:** o assunto ser polêmico (política, crime, vacina) **não basta**. É preciso haver no texto palavras que buscam provocar a emoção: adjetivos de indignação, xingamentos, ameaças, exclamações enfáticas.

## `tema`

O assunto principal: saúde, política, economia, ciência ou outro. Se a mensagem mistura assuntos, escolha o central.

- **Doença, tratamento, vacina, alimento como remédio: saúde**, mesmo que a mensagem cite "a ciência" (GS054, GS057).
- **Ciência** fica para tecnologia, meio ambiente, espaço e pesquisa sem relação com saúde (GS046, inteligência artificial).
- **Eleição, governo, partidos, políticos e tribunais: política.**
- Segurança pública, crimes, celebridades e golpes: outro.

## `sensivel_ao_tempo`

**Sim** quando a resposta depende da data: a afirmação só faz sentido numa época, pode mudar com o tempo ou fala de "agora", "acaba de", "a partir de".

| Sim | Não |
|---|---|
| "A OMS **acaba de** aprovar a vacina cubana" | "Mel com canela cura doenças" (alegação atemporal) |
| "Lázaro fugiu em direção a MG" (só faz sentido durante a fuga, em 2021) | "O TSE entregou o código-fonte em 2018" (fato passado, datado, que não muda) |
| Qualquer previsão | |

## `precisa_contexto`

**Sim** só quando o problema está na **apresentação**: o conteúdo pode ser real, mas a data, o lugar, a autoria ou a situação original precisam ser explicados para entender a mensagem. Quando basta verificar se o fato aconteceu, é **não**.

| Sim | Não |
|---|---|
| Foto ou vídeo real de outra época ou de outro país apresentado como atual ou local | "Mel com canela cura doenças" (basta checar a alegação) |
| Frase verdadeira de alguém, cortada para mudar o sentido | "A ivermectina supera as vacinas" (basta checar os estudos) |
| Boato antigo que voltou a circular como novo | |
| Natureza "fora de contexto" (quase sempre sim) | |

Para saber se o conteúdo é antigo ou de outro lugar, a página da `fonte_esperada` pode ser consultada.

## `tema_de_risco`

**Sim** quando acreditar na mensagem pode causar dano à pessoa ou a terceiros. Nesses casos o bot deve priorizar fontes oficiais e avisar que não é aconselhamento profissional (GR-06).

- **Sempre sim:** saúde, tratamentos e vacinas (GS054, GS057); eleições, urnas e voto (GS048); segurança física; golpes e pedidos de dados pessoais (GS044, cadastro com dados pessoais); desastres e emergências.
- **Não:** crítica política sem orientação de ação, fofoca, curiosidade.

O `tema` diz o assunto; o `tema_de_risco` diz se há dano possível. Saúde é quase sempre risco; política depende (eleição sim, crítica a um político não).

## `anotador` e `observacoes`

- **`anotador`:** nome ou iniciais de quem anotou a linha.
- **`observacoes`:** dúvidas, casos de fronteira, prazo de previsão vencido, mais de 3 afirmações. É o material usado para revisar este guia.

---

## Processo de anotação e concordância

1. **Rodada de concordância:** duas pessoas anotam os mesmos itens (cerca de 20, ainda não discutidos), cada uma na sua cópia, sem ver a outra.
2. **Cálculo:** `make golden-set` gera o golden set e calcula o kappa de Cohen por campo entre as duas abas da rodada (`dados/golden_set.py`); o resultado fica em `reports/golden-set-v1.md`.
3. **Leitura:** kappa acima de 0,6 (faixa "substancial" de Landis e Koch, 1977) indica que o campo está claro. Abaixo disso, a dupla discute as discordâncias e este guia é ajustado.
4. **Consenso:** as discordâncias da rodada são resolvidas em conjunto, e o valor acordado vai para o gabarito final.
5. **Restante:** com os campos acima de 0,6, os demais itens podem ser anotados por uma pessoa, com revisão dos casos marcados em `observacoes`.

A planilha é lida de `data/raw/golden_set/golden_set_v1_anotacao.xlsx`. O script valida cada item contra este guia e lista as pendências (campos vazios, valores fora das listas, afirmações sem natureza) no relatório.

O kappa da rodada de concordância é registrado junto com a versão do golden set.

## Histórico

- **v1:** regras definidas a partir da rodada piloto com 10 itens (GS044–GS059). Pontos que motivaram regras: natureza confundida com veracidade, técnicas marcadas sem critério mínimo, temas de risco de saúde e eleição não marcados, afirmações copiadas em vez de resumidas.
