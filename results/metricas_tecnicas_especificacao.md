# Métricas Técnicas de Avaliação (Avaliação Desacoplada por Etapa)

Este documento define o framework de avaliação técnica do **Copiloto de Checagem de Fatos e Alfabetização Midiática (Sift AI)**.

> [!IMPORTANT]
> **Princípio Fundamental:**
> Todas as métricas devem ser avaliadas **separadamente por etapa**, para evitar que um bom resultado em uma etapa esconda problemas graves em outra (ex.: um acerto na classificação final alcançado por mero "chute" paramétrico, enquanto o módulo de busca de fontes falhou completamente).

---

## 1. Visão Geral das 8 Métricas Técnicas

| # | Métrica Técnica | Etapa do Pipeline | O que avalia isoladamente? |
|---|---|---|---|
| **1** | **Precisão (Precision)** | Etapa 5: Classificação Categórica | De todas as mensagens rotuladas como classe $C$, quantas realmente pertenciam a $C$. |
| **2** | **Recall (Revocação)** | Etapa 5: Classificação Categórica | De todas as mensagens que eram realmente da classe $C$, quantas o sistema identificou. |
| **3** | **F1-Score** | Etapa 5: Classificação Categórica | Média harmônica entre Precisão e Recall da classificação. |
| **4** | **Taxa de identificação correta das afirmações** | Etapa 1: Extração e Decomposição | Capacidade de isolar a alegação central verificável sem fragmentar ou omitir o núcleo. |
| **5** | **Qualidade das fontes recuperadas** | Etapa 2: Recuperação de Evidências (RAG) | Relevância e autoridade das fontes retornadas (agências de checagem, artigos científicos, órgãos oficiais). |
| **6** | **Taxa de alucinação** | Etapa 4: Síntese e Justificativa | Proporção de afirmações na justificativa que não encontram respaldo nas fontes ou no texto. |
| **7** | **Taxa de respostas sem evidência** | Etapa 2/3: Cobertura Factual | Proporção de casos em que o sistema arrisca uma resposta conclusiva mesmo sem recuperar evidências. |
| **8** | **Precisão na identificação de divergências** | Etapa 3: NLI / Verificação Factual | Exatidão do modelo ao apontar contradições entre a alegação do usuário e as evidências factuais. |

---

## 2. Detalhamento e Formulação Matemática

### 2.1. Etapa 1 — Extração de Afirmações
#### Métrica: Taxa de Identificação Correta das Afirmações
Avalia exclusivamente a capacidade de processamento de entrada e identificação de alegações.
$$\text{Taxa de Extração} = \frac{\text{Número de afirmações-chave identificadas com acerto}}{\text{Total de afirmações-chave esperadas no gabarito}}$$

* **Por que avaliar separadamente?** Se o modelo errar a extração da afirmação (ex: extrair um comentário irrelevante em vez da alegação central), todo o pipeline subsequente (busca e verificação) será direcionado ao alvo errado.

---

### 2.2. Etapa 2 — Recuperação RAG
#### Métrica: Qualidade das Fontes Recuperadas
Mede o percentual de fontes de alta confiabilidade (domínios governamentais, acadêmicos, agências de checagem consolidadas) e semanticamente relevantes:
$$\text{Qualidade das Fontes} = \frac{\sum \text{Fontes Confiáveis e Relevantes}}{\text{Total de Fontes Recuperadas}}$$

#### Métrica: Taxa de Respostas sem Evidência
Mede a frequência de respostas assertivas emitidas quando a busca externa retornou vazia ou nula:
$$\text{Taxa Sem Evidência} = \frac{\text{Casos respondidos conclusivamente sem evidência externa}}{\text{Total de casos}}$$

* **Por que avaliar separadamente?** Se o LLM tiver boa memória paramétrica pré-treinada, ele pode acertar se uma notícia é falsa mesmo se a busca trouxer links quebrados ou irrelevantes. Avaliar a qualidade das fontes isola a saúde do módulo de RAG.

---

### 2.3. Etapa 3 — Detecção de Divergências (NLI)
#### Métrica: Precisão na Identificação de Divergências
Avalia se o modelo acerta quando aponta que a alegação contradiz as evidências:
$$\text{Precisão de Divergência} = \frac{\text{Verdadeiros Positivos (Divergências Reais)}}{\text{Verdadeiros Positivos} + \text{Falsos Positivos (Alarmes Falsos)}}$$

* **Por que avaliar separadamente?** Permite detectar se o bot sofre de viés de refutação (apontar divergência onde há concordância) ou se é complacente com alegações enganosas.

---

### 2.4. Etapa 4 — Síntese e Justificativa
#### Métrica: Taxa de Alucinação (Faithfulness / Groundedness)
Proporção de afirmações geradas na justificativa que não possuem ancoragem factual nas evidências:
$$\text{Taxa de Alucinação} = \frac{\text{Afirmações na justificativa não suportadas pelas evidências}}{\text{Total de afirmações geradas na justificativa}}$$

* **Por que avaliar separadamente?** Um modelo pode acertar o rótulo final ("FALSA"), mas inventar fatos, datas ou estudos inexistentes na justificativa. Separar a métrica de alucinação protege a credibilidade do sistema.

---

### 2.5. Etapa 5 — Classificação Categórica Final
#### Métricas: Precisão, Recall e F1-Score
Avaliadas por classe (Falsa, Verdadeira, Fora de contexto, Opinião, Sátira) e via Macro/Weighted Average:
$$\text{Precisão}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = 2 \cdot \frac{\text{Precisão}_c \cdot \text{Recall}_c}{\text{Precisão}_c + \text{Recall}_c}$$

---

## 3. Como Executar a Avaliação

O script `scripts/metricas_tecnicas.py` implementa o cálculo completo:

```bash
# Execução padrão no dataset Sprint 1:
python scripts/metricas_tecnicas.py

# Execução avaliando arquivo com predições reais de um experimento:
python scripts/metricas_tecnicas.py --predictions_file results/Sprint_1_Resultados_Avaliacao.xlsx --output_json results/relatorio_metricas_tecnicas.json
```

## 4. Limites e protocolo de validação

- A taxa de extração usa similaridade lexical e deve ser complementada por revisão humana ou métrica semântica em uma amostra fixa.
- A taxa de alucinação mede ancoragem lexical nas evidências; ela é um sinal de alerta, não uma prova isolada de invenção factual.
- O gabarito de `Divergencia_Esperada` deve ser anotado diretamente. Não derive esse valor da categoria final, em especial para opinião e sátira.
- A avaliação de fontes considera o domínio da URL, não menções textuais a domínios no snippet. Relevância factual requer anotação ou avaliação complementar.
- Cada rodada deve registrar versão de prompt, modelo, adaptador, dataset e data no JSON de saída.
