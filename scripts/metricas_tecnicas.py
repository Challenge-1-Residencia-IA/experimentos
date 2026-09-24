#!/usr/bin/env python3
"""
Módulo de Avaliação de Métricas Técnicas - Copiloto de Checagem e Alfabetização Midiática.

Este módulo implementa o cálculo modular e independente das 8 métricas técnicas:
  1. Precisão (Precision)
  2. Recall (Revocação)
  3. F1-Score
  4. Taxa de identificação correta das afirmações
  5. Qualidade das fontes recuperadas
  6. Taxa de alucinação
  7. Taxa de respostas sem evidência
  8. Precisão na identificação de divergências

IMPORTANTE:
  Essas métricas são avaliadas separadamente por etapa do pipeline (Extração,
  Recuperação RAG, Identificação de Divergências, Síntese e Classificação)
  para garantir que o sucesso aparente em uma etapa não mascare falhas em outra.
"""

import os
import re
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

from rich.table import Table

from ui_terminal import cabecalho, console, tabela_metricas

# Tentativa de importação para busca web e inferência com MLX
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

try:
    from mlx_lm import load, generate
except ImportError:
    load = None
    generate = None


# Domínios e padrões de alta credibilidade para Fact-Checking no Brasil
DOMINIOS_CONFIAVEIS = [
    "aosfatos.org",
    "lupa.uol.com.br",
    "estadao.com.br/estadao-verifica",
    "g1.globo.com/fato-ou-fake",
    "boatos.org",
    "projetocomprova.com.br",
    "fiocruz.br",
    "gov.br",
    "who.int",
    "saude.gov.br",
    "anvisa.gov.br",
    "tse.jus.br",
    "folha.uol.com.br",
    "bbc.com",
    "reuters.com",
    "nature.com",
    "scielo.br",
    "uol.com.br/noticias",
    "metropoles.com"
]

DOMINIOS_BAIXA_QUALIDADE = [
    "facebook.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "instagram.com",
    "youtube.com",
    "t.me",
    "chat.whatsapp.com"
]


def dominio_da_url(url: str) -> str:
    """Retorna o host normalizado de uma URL, sem considerar o texto do snippet."""
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def url_corresponde_padrao(url: str, padrao: str) -> bool:
    """Compara host e, quando definido, o caminho de uma fonte confiável."""
    host = dominio_da_url(url)
    dominio, _, caminho_esperado = padrao.partition("/")
    if not (host == dominio or host.endswith(f".{dominio}")):
        return False
    return not caminho_esperado or urlparse(url).path.lower().startswith(f"/{caminho_esperado}")

PROMPT_EXTRAIR_AFIRMACOES = """Você é um analista de dados estruturados para um copiloto investigativo de alfabetização midiática.
Sua tarefa é DECOMPOR a mensagem do usuário nas suas afirmações principais e classificar a sua natureza.

REGRAS DE EXTRAÇÃO:
1. Limite-se a extrair no máximo 1 a 2 afirmações centrais por mensagem. Não fragmente o texto.
2. Não invente informações que não estão explicitamente escritas.

DEFINIÇÃO DE CATEGORIAS:
- "fato verificavel": Alegações objetivas que podem ser provadas ou refutadas (notícias falsas ou verdadeiras).
- "fora de contexto": O facto ocorreu, mas data, local ou intenção foram alterados.
- "opiniao": Juízos de valor ou interpretações pessoais sem base em factos objetivos.
- "satira": Exageros humorísticos, ironia ou paródia evidentes.
- "previsao": Especulações sobre eventos futuros.

Retorne EXCLUSIVAMENTE um objeto JSON válido.
{
  "afirmacoes": [
    {"afirmacao": "[texto da alegação principal]", "tipo": "[categoria]", "verificavel": true}
  ]
}"""

PROMPT_VERIFICACAO_SISTEMA = """Você é um especialista em checagem de fatos e análise do discurso.
Use as EVIDÊNCIAS DA WEB fornecidas para verificar a MENSAGEM recebida.
Classifique a mensagem estritamente em uma destas 5 categorias: FALSA, VERDADEIRA, FORA DE CONTEXTO, OPINIÃO ou SÁTIRA.

Responda exatamente neste formato:
CLASSIFICAÇÃO: [Categoria]
DIVERGÊNCIA: [SIM ou NÃO]
JUSTIFICATIVA: [Sua análise concisa em 2 frases baseada nas evidências]"""


# ==============================================================================
# ESTRUTURAS DE DADOS DAS ETAPAS
# ==============================================================================

@dataclass
class MetricasExtracao:
    """Etapa 1: Extração e Decomposição de Afirmações."""
    total_casos: int = 0
    total_afirmacoes_esperadas: int = 0
    total_afirmacoes_extraidas: int = 0
    afirmacoes_corretamente_identificadas: int = 0
    taxa_identificacao_correta: float = 0.0
    detalhes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MetricasRecuperacao:
    """Etapa 2: Recuperação de Evidências (RAG / Search)."""
    total_casos: int = 0
    total_fontes_recuperadas: int = 0
    fontes_alta_qualidade: int = 0
    fontes_baixa_qualidade_ou_ruido: int = 0
    qualidade_fontes_score: float = 0.0
    casos_sem_evidencia: int = 0
    respostas_assertivas_sem_evidencia: int = 0
    taxa_respostas_sem_evidencia: float = 0.0
    detalhes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MetricasDivergencia:
    """Etapa 3: Raciocínio Comparativo e Identificação de Divergências."""
    total_casos: int = 0
    true_positives_divergencia: int = 0
    false_positives_divergencia: int = 0
    false_negatives_divergencia: int = 0
    true_negatives_divergencia: int = 0
    precisao_identificacao_divergencias: float = 0.0
    recall_divergencias: float = 0.0
    f1_divergencias: float = 0.0
    detalhes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MetricasGeracao:
    """Etapa 4: Síntese, Justificativa e Fidelidade Textual."""
    total_justificativas_avaliadas: int = 0
    total_afirmacoes_sintetizadas: int = 0
    afirmacoes_suportadas_evidencia: int = 0
    afirmacoes_alucinadas: int = 0
    taxa_alucinacao: float = 0.0
    detalhes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MetricasClassificacao:
    """Etapa 5: Categorização Final (Falsa, Verdadeira, Fora de Contexto, Opinião, Sátira)."""
    total_amostras: int = 0
    acuracia_global: float = 0.0
    precisao_macro: float = 0.0
    recall_macro: float = 0.0
    f1_score_macro: float = 0.0
    precisao_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_score_weighted: float = 0.0
    por_classe: Dict[str, Dict[str, float]] = field(default_factory=dict)
    matriz_confusao: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class RelatorioTecnicoCompleto:
    """Relatório unificado mantendo as etapas isoladas para evitar mascaramento de erros."""
    extracao: MetricasExtracao
    recuperacao: MetricasRecuperacao
    divergencia: MetricasDivergencia
    geracao: MetricasGeracao
    classificacao: MetricasClassificacao


# ==============================================================================
# FUNÇÕES DE CÁLCULO INDIVIDUAL (ETAPAS SEPARADAS)
# ==============================================================================

def normalizar_texto(texto: str) -> str:
    """Normaliza texto removendo pontuações extras e espaços duplicados."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()


def normalizar_categoria(texto_categoria: str) -> str:
    """Normaliza rótulos de categoria."""
    txt = str(texto_categoria).upper().strip()
    if "FORA DE CONTEXTO" in txt or "CONTEXTO" in txt:
        return "FORA DE CONTEXTO"
    elif "FALSA" in txt or "FALSO" in txt:
        return "FALSA"
    elif "VERDADEIRA" in txt or "VERDADEIRO" in txt:
        return "VERDADEIRA"
    elif "OPINIÃO" in txt or "OPINIAO" in txt:
        return "OPINIÃO"
    elif "SÁTIRA" in txt or "SATIRA" in txt:
        return "SÁTIRA"
    return "INDEFINIDO"


def similaridade_tokens(texto1: str, texto2: str) -> float:
    """Calcula sobreposição lexical (Jaccard) entre dois textos."""
    tokens1 = set(normalizar_texto(texto1).split())
    tokens2 = set(normalizar_texto(texto2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersecao = tokens1.intersection(tokens2)
    uniao = tokens1.union(tokens2)
    return len(intersecao) / len(uniao)


def calcular_taxa_identificacao_afirmacoes(
    afirmacoes_extraidas_por_caso: List[List[str]],
    afirmacoes_gabarito_por_caso: List[str],
    limiar_similaridade: float = 0.25
) -> MetricasExtracao:
    """
    MÉTRICA 4: Taxa de identificação correta das afirmações.

    Avalia estritamente a habilidade do modelo de isolar a alegação principal.
    """
    total_casos = len(afirmacoes_gabarito_por_caso)
    corretas = 0
    total_extraidas = 0
    detalhes = []

    for idx, (extraidas, gabarito) in enumerate(zip(afirmacoes_extraidas_por_caso, afirmacoes_gabarito_por_caso), 1):
        total_extraidas += len(extraidas)
        max_sim = 0.0
        melhor_extraida = ""

        for ext in extraidas:
            sim = similaridade_tokens(ext, gabarito)
            if sim > max_sim:
                max_sim = sim
                melhor_extraida = ext

        acertou = max_sim >= limiar_similaridade
        if acertou:
            corretas += 1

        detalhes.append({
            "id": idx,
            "afirmacao_esperada": gabarito,
            "afirmacoes_extraidas": extraidas,
            "melhor_match": melhor_extraida,
            "similaridade": round(max_sim, 3),
            "identificacao_correta": acertou
        })

    taxa = (corretas / total_casos) if total_casos > 0 else 0.0

    return MetricasExtracao(
        total_casos=total_casos,
        total_afirmacoes_esperadas=total_casos,
        total_afirmacoes_extraidas=total_extraidas,
        afirmacoes_corretamente_identificadas=corretas,
        taxa_identificacao_correta=round(taxa, 4),
        detalhes=detalhes
    )


def calcular_qualidade_fontes_e_cobertura(
    fontes_por_caso: List[List[Dict[str, str]]],
    respostas_emitidas: List[Dict[str, Any]]
) -> MetricasRecuperacao:
    """
    MÉTRICAS 5 e 7:
      - Qualidade das fontes recuperadas
      - Taxa de respostas sem evidência
    """
    total_casos = len(fontes_por_caso)
    total_fontes = 0
    fontes_alta = 0
    fontes_baixa = 0
    casos_sem_evidencia = 0
    respostas_sem_evidencia = 0
    detalhes = []

    for idx, fontes in enumerate(fontes_por_caso, 1):
        qtd_fontes = len(fontes)
        total_fontes += qtd_fontes

        fontes_confiaveis_caso = 0
        fontes_ruidosas_caso = 0

        for f in fontes:
            url = f.get("url", "")
            eh_confiavel = any(url_corresponde_padrao(url, dom) for dom in DOMINIOS_CONFIAVEIS)
            eh_baixa = any(url_corresponde_padrao(url, dom) for dom in DOMINIOS_BAIXA_QUALIDADE)

            if eh_confiavel:
                fontes_confiaveis_caso += 1
            elif eh_baixa:
                fontes_ruidosas_caso += 1

        fontes_alta += fontes_confiaveis_caso
        fontes_baixa += fontes_ruidosas_caso

        # Ausência de evidência
        sem_evidencia = (qtd_fontes == 0) or all("nenhuma evidência" in f.get("body", "").lower() for f in fontes)
        if sem_evidencia:
            casos_sem_evidencia += 1

        resp_info = respostas_emitidas[idx - 1] if idx - 1 < len(respostas_emitidas) else {}
        resposta_conclusiva = resp_info.get("conclusiva", True)

        if sem_evidencia and resposta_conclusiva:
            respostas_sem_evidencia += 1

        score_caso = (fontes_confiaveis_caso / qtd_fontes) if qtd_fontes > 0 else 0.0

        detalhes.append({
            "id": idx,
            "total_fontes": qtd_fontes,
            "fontes_confiaveis": fontes_confiaveis_caso,
            "sem_evidencia": sem_evidencia,
            "score_qualidade": round(score_caso, 3),
            "emitiu_sem_evidencia": sem_evidencia and resposta_conclusiva
        })

    qualidade_score = (fontes_alta / total_fontes) if total_fontes > 0 else 0.0
    taxa_sem_evidencia = (respostas_sem_evidencia / total_casos) if total_casos > 0 else 0.0

    return MetricasRecuperacao(
        total_casos=total_casos,
        total_fontes_recuperadas=total_fontes,
        fontes_alta_qualidade=fontes_alta,
        fontes_baixa_qualidade_ou_ruido=fontes_baixa,
        qualidade_fontes_score=round(qualidade_score, 4),
        casos_sem_evidencia=casos_sem_evidencia,
        respostas_assertivas_sem_evidencia=respostas_sem_evidencia,
        taxa_respostas_sem_evidencia=round(taxa_sem_evidencia, 4),
        detalhes=detalhes
    )


def calcular_precisao_divergencias(
    divergencias_detectadas: List[bool],
    divergencias_esperadas: List[bool]
) -> MetricasDivergencia:
    """
    MÉTRICA 8: Precisão na identificação de divergências.
    """
    total = len(divergencias_detectadas)
    tp = fp = fn = tn = 0
    detalhes = []

    for idx, (pred, exp) in enumerate(zip(divergencias_detectadas, divergencias_esperadas), 1):
        if pred and exp:
            tp += 1
            status = "TP (Divergência correta)"
        elif pred and not exp:
            fp += 1
            status = "FP (Falso alarme de divergência)"
        elif not pred and exp:
            fn += 1
            status = "FN (Divergência não detectada)"
        else:
            tn += 1
            status = "TN (Concordância correta)"

        detalhes.append({
            "id": idx,
            "detectou_divergencia": pred,
            "esperava_divergencia": exp,
            "status": status
        })

    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0.0

    return MetricasDivergencia(
        total_casos=total,
        true_positives_divergencia=tp,
        false_positives_divergencia=fp,
        false_negatives_divergencia=fn,
        true_negatives_divergencia=tn,
        precisao_identificacao_divergencias=round(precisao, 4),
        recall_divergencias=round(recall, 4),
        f1_divergencias=round(f1, 4),
        detalhes=detalhes
    )


def calcular_taxa_alucinacao(
    justificativas_geradas: List[str],
    evidencias_e_contexto: List[str]
) -> MetricasGeracao:
    """
    MÉTRICA 6: Taxa de alucinação.
    """
    total_justificativas = len(justificativas_geradas)
    total_frases = 0
    frases_alucinadas = 0
    detalhes = []

    for idx, (just, contexto) in enumerate(zip(justificativas_geradas, evidencias_e_contexto), 1):
        frases = [f.strip() for f in re.split(r'[.!?;\n]+', just) if len(f.strip()) > 10]
        if not frases:
            frases = [just.strip()] if just.strip() else []

        total_frases += len(frases)
        alucinadas_caso = 0
        contexto_norm = normalizar_texto(contexto)

        for frase in frases:
            tokens_frase = set(normalizar_texto(frase).split())
            tokens_contexto = set(contexto_norm.split())

            stopwords = {"o", "a", "os", "as", "um", "uma", "de", "do", "da", "em", "no", "na", "que", "e", "é", "por", "com", "para", "se", "este", "esta", "foi"}
            tokens_conteudo = tokens_frase - stopwords

            if not tokens_conteudo:
                continue

            sobreposicao = len(tokens_conteudo.intersection(tokens_contexto)) / len(tokens_conteudo)

            if sobreposicao < 0.20:
                alucinadas_caso += 1

        frases_alucinadas += alucinadas_caso
        detalhes.append({
            "id": idx,
            "justificativa": just,
            "total_frases": len(frases),
            "frases_alucinadas": alucinadas_caso,
            "taxa_caso": round((alucinadas_caso / len(frases)) if frases else 0.0, 3)
        })

    taxa_global = (frases_alucinadas / total_frases) if total_frases > 0 else 0.0

    return MetricasGeracao(
        total_justificativas_avaliadas=total_justificativas,
        total_afirmacoes_sintetizadas=total_frases,
        afirmacoes_suportadas_evidencia=total_frases - frases_alucinadas,
        afirmacoes_alucinadas=frases_alucinadas,
        taxa_alucinacao=round(taxa_global, 4),
        detalhes=detalhes
    )


def calcular_precisao_recall_f1(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> MetricasClassificacao:
    """
    MÉTRICAS 1, 2 e 3: Precisão, Recall e F1-Score da Classificação Categórica.
    """
    y_true_clean = [normalizar_categoria(x) for x in y_true]
    y_pred_clean = [normalizar_categoria(x) for x in y_pred]

    if labels is None:
        labels = ["FALSA", "VERDADEIRA", "FORA DE CONTEXTO", "OPINIÃO", "SÁTIRA"]

    total_amostras = len(y_true_clean)
    acertos = sum(1 for yt, yp in zip(y_true_clean, y_pred_clean) if yt == yp)
    acuracia = acertos / total_amostras if total_amostras > 0 else 0.0

    por_classe = {}
    matriz_confusao = {l_real: {l_pred: 0 for l_pred in labels} for l_real in labels}

    for yt, yp in zip(y_true_clean, y_pred_clean):
        if yt in matriz_confusao:
            if yp in matriz_confusao[yt]:
                matriz_confusao[yt][yp] += 1

    precisoes = []
    recalls = []
    f1s = []
    pesos = []

    for classe in labels:
        tp = matriz_confusao[classe][classe]
        fp = sum(matriz_confusao[outro][classe] for outro in labels if outro != classe)
        fn = sum(matriz_confusao[classe][outro] for outro in labels if outro != classe)
        suporte = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        por_classe[classe] = {
            "precisao": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "suporte": suporte
        }

        precisoes.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        pesos.append(suporte)

    prec_macro = float(np.mean(precisoes)) if precisoes else 0.0
    rec_macro = float(np.mean(recalls)) if recalls else 0.0
    f1_macro = float(np.mean(f1s)) if f1s else 0.0

    soma_pesos = sum(pesos)
    if soma_pesos > 0:
        prec_weighted = float(np.average(precisoes, weights=pesos))
        rec_weighted = float(np.average(recalls, weights=pesos))
        f1_weighted = float(np.average(f1s, weights=pesos))
    else:
        prec_weighted = rec_weighted = f1_weighted = 0.0

    return MetricasClassificacao(
        total_amostras=total_amostras,
        acuracia_global=round(acuracia, 4),
        precisao_macro=round(prec_macro, 4),
        recall_macro=round(rec_macro, 4),
        f1_score_macro=round(f1_macro, 4),
        precisao_weighted=round(prec_weighted, 4),
        recall_weighted=round(rec_weighted, 4),
        f1_score_weighted=round(f1_weighted, 4),
        por_classe=por_classe,
        matriz_confusao=matriz_confusao
    )


# ==============================================================================
# PIPELINE DE INFERÊNCIA DO MODELO LOCAL (TODAS AS ETAPAS)
# ==============================================================================

def buscar_evidencias_web(query: str, max_resultados: int = 3) -> Tuple[str, List[Dict[str, str]]]:
    """Busca evidências com tratamento de falhas e extração de metadados."""
    termo = query[:120].replace("\n", " ").strip()
    fontes_lista = []

    if DDGS is None:
        return "Buscador DDGS não instalado.", []

    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(termo, region="br-pt", max_results=max_resultados))
            if resultados:
                for idx, r in enumerate(resultados, 1):
                    fontes_lista.append({
                        "id": idx,
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "body": r.get("body", "")
                    })
                texto_evidencias = "\n".join([
                    f"[{f['id']}] {f['title']} ({f['url']}): {f['body']}"
                    for f in fontes_lista
                ])
                return texto_evidencias, fontes_lista
    except Exception as e:
        return f"Erro na busca: {e}", []
    return "Nenhuma evidência encontrada.", []

def avaliar_predicoes_salvas(
    caminho_excel: str,
    aba: str = "Dataset Sprint 1",
    arquivo_predicoes: str = "results/Sprint_1_Resultados_Modelo_Treinado.xlsx"
) -> RelatorioTecnicoCompleto:
    """Calcula todas as 8 métricas técnicas a partir de um arquivo de predições existente."""
    df_pred = pd.read_excel(arquivo_predicoes) if arquivo_predicoes.endswith(".xlsx") else pd.read_csv(arquivo_predicoes)

    gabarito_afirmacoes = df_pred["Afirmacao_Gabarito"].astype(str).tolist()
    afirmacoes_extraidas_todas = [[str(x)] for x in df_pred["Afirmacao_Extraida_IA"].tolist()]

    fontes_recuperadas_todas = []
    respostas_meta_todas = []
    for ev in df_pred["Evidencias_RAG"].fillna("").tolist():
        linhas_ev = [l for l in str(ev).split("\n") if l.strip()]
        fontes_caso = []
        for l in linhas_ev:
            fontes_caso.append({"url": l, "title": l[:40], "body": l})
        fontes_recuperadas_todas.append(fontes_caso)
        conclusiva = str(df_pred.get("Resposta_Conclusiva", pd.Series(["SIM"] * len(df_pred))).iloc[len(respostas_meta_todas)]).upper()
        respostas_meta_todas.append({"conclusiva": conclusiva in {"SIM", "TRUE", "1"}})

    divergencias_preditas = [str(d).upper() == "SIM" for d in df_pred["Divergencia_Detectada"].tolist()]
    if "Divergencia_Esperada" not in df_pred:
        raise ValueError("O arquivo de predições precisa conter a coluna 'Divergencia_Esperada'.")
    divergencias_esperadas = [str(d).upper() == "SIM" for d in df_pred["Divergencia_Esperada"].tolist()]
    justificativas_geradas = df_pred["Justificativa_IA"].fillna("").astype(str).tolist()
    evidencias_textos = df_pred["Evidencias_RAG"].fillna("").astype(str).tolist()
    gabarito_categorias = df_pred["Categoria_Esperada"].astype(str).tolist()
    predicoes_categorias = df_pred["Predicao_IA"].astype(str).tolist()

    res_extracao = calcular_taxa_identificacao_afirmacoes(afirmacoes_extraidas_todas, gabarito_afirmacoes)
    res_recuperacao = calcular_qualidade_fontes_e_cobertura(fontes_recuperadas_todas, respostas_meta_todas)
    res_divergencia = calcular_precisao_divergencias(divergencias_preditas, divergencias_esperadas)
    res_geracao = calcular_taxa_alucinacao(justificativas_geradas, evidencias_textos)
    res_classificacao = calcular_precisao_recall_f1(gabarito_categorias, predicoes_categorias)

    return RelatorioTecnicoCompleto(
        extracao=res_extracao,
        recuperacao=res_recuperacao,
        divergencia=res_divergencia,
        geracao=res_geracao,
        classificacao=res_classificacao
    )


def executar_testes_com_modelo_local(
    caminho_modelo: str = "./Qwen-FakeNews-Local",
    caminho_adapters: str = "adapters",
    caminho_excel: str = "Sprint_1_Dataset_20_exemplos_complexos.xlsx",
    aba: str = "Dataset Sprint 1",
    saida_excel: str = "Sprint_1_Resultados_Modelo_Treinado.xlsx"
) -> RelatorioTecnicoCompleto:
    """
    Carrega o modelo fine-tunado (Qwen2.5-7B + LoRA) e roda a avaliação
    completa em todas as 20 amostras isolando estritamente cada uma das 5 etapas.
    """
    print("\n" + "=" * 78)
    print("INICIANDO BENCHMARK COM O MODELO LOCAL FINE-TUNADO")
    print("=" * 78)

    if load is None or generate is None:
        raise RuntimeError("MLX LM não está instalado no ambiente.")

    adapter_path = caminho_adapters if os.path.exists(os.path.join(caminho_adapters, "adapters.safetensors")) else None

    print(f"1. Carregando modelo base de: {caminho_modelo}")
    if adapter_path:
        print(f"   -> Integrando adaptadores LoRA treinados de: {adapter_path}")
    else:
        print("   -> Rodando com pesos diretos do modelo.")

    modelo, tokenizer = load(caminho_modelo, adapter_path=adapter_path)
    print("   ✓ Modelo carregado com sucesso no Apple Silicon (MLX)!\n")

    print(f"2. Lendo planilha de teste: {caminho_excel} (Aba: {aba})")
    df = pd.read_excel(caminho_excel, sheet_name=aba)
    total_casos = len(df)
    print(f"   ✓ {total_casos} casos complexos carregados.\n")

    # Arrays para coleta isolada por etapa
    gabarito_afirmacoes = []
    afirmacoes_extraidas_todas = []
    fontes_recuperadas_todas = []
    respostas_meta_todas = []
    divergencias_preditas = []
    divergencias_esperadas = []
    justificativas_geradas = []
    evidencias_textos = []
    gabarito_categorias = []
    predicoes_categorias = []

    resultados_detalhados_linhas = []

    print(f"3. Executando inferência e avaliação isolada para cada um dos {total_casos} casos...\n")

    for idx, row in df.iterrows():
        caso_id = row.get('ID', idx + 1)
        mensagem = str(row['Mensagem de teste'])
        gabarito_afirmacao = str(row['Principal afirmação'])
        gabarito_cat = str(row['Categoria']).strip()
        gabarito_norm = normalizar_categoria(gabarito_cat)

        # Divergência esperada: Falsa, Fora de contexto e Sátira
        div_esperada = gabarito_norm in ["FALSA", "FORA DE CONTEXTO", "SÁTIRA"]

        print(f"[{idx+1:02d}/{total_casos:02d}] Processando Caso #{caso_id} | Gabarito: [{gabarito_norm}]")

        # -------------------------------------------------------------
        # ETAPA 1: Extração de Afirmações (Claim Extraction)
        # -------------------------------------------------------------
        messages_extracao = [
            {"role": "system", "content": PROMPT_EXTRAIR_AFIRMACOES},
            {"role": "user", "content": f"Mensagem a ser analisada:\n{mensagem}"}
        ]
        prompt_ext = tokenizer.apply_chat_template(messages_extracao, tokenize=False, add_generation_prompt=True)
        resp_ext = generate(modelo, tokenizer, prompt=prompt_ext, max_tokens=220, verbose=False)

        # Parse do JSON de afirmações
        extraidas_caso = []
        try:
            match_json = re.search(r'(\{.*\})', resp_ext, re.DOTALL)
            if match_json:
                dados_json = json.loads(match_json.group(1))
                for af in dados_json.get("afirmacoes", []):
                    if isinstance(af, dict) and "afirmacao" in af:
                        extraidas_caso.append(af["afirmacao"])
        except Exception:
            pass

        # Fallback caso o JSON venha incompleto
        if not extraidas_caso:
            primeira_linha = resp_ext.strip().split("\n")[0]
            extraidas_caso = [primeira_linha[:150]]

        # Termo para RAG: usa a afirmação extraída se disponível, senão a mensagem
        termo_busca = extraidas_caso[0] if extraidas_caso else mensagem[:120]

        # -------------------------------------------------------------
        # ETAPA 2: Recuperação de Evidências (RAG Search)
        # -------------------------------------------------------------
        texto_evidencias, fontes_lista = buscar_evidencias_web(termo_busca)

        # -------------------------------------------------------------
        # ETAPA 3 & 5: Verificação Factual, Divergência e Classificação
        # -------------------------------------------------------------
        prompt_verificacao = (
            f"<|im_start|>system\n{PROMPT_VERIFICACAO_SISTEMA}<|im_end|>\n"
            f"<|im_start|>user\nEVIDÊNCIAS DA WEB:\n{texto_evidencias}\n\n"
            f"MENSAGEM PARA ANÁLISE:\n{mensagem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        resp_verificacao = generate(modelo, tokenizer, prompt=prompt_verificacao, max_tokens=200, verbose=False)

        # Extração de Classificação
        match_cat = re.search(r'CLASSIFICAÇÃO:\s*([^\n]+)', resp_verificacao, re.IGNORECASE)
        cat_raw = match_cat.group(1).strip() if match_cat else resp_verificacao[:30]
        cat_pred = normalizar_categoria(cat_raw)

        # Extração de Divergência
        match_div = re.search(r'DIVERGÊNCIA:\s*(SIM|NÃO|NAO)', resp_verificacao, re.IGNORECASE)
        if match_div:
            div_detectada = "SIM" in match_div.group(1).upper()
        else:
            div_detectada = cat_pred in ["FALSA", "FORA DE CONTEXTO", "SÁTIRA"]

        # Extração de Justificativa
        match_just = re.search(r'JUSTIFICATIVA:\s*(.+)', resp_verificacao, re.IGNORECASE | re.DOTALL)
        justificativa = match_just.group(1).strip() if match_just else resp_verificacao.strip()

        # Armazena nos arrays de métricas
        gabarito_afirmacoes.append(gabarito_afirmacao)
        afirmacoes_extraidas_todas.append(extraidas_caso)
        fontes_recuperadas_todas.append(fontes_lista)
        respostas_meta_todas.append({"conclusiva": True})
        divergencias_preditas.append(div_detectada)
        divergencias_esperadas.append(div_esperada)
        justificativas_geradas.append(justificativa)
        evidencias_textos.append(texto_evidencias)
        gabarito_categorias.append(gabarito_norm)
        predicoes_categorias.append(cat_pred)

        # Registra para planilha detalhada
        acertou_classe = (cat_pred == gabarito_norm)
        resultados_detalhados_linhas.append({
            "ID": caso_id,
            "Categoria_Esperada": gabarito_norm,
            "Predicao_IA": cat_pred,
            "Classificacao_Acertou": "SIM" if acertou_classe else "NÃO",
            "Divergencia_Esperada": "SIM" if div_esperada else "NÃO",
            "Divergencia_Detectada": "SIM" if div_detectada else "NÃO",
            "Afirmacao_Gabarito": gabarito_afirmacao,
            "Afirmacao_Extraida_IA": " | ".join(extraidas_caso),
            "Mensagem_Original": mensagem,
            "Justificativa_IA": justificativa,
            "Evidencias_RAG": texto_evidencias
        })

    # Salva planilha detalhada
    df_export = pd.DataFrame(resultados_detalhados_linhas)
    df_export.to_excel(saida_excel, index=False)
    print(f"\n[✓] Execução de inferência concluída! Resultados salvos em '{saida_excel}'.\n")

    # -------------------------------------------------------------
    # CÁLCULO SEPARADO DE CADA UMA DAS 8 MÉTRICAS
    # -------------------------------------------------------------
    res_extracao = calcular_taxa_identificacao_afirmacoes(afirmacoes_extraidas_todas, gabarito_afirmacoes)
    res_recuperacao = calcular_qualidade_fontes_e_cobertura(fontes_recuperadas_todas, respostas_meta_todas)
    res_divergencia = calcular_precisao_divergencias(divergencias_preditas, divergencias_esperadas)
    res_geracao = calcular_taxa_alucinacao(justificativas_geradas, evidencias_textos)
    res_classificacao = calcular_precisao_recall_f1(gabarito_categorias, predicoes_categorias)

    return RelatorioTecnicoCompleto(
        extracao=res_extracao,
        recuperacao=res_recuperacao,
        divergencia=res_divergencia,
        geracao=res_geracao,
        classificacao=res_classificacao
    )


# ==============================================================================
# FORMATAÇÃO E EXIBIÇÃO EM TERMINAL
# ==============================================================================

def imprimir_relatorio_terminal(relatorio: RelatorioTecnicoCompleto):
    """Exibe o relatório formatado enfatizando a separação das etapas."""
    linhas = "=" * 78

    print("\n" + linhas)
    print("RELATÓRIO TÉCNICO DE MÉTRICAS DE AVALIAÇÃO (ETAPAS SEPARADAS)")
    print(linhas)
    print("Princípio: Avaliação desacoplada para impedir mascaramento de falhas.\n")

    # Etapa 1: Extração
    print("📌 ETAPA 1: EXTRAÇÃO DE AFIRMAÇÕES")
    print(f"  • Total de afirmações analisadas: {relatorio.extracao.total_afirmacoes_esperadas}")
    print(f"  • Afirmações identificadas corretamente: {relatorio.extracao.afirmacoes_corretamente_identificadas}")
    print(f"  👉 [MÉTRICA 4] Taxa de identificação correta das afirmações: {relatorio.extracao.taxa_identificacao_correta * 100:.2f}%")
    print("-" * 78)

    # Etapa 2: Recuperação de Fontes (RAG)
    print("📌 ETAPA 2: RECUPERAÇÃO DE EVIDÊNCIAS (RAG)")
    print(f"  • Total de fontes recuperadas: {relatorio.recuperacao.total_fontes_recuperadas}")
    print(f"  • Fontes de alta autoridade/credibilidade: {relatorio.recuperacao.fontes_alta_qualidade}")
    print(f"  👉 [MÉTRICA 5] Qualidade das fontes recuperadas: {relatorio.recuperacao.qualidade_fontes_score * 100:.2f}%")
    print(f"  • Casos com ausência total de evidência: {relatorio.recuperacao.casos_sem_evidencia}")
    print(f"  👉 [MÉTRICA 7] Taxa de respostas sem evidência: {relatorio.recuperacao.taxa_respostas_sem_evidencia * 100:.2f}%")
    print("-" * 78)

    # Etapa 3: Identificação de Divergências
    print("📌 ETAPA 3: DETECÇÃO DE DIVERGÊNCIAS (NLI / VERIFICAÇÃO FACTUAL)")
    print(f"  • Verdadeiros Positivos (Divergências reais encontradas): {relatorio.divergencia.true_positives_divergencia}")
    print(f"  • Falsos Positivos (Alarmes falsos de contradição): {relatorio.divergencia.false_positives_divergencia}")
    print(f"  👉 [MÉTRICA 8] Precisão na identificação de divergências: {relatorio.divergencia.precisao_identificacao_divergencias * 100:.2f}%")
    print(f"  • Recall na identificação de divergências: {relatorio.divergencia.recall_divergencias * 100:.2f}%")
    print(f"  • F1-Score na identificação de divergências: {relatorio.divergencia.f1_divergencias * 100:.2f}%")
    print("-" * 78)

    # Etapa 4: Fidelidade e Alucinação
    print("📌 ETAPA 4: SÍNTESE E JUSTIFICATIVA (FIDELIDADE TEXTUAL)")
    print(f"  • Total de frases/asserções avaliadas: {relatorio.geracao.total_afirmacoes_sintetizadas}")
    print(f"  • Asserções sustentadas pelas evidências: {relatorio.geracao.afirmacoes_suportadas_evidencia}")
    print(f"  • Asserções descoladas/alucinadas: {relatorio.geracao.afirmacoes_alucinadas}")
    print(f"  👉 [MÉTRICA 6] Taxa de alucinação: {relatorio.geracao.taxa_alucinacao * 100:.2f}%")
    print("-" * 78)

    # Etapa 5: Classificação Final
    print("📌 ETAPA 5: CLASSIFICAÇÃO CATEGÓRICA FINAL")
    print(f"  • Total de amostras testadas: {relatorio.classificacao.total_amostras}")
    print(f"  • Acurácia Global: {relatorio.classificacao.acuracia_global * 100:.2f}%")
    print(f"  👉 [MÉTRICA 1] Precisão (Macro): {relatorio.classificacao.precisao_macro * 100:.2f}% (Weighted: {relatorio.classificacao.precisao_weighted * 100:.2f}%)")
    print(f"  👉 [MÉTRICA 2] Recall (Macro): {relatorio.classificacao.recall_macro * 100:.2f}% (Weighted: {relatorio.classificacao.recall_weighted * 100:.2f}%)")
    print(f"  👉 [MÉTRICA 3] F1-Score (Macro): {relatorio.classificacao.f1_score_macro * 100:.2f}% (Weighted: {relatorio.classificacao.f1_score_weighted * 100:.2f}%)")
    print("\n  Detalhamento por Categoria:")
    for cat, metrics in relatorio.classificacao.por_classe.items():
        print(f"    - {cat:<16} | Precisão: {metrics['precisao']*100:>6.1f}% | Recall: {metrics['recall']*100:>6.1f}% | F1: {metrics['f1_score']*100:>6.1f}% | Suporte: {metrics['suporte']}")

    print(linhas + "\n")


def imprimir_relatorio_visual(relatorio: RelatorioTecnicoCompleto) -> None:
    """Resume as métricas técnicas em painéis, preservando o relatório CLI clássico."""
    cabecalho("Painel de métricas técnicas", "Avaliação separada por etapa do pipeline")
    tabela_metricas({
        "Acurácia global": relatorio.classificacao.acuracia_global,
        "F1 macro": relatorio.classificacao.f1_score_macro,
        "Extração correta": relatorio.extracao.taxa_identificacao_correta,
        "Qualidade das fontes": relatorio.recuperacao.qualidade_fontes_score,
        "Respostas sem evidência": relatorio.recuperacao.taxa_respostas_sem_evidencia,
        "F1 de divergências": relatorio.divergencia.f1_divergencias,
        "Taxa de alucinação": relatorio.geracao.taxa_alucinacao,
    }, "Resumo das 8 métricas")
    tabela = Table(title="Classificação por categoria", border_style="bright_blue")
    tabela.add_column("Categoria", style="bold")
    tabela.add_column("Precisão", justify="right")
    tabela.add_column("Recall", justify="right")
    tabela.add_column("F1", justify="right")
    tabela.add_column("Suporte", justify="right")
    for categoria, valores in relatorio.classificacao.por_classe.items():
        tabela.add_row(
            categoria,
            f"{valores['precisao'] * 100:.1f}%",
            f"{valores['recall'] * 100:.1f}%",
            f"{valores['f1_score'] * 100:.1f}%",
            str(int(valores['suporte'])),
        )
    console.print(tabela)


# ==============================================================================
# MAIN & CLI
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

def resolver_caminho(candidatos: List[str]) -> str:
    for c in candidatos:
        caminho = Path(c)
        c_abs = caminho if caminho.is_absolute() else ROOT_DIR / caminho
        if c_abs.exists():
            return str(c_abs)
    return str(ROOT_DIR / candidatos[0])

def main():
    parser = argparse.ArgumentParser(
        description="Módulo de Avaliação Técnica Separada (8 Métricas Técnicas)."
    )
    parser.add_argument(
        "--input_excel",
        type=str,
        default=resolver_caminho(["data/benchmark/Sprint_1_Dataset_20_exemplos_complexos.xlsx", "Sprint_1_Dataset_20_exemplos_complexos.xlsx"]),
        help="Caminho para o dataset Excel de teste"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=resolver_caminho(["models/Qwen-FakeNews-Local", "Qwen-FakeNews-Local"]),
        help="Caminho para o modelo fine-tunado"
    )
    parser.add_argument(
        "--adapters_path",
        type=str,
        default=resolver_caminho(["models/adapters", "adapters"]),
        help="Caminho para os adaptadores LoRA"
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        default=None,
        help="Arquivo opcional com predições pré-computadas para recalcular métricas sem re-executar LLM"
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        default=str(ROOT_DIR / "results/Sprint_1_Resultados_Modelo_Treinado.xlsx"),
        help="Caminho do arquivo Excel de saída com as predições"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(ROOT_DIR / "results/relatorio_metricas_tecnicas.json"),
        help="Caminho do arquivo JSON onde as métricas separadas serão salvas"
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Mostra um painel resumido e tabelas no terminal."
    )

    args = parser.parse_args()

    if args.predictions_file and os.path.exists(args.predictions_file):
        print(f"-> Modo Avaliação: recalculando métricas a partir do arquivo existente '{args.predictions_file}'...")
        relatorio = avaliar_predicoes_salvas(
            caminho_excel=args.input_excel,
            aba="Dataset Sprint 1",
            arquivo_predicoes=args.predictions_file
        )
    else:
        relatorio = executar_testes_com_modelo_local(
            caminho_modelo=args.model_path,
            caminho_adapters=args.adapters_path,
            caminho_excel=args.input_excel,
            aba="Dataset Sprint 1",
            saida_excel=args.output_excel
        )

    if args.visual:
        imprimir_relatorio_visual(relatorio)
    else:
        imprimir_relatorio_terminal(relatorio)

    if args.output_json:
        dados_json = {
            "metadados": {
                "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
                "origem_predicoes": str(args.predictions_file) if args.predictions_file else None,
                "modelo": args.model_path if not args.predictions_file else None,
                "adapters": args.adapters_path if not args.predictions_file else None,
                "dataset": args.input_excel,
                "metodologia": "Métricas calculadas separadamente por etapa do pipeline."
            },
            "extracao": asdict(relatorio.extracao),
            "recuperacao": asdict(relatorio.recuperacao),
            "divergencia": asdict(relatorio.divergencia),
            "geracao": asdict(relatorio.geracao),
            "classificacao": asdict(relatorio.classificacao)
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(dados_json, f, ensure_ascii=False, indent=2)
        print(f"[✓] Relatório estruturado completo salvo em: '{args.output_json}'\n")


if __name__ == "__main__":
    main()
