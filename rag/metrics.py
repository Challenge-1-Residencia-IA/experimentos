"""Métricas numéricas comuns para consultas e avaliações do RAG."""

from __future__ import annotations

import math
import re

EVALUATION_NAMES = (
    "EVIDENCIAS_CONFLITANTES",
    "EVIDENCIAS_SUFICIENTES_FAVORAVEIS",
    "EVIDENCIAS_SUFICIENTES_CONTRARIAS",
    "EVIDENCIAS_INSUFICIENTES",
    "NAO_APLICAVEL",
)


def citation_metrics(items: list[dict], answer: dict) -> dict[str, float]:
    """Mede referências resolvidas; não mede se a fonte prova a alegação."""
    available = {item.get("numero") for item in items if item.get("tipo") == "evidencia"}
    references = []
    for field in ("evidencias_favoraveis", "evidencias_contrarias"):
        for text in answer.get(field, []):
            for group in re.findall(r"\[([\d,\s]+)\]", text):
                references.extend(int(part.strip()) if part.strip().isdigit() else None
                                  for part in group.split(","))
    valid = sum(number in available and number is not None for number in references)
    metrics = {"citacoes_avaliadas": float(len(references)),
               "citacoes_resolvidas": float(valid),
               "resposta_com_citacoes": float(bool(references))}
    if references:
        metrics["taxa_citacoes_validas"] = valid / len(references)
    return metrics


def answer_metrics(items: list[dict], answer: dict, context: dict | None = None) -> dict[str, float]:
    """Extrai somente métricas objetivas que não exigem gabarito humano."""
    evidence = [item for item in items if item.get("tipo") == "evidencia"]
    teaching = [item for item in items if item.get("tipo") == "educativo"]
    cited = answer.get("fontes", [])
    cited_evidence = [item for item in cited if item.get("tipo") == "evidencia"]
    groups = {item.get("grupo_origem") for item in cited_evidence if item.get("grupo_origem")}
    distances = [float(item["distancia_vetorial"]) for item in items
                 if isinstance(item.get("distancia_vetorial"), (int, float))]
    evaluation = str(answer.get("avaliacao_evidencias", ""))
    metrics = {
        "resposta_valida": float(
            evaluation in EVALUATION_NAMES and isinstance(answer.get("afirmacao"), str)
            and bool(answer["afirmacao"].strip())
            and all(isinstance(answer.get(field), list) and
                    all(isinstance(value, str) for value in answer[field])
                    for field in ("evidencias_favoraveis", "evidencias_contrarias"))
        ),
        "documentos_recuperados": float(len(items)),
        "evidencias_recuperadas": float(len(evidence)),
        "materiais_educativos_recuperados": float(len(teaching)),
        "documentos_citados": float(len(cited)),
        "evidencias_citadas": float(len(cited_evidence)),
        "fontes_independentes_citadas": float(len(groups)),
        "taxa_documentos_citados": len(cited) / len(items) if items else 0.0,
        "resposta_com_limitacoes": float(bool(str(answer.get("limitacoes", "")).strip())),
        "evidencias_favoraveis": float(len(answer.get("evidencias_favoraveis", []))),
        "evidencias_contrarias": float(len(answer.get("evidencias_contrarias", []))),
        "pontos_de_atencao": float(len(answer.get("pontos_de_atencao", []))),
    }
    metrics.update(citation_metrics(items, answer))
    for name in EVALUATION_NAMES:
        metrics[f"avaliacao_{name.casefold()}"] = float(evaluation == name)
    if distances:
        metrics.update({
            "distancia_vetorial_min": min(distances),
            "distancia_vetorial_media": sum(distances) / len(distances),
            "distancia_vetorial_max": max(distances),
        })
    execution = context or {}
    research = execution.get("pesquisa_web") or {}
    for source, target in (
        ("resultados", "resultados_web"),
        ("paginas_lidas", "paginas_web_lidas"),
        ("fontes_qualificadas", "fontes_web_qualificadas"),
        ("evidencias_locais", "evidencias_locais_adicionadas"),
    ):
        if isinstance(research.get(source), (int, float)):
            metrics[target] = float(research[source])
    timings = execution.get("metricas_execucao") or {}
    metrics.update({name: float(value) for name, value in timings.items()
                    if isinstance(value, (int, float)) and math.isfinite(float(value))})
    return metrics
