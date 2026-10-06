"""Métricas numéricas comuns para consultas e avaliações do RAG."""

from __future__ import annotations

import math

EVALUATION_NAMES = (
    "EVIDENCIAS_CONFLITANTES",
    "EVIDENCIAS_SUFICIENTES_FAVORAVEIS",
    "EVIDENCIAS_SUFICIENTES_CONTRARIAS",
    "EVIDENCIAS_INSUFICIENTES",
    "NAO_APLICAVEL",
)


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
        "resposta_valida": 1.0,
        "documentos_recuperados": float(len(items)),
        "evidencias_recuperadas": float(len(evidence)),
        "materiais_educativos_recuperados": float(len(teaching)),
        "documentos_citados": float(len(cited)),
        "evidencias_citadas": float(len(cited_evidence)),
        "fontes_independentes_citadas": float(len(groups)),
        "taxa_documentos_citados": len(cited) / len(items) if items else 0.0,
        "taxa_citacoes_validas": 1.0,
        "resposta_com_limitacoes": float(bool(str(answer.get("limitacoes", "")).strip())),
        "evidencias_favoraveis": float(len(answer.get("evidencias_favoraveis", []))),
        "evidencias_contrarias": float(len(answer.get("evidencias_contrarias", []))),
        "pontos_de_atencao": float(len(answer.get("pontos_de_atencao", []))),
    }
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
