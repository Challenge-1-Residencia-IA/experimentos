"""Resposta educativa com evidências e validação de citações."""

from __future__ import annotations

import json
import os
import re

import requests
from dotenv import load_dotenv

from rag.corpus import ROOT
from rag.store import VectorStore

SYSTEM_PROMPT = (ROOT / "prompts" / "tutora_rag.txt").read_text(encoding="utf-8")
CONCLUSIVE = {"VERDADEIRA", "FALSA", "ENGANOSA"}
VERDICTS = CONCLUSIVE | {"INCONCLUSIVA"}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "como_verificar": {"type": "string"},
        "analise": {"type": "string"},
        "veredito": {"type": "string", "enum": sorted(VERDICTS)},
        "justificativa": {"type": "string"},
        "fontes_usadas": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["como_verificar", "analise", "veredito", "justificativa", "fontes_usadas"],
}


def ask(
    question: str,
    store: VectorStore | None = None,
    *,
    evidence: list[dict] | None = None,
    teaching: list[dict] | None = None,
) -> dict:
    load_dotenv(ROOT / ".env")
    store = store or VectorStore()
    if evidence is None:
        evidence = store.search(question, "evidencia", limit=6, max_per_source=2)
    if teaching is None:
        teaching = store.search(question, "educativo", limit=3, max_per_source=2)
    items = []
    for index, item in enumerate(evidence + teaching, 1):
        metadata = item["metadata"]
        items.append({
            "numero": index,
            "tipo": metadata["kind"],
            "titulo": metadata["title"],
            "origem": metadata["source"],
            "grupo_origem": metadata["origin_group"],
            "url": metadata["url"],
            "data": metadata.get("published_at", ""),
            "trecho": item["text"],
        })
    prompt = json.dumps({"pergunta": question, "documentos": items}, ensure_ascii=False)
    endpoint = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    response = requests.post(
        endpoint,
        json={
            "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            "stream": False,
            "format": ANSWER_SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = json.loads(response.json()["message"]["content"])
    return validate_answer(payload, items)


def validate_answer(payload: dict, items: list[dict]) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("O modelo não retornou um objeto JSON")
    for field in ("como_verificar", "analise", "veredito", "justificativa", "fontes_usadas"):
        if field not in payload:
            raise ValueError(f"Campo ausente na resposta: {field}")
    if payload["veredito"] not in VERDICTS:
        raise ValueError("Veredito inválido")
    if not isinstance(payload["fontes_usadas"], list):
        raise ValueError("fontes_usadas deve ser uma lista")
    indices = payload["fontes_usadas"]
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 1 or index > len(items) for index in indices):
        raise ValueError("A resposta citou um documento inexistente")
    for field in ("como_verificar", "analise", "justificativa"):
        if not isinstance(payload[field], str):
            raise ValueError(f"{field} deve ser texto")
    cited_in_text = {int(value) for value in re.findall(r"\[(\d+)\]", payload["analise"] + " " + payload["justificativa"])}
    if not cited_in_text.issubset(set(indices)):
        raise ValueError("A resposta contém citação ausente de fontes_usadas")
    if not any(item["tipo"] == "evidencia" for item in items):
        payload["veredito"] = "INCONCLUSIVA"
        payload["como_verificar"] = (
            "Localize a publicação original, confira autoria e data e identifique quais provas ela apresenta. "
            "Procure registros primários e checagens independentes sobre a afirmação específica. "
            "Avalie o contexto completo; imagens isoladas e diferenças de aparência não bastam como prova."
        )
        payload["analise"] = (
            "A base consultada não trouxe documentos de evidência sobre esta alegação. "
            "Isso não a comprova e não coloca as hipóteses em pé de igualdade."
        )
        payload["justificativa"] = (
            "Não há suporte verificável nesta base para confirmar a alegação. "
            "O veredito indica uma limitação da consulta local; para classificar o fato, "
            "é preciso examinar a origem da publicação e checagens ou registros independentes."
        )
        payload["fontes_usadas"] = [index for index in indices if items[index - 1]["tipo"] == "educativo"]
    if payload["veredito"] in CONCLUSIVE:
        groups = {
            items[index - 1]["grupo_origem"]
            for index in indices
            if items[index - 1]["tipo"] == "evidencia" and index in cited_in_text
        }
        if len(groups) < 2:
            payload["veredito"] = "INCONCLUSIVA"
            payload["justificativa"] = (
                "As evidências citadas não incluem duas origens independentes. "
                "É necessário buscar e comparar mais fontes antes de concluir."
            )
            payload["fontes_usadas"] = sorted(cited_in_text)
    payload["fontes"] = [items[index - 1] for index in sorted(set(payload["fontes_usadas"]))]
    payload["limite_da_validacao"] = (
        "A validação confere formato, referências e diversidade declarada de origens; "
        "o suporte factual das citações requer avaliação humana."
    )
    return payload
