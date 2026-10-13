"""Cliente da ponte RAG local e apresentação curta para WhatsApp."""

from __future__ import annotations

import asyncio
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class BridgeError(RuntimeError):
    """A ponte não produziu uma análise utilizável."""

    def __init__(self, message: str, code: str = "BRIDGE_ERROR"):
        super().__init__(message)
        self.code = code


BRIDGE_URL = os.getenv("RAG_BRIDGE_URL", "http://127.0.0.1:8765").rstrip("/")
if urlsplit(BRIDGE_URL).hostname != "127.0.0.1":
    raise ValueError("RAG_BRIDGE_URL deve apontar para 127.0.0.1")
BRIDGE_TIMEOUT = float(os.getenv("RAG_BRIDGE_TIMEOUT", "200"))
_CITATION = re.compile(r"\[([\d,\s]+)\]")


def _clickable_url(value: object) -> str | None:
    if not isinstance(value, str) or any(char.isspace() for char in value):
        return None
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    return value if parts.scheme in {"http", "https"} and parts.hostname else None


def _request(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(
        BRIDGE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=BRIDGE_TIMEOUT) as response:
            result = json.load(response)
    except HTTPError as error:
        try:
            details = json.load(error)
            code = details.get("error", {}).get("code", "HTTP_ERROR")
        except (ValueError, AttributeError):
            code = "HTTP_ERROR"
        raise BridgeError(f"Ponte RAG indisponível: {code}", code=code) from error
    except (URLError, TimeoutError, ValueError) as error:
        raise BridgeError("Ponte RAG indisponível ou resposta inválida") from error
    if not isinstance(result, dict):
        raise BridgeError("Resposta da ponte não é um objeto")
    return result


async def health() -> bool:
    try:
        result = await asyncio.to_thread(_request, "/health")
        return result.get("status") == "ready"
    except BridgeError:
        return False


async def analyze(request_id: str, message: str, history: list[dict] | None = None) -> dict:
    result = await asyncio.to_thread(
        _request,
        "/analyze",
        {"request_id": request_id, "message": message, "history": (history or [])[-2:]},
    )
    if result.get("request_id") != request_id or not isinstance(result.get("answer"), dict):
        raise BridgeError("Resposta da ponte não corresponde à solicitação")
    return result["answer"]


def _format_one(answer: dict) -> str:
    body = (answer.get("resposta_usuario") or answer.get("sintese") or "").strip()
    if not body or body.startswith("[Simulação]"):
        raise BridgeError("Análise vazia ou de simulação")
    citations = {int(value) for group in _CITATION.findall(body)
                 for value in group.split(",") if value.strip().isdigit()}
    sources = {source.get("numero"): source for source in answer.get("fontes", [])
               if isinstance(source, dict) and source.get("url")}
    if citations - sources.keys():
        raise BridgeError("A análise cita fontes sem URL validada")
    listed_urls: set[str] = set()
    sections: list[str] = []
    lines = []
    for number in sorted(citations):
        source = sources[number]
        url = _clickable_url(source.get("url"))
        if not url:
            raise BridgeError("A análise cita fonte sem link HTTP válido")
        title = source.get("titulo") or source.get("origem") or "Fonte"
        origin = source.get("origem") or ""
        date = source.get("data") or ""
        label = " — ".join(str(part).replace("\n", " ") for part in (title, origin, date) if part)
        lines.append(f"[{number}] {label}\n{url}")
        listed_urls.add(url)
    if lines:
        sections.append("Fontes citadas na análise:\n" + "\n".join(lines))
    if answer.get("avaliacao_evidencias") == "EVIDENCIAS_INSUFICIENTES":
        research = answer.get("pesquisa_web")
        if isinstance(research, dict):
            found = research.get("resultados")
            read = research.get("paginas_lidas")
            if isinstance(found, int) and isinstance(read, int):
                sections.append(f"Nesta busca: {found} resultados, {read} páginas lidas.")
            if research.get("erro_busca"):
                sections.append("A busca web teve uma falha; esta análise pode estar incompleta.")

    def add_group(label: str, entries: object) -> None:
        lines = []
        if not isinstance(entries, list):
            return
        for source in entries:
            if not isinstance(source, dict):
                continue
            url = _clickable_url(source.get("url"))
            if not url or url in listed_urls:
                continue
            listed_urls.add(url)
            title = str(source.get("titulo") or source.get("origem") or source.get("veiculo") or "Link")
            lines.append(f"{title.replace(chr(10), ' ').replace(chr(13), ' ')}\n{url}")
        if lines:
            sections.append(label + ":\n" + "\n".join(lines))

    supplied = answer.get("fonte_enviada")
    add_group("Publicação enviada", [supplied] if isinstance(supplied, dict) else [])
    add_group("Páginas consultadas (não comprovam a alegação)", answer.get("fontes_encontradas"))
    add_group("Páginas de outros assuntos (não comprovam a alegação)", answer.get("fontes_descartadas"))
    add_group("Checagens candidatas (não são prova automática)", answer.get("checagens_encontradas"))
    research = answer.get("pesquisa_web")
    if isinstance(research, dict):
        add_group("Páginas não lidas", research.get("paginas_ignoradas"))
        add_group("Resultados da busca não verificados",
                  [{"url": url} for url in research.get("links_resultados", [])]
                  if isinstance(research.get("links_resultados"), list) else [])
    return body + ("\n\n" + "\n\n".join(sections) if sections else "")


def format_answer(answer: dict) -> str:
    analyses = answer.get("analises")
    if isinstance(analyses, list) and analyses:
        parts = []
        for index, item in enumerate(analyses, start=1):
            if not isinstance(item, dict):
                raise BridgeError("Análise múltipla incompleta")
            claim = (item.get("alegacao_verificada") or item.get("afirmacao") or "").strip()
            heading = f"Afirmação {index}: {claim[:160]}\n" if claim else f"Afirmação {index}\n"
            parts.append(heading + _format_one(item))
        if len(parts) != len(analyses):
            raise BridgeError("Análise múltipla incompleta")
        return "Análise por afirmação:\n\n" + "\n\n".join(parts)
    return _format_one(answer)
