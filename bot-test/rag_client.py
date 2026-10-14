"""Cliente assíncrono da ponte local e formatação conservadora de fontes."""
import asyncio
import re
from urllib.parse import urlsplit

import httpx


class BridgeError(RuntimeError):
    pass


def link(value):
    if not isinstance(value, str) or any(char.isspace() for char in value):
        return ""
    try:
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password:
            return value
    except ValueError:
        pass
    return ""


def format_analysis(analysis):
    if not isinstance(analysis, dict):
        raise BridgeError("invalid_analysis")
    if isinstance(analysis.get("analises"), list) and analysis["analises"]:
        return "\n\n".join(f"Alegação {i}\n{format_analysis(item)}" for i, item in enumerate(analysis["analises"], 1))
    sources = {}
    for field in ("fontes", "fontes_encontradas", "fontes_consultadas", "fontes_descartadas"):
        values = analysis.get(field, [])
        if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
            raise BridgeError("invalid_sources")
        sources[field] = values
    text = analysis.get("resposta_usuario") or analysis.get("sintese")
    if not isinstance(text, str) or not text.strip():
        pieces = [analysis.get("veredito"), analysis.get("contexto")]
        # Achados do fallback podem conter [n] de páginas não citadas em fontes.
        # Não promovê-los a evidência nem resolver esses números por consultadas.
        for field in ("evidencias_favoraveis", "evidencias_contrarias"):
            if isinstance(analysis.get(field), list):
                pieces.extend(analysis[field])
        text = "\n\n".join(item.strip() for item in pieces if isinstance(item, str) and item.strip())
    if not text or "[Simulação]" in text:
        raise BridgeError("empty_analysis")
    cited = set()
    for group in re.findall(r"\[([\d,\s]+)\]", text):
        cited.update(int(number.strip()) for number in group.split(",") if number.strip())
    mapped = {source.get("numero"): source for source in sources["fontes"]}
    if any(number not in mapped or not link(mapped[number].get("url")) for number in cited):
        raise BridgeError("unresolved_citation")
    known = {link(source.get("url")) for items in sources.values() for source in items} - {""}
    if any(url.rstrip(".,;:!?)]}") not in known for url in re.findall(r"https?://[^\s<>]+", text)):
        raise BridgeError("unverified_url_in_text")
    parts, seen = [text.strip()], set()
    for number in sorted(cited):
        source = mapped[number]
        url = link(source.get("url"))
        title = " — ".join(str(source[key]) for key in ("titulo", "origem", "data") if source.get(key))
        parts.append(f"[{number}] {title or 'Fonte citada'}\n{url}")
        seen.add(url)
    for label, fields in (
        ("Consultadas (não são prova):", ("fontes_encontradas", "fontes_consultadas")),
        ("Descartadas (não sustentam a análise):", ("fontes_descartadas",)),
    ):
        urls = []
        for field in fields:
            for source in sources[field]:
                url = link(source.get("url"))
                if url and url not in seen:
                    urls.append(url)
                    seen.add(url)
        if urls:
            parts.append(label + "\n" + "\n".join(urls))
    parts.append("Posso ajudar a aprofundar uma alegação ou conferir uma fonte específica.")
    return "\n\n".join(parts)


class RagClient:
    def __init__(self, base_url="http://127.0.0.1:8765", timeout=200):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment or not 10 <= timeout <= 660):
            raise ValueError("Ponte deve ser HTTP local; timeout entre 10 e 660 segundos")
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout,
                                     trust_env=False, follow_redirects=False)
        self.timeout = timeout

    async def health(self):
        try:
            response = await self.http.get("/health", timeout=10)
            payload = response.json()
            if response.status_code != 200 or payload.get("ready") is not True or payload.get("schema_version") != 1:
                raise BridgeError("not_ready")
        except (httpx.HTTPError, ValueError, AttributeError) as error:
            raise BridgeError("bridge_unavailable") from error

    async def analyze(self, message, request_id):
        try:
            response = await asyncio.wait_for(self.http.post("/analyze", json={"request_id": request_id, "message": message, "history": []}), timeout=self.timeout)
            payload = response.json()
            if response.status_code != 200:
                raise BridgeError(f"bridge_http_{response.status_code}")
            if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                    or payload.get("request_id") != request_id or "error" in payload):
                raise BridgeError("invalid_contract")
            return format_analysis(payload.get("analysis"))
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError) as error:
            raise BridgeError("bridge_unavailable_or_invalid") from error

    async def close(self):
        await self.http.aclose()
