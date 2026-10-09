"""Busca gratuita e leitura limitada de páginas para a conversa RAG."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from ddgs import DDGS
from pypdf import PdfReader
from dotenv import load_dotenv

from rag.corpus import chunks, stable_id
from rag.relevance import relevance
from rag.cache import ResearchCache

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
MAX_BYTES = 5_000_000
MAX_PAGES = 6
SEARCH_RESULTS = 12
USER_AGENT = "NexoVerdict/0.2 (educational fact checking)"
POLICY_PATH = Path(__file__).resolve().parents[1] / "web_sources.json"
MULTIPART_SUFFIXES = {"com.br", "gov.br", "org.br", "jus.br", "leg.br", "co.uk", "com.au"}


def repair_mojibake(value: str) -> str:
    """Corrige sequências UTF-8 interpretadas como Latin-1, sem alterar texto íntegro."""
    def fix_word(match: re.Match[str]) -> str:
        word = match.group()
        if not any(marker in word for marker in ("Ã", "Â", "â€")):
            return word
        try:
            fixed = word.encode("cp1252").decode("utf-8")
        except (UnicodeError, LookupError):
            return word
        return fixed if fixed.count("Ã") + fixed.count("Â") < word.count("Ã") + word.count("Â") else word

    return re.sub(r"\S+", fix_word, value)


def decode_page(raw: bytes, declared_encoding: str | None) -> str:
    """Páginas UTF-8 às vezes anunciam ISO-8859-1; UTF-8 válido tem prioridade."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode(declared_encoding or "cp1252", errors="replace")


@lru_cache(maxsize=1)
def source_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def matches_domain(url: str, domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def split_input(message: str) -> tuple[str, str | None]:
    match = URL_PATTERN.search(message)
    if not match:
        return message.strip(), None
    url = match.group().rstrip(".,;:!?)]}")
    return (message[:match.start()] + message[match.end():]).strip(), url


def origin_group(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if host == "ibge.gov.br" or host.endswith(".ibge.gov.br"):
        return "ibge"
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in MULTIPART_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def check_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Informe uma URL pública HTTP(S) válida")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Porta da URL não permitida")
    host = parsed.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise ValueError("Endereço local não permitido")
    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                       type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ValueError(f"Não foi possível resolver o domínio: {host}") from error
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("A URL resolve para endereço não público")


def read_page(url: str) -> dict:
    """Baixa HTML, PDF ou texto. Confere URLs a cada redirecionamento."""
    current = url
    for _ in range(5):
        check_public_url(current)
        if matches_domain(current, source_policy()["blocked_automated_domains"]):
            raise ValueError("Coleta automatizada não habilitada para este publicador")
        response = requests.get(current, headers={"User-Agent": USER_AGENT}, timeout=(5, 15),
                                allow_redirects=False, stream=True)
        if response.is_redirect:
            target = urljoin(current, response.headers.get("Location", ""))
            response.close()
            if target == current:
                raise ValueError("Redirecionamento inválido")
            current = target
            continue
        try:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if not any(value in content_type for value in ("text/html", "application/pdf", "text/plain")):
                raise ValueError(f"Tipo de página não suportado: {content_type or 'desconhecido'}")
            parts = []
            size = 0
            for part in response.iter_content(chunk_size=65536):
                size += len(part)
                if size > MAX_BYTES:
                    raise ValueError("Página maior que 5 MB")
                parts.append(part)
            raw = b"".join(parts)
            title = urlparse(current).path.rsplit("/", 1)[-1] or origin_group(current)
            date = ""
            if "pdf" in content_type:
                reader = PdfReader(BytesIO(raw))
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            elif "html" in content_type:
                html = decode_page(raw, response.encoding)
                text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
                metadata = trafilatura.extract_metadata(html)
                if metadata:
                    title = metadata.title or title
                    date = metadata.date or ""
            else:
                text = decode_page(raw, response.encoding)
            title = repair_mojibake(title)
            text = repair_mojibake(text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) < 100:
                raise ValueError("A página não forneceu texto suficiente para checagem")
            return {"url": current, "title": title, "date": date, "text": text[:30000],
                    "origin_group": origin_group(current)}
        finally:
            response.close()
    raise ValueError("Redirecionamentos em excesso")


class WebResearcher:
    def __init__(self) -> None:
        self.pages = ResearchCache()
        self.searches = ResearchCache()
        self.answer_cache: dict[tuple, tuple[float, dict, list[dict]]] = {}
        self.fact_check_cache = ResearchCache()

    def fact_checks(self, query: str) -> tuple[list[dict], str]:
        """Consulta checagens publicadas; resultado é candidato, não prova automática."""
        cached = self.fact_check_cache.get(query)
        if cached is not None:
            return cached
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        key = os.getenv("FACT_CHECK_API_KEY", "").strip()
        if not key or key.casefold() in {"sua_chave_aqui", "your_api_key", "..."}:
            result = ([], "chave_nao_configurada")
        else:
            try:
                response = requests.get(
                    "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                    params={"query": query[:300], "languageCode": "pt", "pageSize": 10, "key": key},
                    timeout=(5, 10),
                )
                response.raise_for_status()
                checks = []
                for claim in response.json().get("claims", []):
                    if not isinstance(claim, dict):
                        continue
                    for review in claim.get("claimReview", []):
                        if not isinstance(review, dict) or not review.get("url"):
                            continue
                        checks.append({
                            "alegacao_checada": claim.get("text", ""),
                            "titulo": review.get("title", ""),
                            "url": review["url"],
                            "veiculo": review.get("publisher", {}).get("name", ""),
                            "avaliacao_do_veiculo": review.get("textualRating", ""),
                            "data_revisao": review.get("reviewDate", ""),
                        })
                result = (checks, "ok")
            except requests.RequestException as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                result = ([], f"http_{status}" if status else type(error).__name__)
            except (ValueError, TypeError, AttributeError, KeyError):
                result = ([], "resposta_invalida")
        if result[1] == "ok":
            self.fact_check_cache.put(query, result)
        return result

    def page(self, url: str) -> dict:
        cached = self.pages.get(url)
        if cached is not None:
            return cached
        page = read_page(url)
        self.pages.put(url, page)
        return page

    def search(self, query: str) -> list[dict]:
        query = query.strip()[:300]
        if not query:
            return []
        cached = self.searches.get(query)
        if cached is not None:
            return cached
        results = DDGS(timeout=10).text(
            query, region="br-pt", safesearch="moderate", max_results=SEARCH_RESULTS
        ) or []
        self.searches.put(query, results)
        return results

    def research(self, query: str, *, exclude_url: str | None = None,
                 exclude_group: str | None = None) -> tuple[list[dict], dict]:
        diagnostics = {"busca": query, "consultas": [], "resultados": 0, "paginas_lidas": 0,
                       "fontes_qualificadas": 0,
                       "paginas_ignoradas": [], "erro_busca": ""}
        documents = []
        groups = set()
        preferred = source_policy()["conclusive_domains"]
        for query_number, search_query in enumerate((query, f"{query} checagem de fatos")):
            if len(documents) >= MAX_PAGES or diagnostics["fontes_qualificadas"] >= 2:
                break
            page_limit = MAX_PAGES - 2 if query_number == 0 else MAX_PAGES
            try:
                results = self.search(search_query)
            except Exception as error:  # Falhas externas do buscador não impedem usar outras fontes.
                diagnostics["erro_busca"] = f"{type(error).__name__}: {str(error)[:160]}"
                continue
            diagnostics["consultas"].append(search_query[:300])
            diagnostics["resultados"] += len(results)
            results = sorted(
                results,
                key=lambda item: (
                    {"direta": 2, "relacionada": 1, "nao_relacionada": 0}[
                        relevance(query, item.get("title", ""), item.get("body", ""))[0]
                    ],
                    relevance(query, item.get("title", ""), item.get("body", ""))[1],
                    matches_domain(item.get("href") or item.get("url") or "", preferred),
                ),
                reverse=True,
            )
            candidate_urls = [result.get("href") or result.get("url") or "" for result in results]
            candidate_urls = list(dict.fromkeys(
                url for url in candidate_urls if url and url != exclude_url
                and origin_group(url) != exclude_group and origin_group(url) not in groups
            ))
            # Limita downloads simultâneos sem criar mais páginas que o teto da busca.
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {url: executor.submit(self.page, url) for url in candidate_urls[:MAX_PAGES]}
                for url in candidate_urls:
                    if len(documents) >= page_limit or diagnostics["fontes_qualificadas"] >= 2:
                        break
                    if url not in futures:
                        futures[url] = executor.submit(self.page, url)
                    try:
                        page = futures[url].result()
                    except Exception as error:  # Parser e sites externos podem falhar de formas diferentes.
                        diagnostics["paginas_ignoradas"].append(
                            {"url": url, "motivo": f"{type(error).__name__}: {str(error)[:100]}"}
                        )
                        continue
                    if page["origin_group"] in groups or page["origin_group"] == exclude_group or page["url"] == exclude_url:
                        continue
                    groups.add(page["origin_group"])
                    documents.append(page)
                    if (matches_domain(page["url"], preferred) and
                            relevance(query, page["title"], page["text"][:1200])[0] == "direta"):
                        diagnostics["fontes_qualificadas"] += 1
        diagnostics["paginas_lidas"] = len(documents)
        return documents, diagnostics


def passages(query: str, documents: list[dict], encoder, *, limit: int = 6) -> list[dict]:
    """Seleciona um trecho relevante por página, mantendo origens distintas."""
    if not documents:
        return []
    parts = []
    for page in documents:
        for part in chunks(page["text"], max_words=160, overlap=25)[:80]:
            parts.append((page, part))
    query_vector = encoder.encode([f"query: {query}"], normalize_embeddings=True)[0]
    vectors = encoder.encode([f"passage: {part}" for _, part in parts],
                             normalize_embeddings=True, batch_size=32)
    scored = sorted(((float(query_vector @ vector), page, part)
                     for (page, part), vector in zip(parts, vectors)),
                    reverse=True,
                    key=lambda item: (
                        {"direta": 2, "relacionada": 1, "nao_relacionada": 0}[
                            relevance(query, item[1]["title"], item[2])[0]
                        ],
                        relevance(query, item[1]["title"], item[2])[1],
                        item[0],
                    ))
    selected = []
    groups = set()
    for score, page, part in scored:
        group = page["origin_group"]
        if group in groups:
            continue
        groups.add(group)
        selected.append({"id": stable_id(page["url"] + part), "text": part,
                         "distance": 1 - score,
                         "metadata": {"kind": "evidencia", "source": group,
                                      "origin_group": group, "title": page["title"],
                                      "url": page["url"], "published_at": page["date"],
                                      "conclusive_eligible": matches_domain(page["url"], source_policy()["conclusive_domains"]),
                                      "relevancia": relevance(query, page["title"], part)[0]}})
        if len(selected) >= limit:
            break
    return selected
