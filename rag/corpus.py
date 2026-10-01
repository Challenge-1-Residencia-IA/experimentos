"""Coleta e normalização de documentos para o RAG."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "documents.jsonl"
USER_AGENT = "NexoVerdict-RAG/0.1 (document research)"
SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".html", ".htm"}


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def clean_text(value: str) -> str:
    value = value.replace("\x00", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def read_corpus(path: Path = CORPUS) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["id"]: row for line in handle if (row := json.loads(line))}


def write_corpus(documents: dict[str, dict], path: Path = CORPUS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for document in sorted(documents.values(), key=lambda item: item["id"]):
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    temporary.replace(path)


def extract_local(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Formato não suportado: {path}")
    if suffix == ".pdf":
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        if not any(page.strip() for page in pages):
            raise ValueError(f"PDF sem texto extraível; faça OCR antes: {path}")
        text = "\n\n".join(f"[Página {number}]\n{page}" for number, page in enumerate(pages, 1))
        return clean_text(text), path.stem
    raw = path.read_text(encoding="utf-8")
    if suffix in {".html", ".htm"}:
        return extract_html(raw)
    title = next((line.lstrip("# ").strip() for line in raw.splitlines() if line.strip()), path.stem)
    return clean_text(raw), title


def extract_html(raw: str) -> tuple[str, str]:
    soup = BeautifulSoup(raw, "html.parser")
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else "Documento web"
    for node in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        node.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    for node in main.find_all(["h1", "h2", "h3", "p", "li", "blockquote", "tr"]):
        node.append("\n")
    return clean_text(main.get_text(" ", strip=True)), title


def html_publication_date(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for key, value in (
        ("property", "article:published_time"),
        ("name", "date"),
        ("name", "pubdate"),
        ("itemprop", "datePublished"),
    ):
        tag = soup.find("meta", attrs={key: value})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return ""


def download(url: str, timeout: int = 30) -> tuple[str, str, str, str]:
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"URL HTTP(S) exigida: {url}")
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    if len(response.content) > 15_000_000:
        raise ValueError(f"Documento maior que 15 MB: {url}")
    content_type = response.headers.get("content-type", "").lower()
    published_at = ""
    if "pdf" in content_type or urlparse(response.url).path.lower().endswith(".pdf"):
        from io import BytesIO

        reader = PdfReader(BytesIO(response.content))
        pages = [page.extract_text() or "" for page in reader.pages]
        if not any(page.strip() for page in pages):
            raise ValueError(f"PDF sem texto extraível; faça OCR antes: {url}")
        text = "\n\n".join(f"[Página {number}]\n{page}" for number, page in enumerate(pages, 1))
        extracted, title, suffix = clean_text(text), Path(urlparse(response.url).path).stem, ".pdf"
    elif "html" in content_type:
        extracted, title = extract_html(response.text)
        published_at = html_publication_date(response.text)
        suffix = ".html"
    elif "text/plain" in content_type or "text/markdown" in content_type:
        extracted, title = clean_text(response.text), Path(urlparse(response.url).path).stem
        suffix = ".md" if "markdown" in content_type else ".txt"
    else:
        raise ValueError(f"Tipo de conteúdo não suportado ({content_type}): {url}")
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{hashlib.sha256(response.content).hexdigest()}{suffix}"
    if not raw_path.exists():
        raw_path.write_bytes(response.content)
    return extracted, title, str(raw_path.relative_to(ROOT)), published_at


def discover(config: dict) -> list[dict]:
    entries = list(config.get("documents", []))
    for feed in config.get("feeds", []):
        if feed.get("usage_approved") is not True:
            raise ValueError(f"Confirme uso permitido do feed com usage_approved=true: {feed['url']}")
        if urlparse(feed["url"]).scheme not in {"http", "https"}:
            raise ValueError(f"URL HTTP(S) exigida para feed: {feed['url']}")
        response = requests.get(feed["url"], headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
        if parsed.get("bozo") and not parsed.entries:
            raise ValueError(f"Não foi possível ler o feed: {feed['url']}")
        for item in parsed.entries[: int(feed.get("limit", 20))]:
            entries.append({
                "url": item.link,
                "kind": feed.get("kind", "evidencia"),
                "source": feed["source"],
                "origin_group": feed["origin_group"],
                "license": feed.get("license", "verificar"),
                "usage_approved": feed.get("usage_approved", False),
                "published_at": item.get("published", ""),
            })
    return entries


def ingest(config_path: Path, corpus_path: Path = CORPUS) -> dict[str, int]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    documents = read_corpus(corpus_path)
    counts = {"new": 0, "updated": 0, "unchanged": 0, "failed": 0}
    for entry in discover(config):
        try:
            kind = entry["kind"]
            if kind not in {"evidencia", "educativo"}:
                raise ValueError(f"Tipo inválido: {kind}")
            source = entry["source"]
            origin_group = entry["origin_group"]
            location = entry.get("url") or entry.get("path")
            if not location:
                raise ValueError("Documento sem url ou path")
            if entry.get("url"):
                if entry.get("usage_approved") is not True:
                    raise ValueError("Confirme uso permitido com usage_approved=true antes da coleta remota")
                text, title, raw_path, detected_date = download(location)
            else:
                path = (config_path.parent / location).resolve()
                text, title = extract_local(path)
                location = path.as_uri()
                raw_path = str(path)
                detected_date = ""
            reference_url = entry.get("reference_url")
            if reference_url and urlparse(reference_url).scheme not in {"http", "https"}:
                raise ValueError(f"reference_url deve ser HTTP(S): {reference_url}")
            citation_url = reference_url or location
            if len(text) < 100:
                raise ValueError("Texto extraído muito curto")
            identifier = stable_id(location)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            previous = documents.get(identifier)
            metadata = (
                kind,
                source,
                origin_group,
                entry.get("title") or title,
                entry.get("published_at") or detected_date,
                entry.get("license", "verificar"),
                citation_url,
            )
            previous_metadata = (
                previous.get("kind"),
                previous.get("source"),
                previous.get("origin_group"),
                previous.get("title"),
                previous.get("published_at", ""),
                previous.get("license", "verificar"),
                previous.get("url"),
            ) if previous else None
            if previous and previous["content_hash"] == digest and previous_metadata == metadata:
                counts["unchanged"] += 1
                continue
            status = "updated" if identifier in documents else "new"
            documents[identifier] = {
                "id": identifier,
                "kind": kind,
                "source": source,
                "origin_group": origin_group,
                "url": citation_url,
                "title": entry.get("title") or title,
                "published_at": entry.get("published_at") or detected_date,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "license": entry.get("license", "verificar"),
                "content_hash": digest,
                "raw_path": raw_path,
                "text": text,
            }
            counts[status] += 1
        except (KeyError, OSError, ValueError, requests.RequestException) as error:
            print(f"Falha ao coletar {entry.get('url') or entry.get('path')}: {error}")
            counts["failed"] += 1
    write_corpus(documents, corpus_path)
    return counts


def chunks(text: str, max_words: int = 180, overlap: int = 30) -> list[str]:
    if max_words <= overlap or overlap < 0:
        raise ValueError("max_words deve ser maior que overlap")
    words = text.split()
    step = max_words - overlap
    return [
        " ".join(words[start:start + max_words])
        for start in range(0, len(words), step)
        if start == 0 or len(words) - start > overlap
    ]
