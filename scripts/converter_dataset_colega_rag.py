"""Converte as visões preparadas do FakeTrue.Br para o corpus e a avaliação do RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

from rag.corpus import CORPUS, ROOT, read_corpus, stable_id, write_corpus

DEFAULT_EVIDENCE = ROOT / "data" / "processed" / "base_checagens.parquet"
DEFAULT_EVALUATION = ROOT / "data" / "processed" / "avaliacao_busca.parquet"
DEFAULT_INTERIM = ROOT / "data" / "interim" / "textos.parquet"
DEFAULT_CASES = ROOT / "data" / "avaliacao" / "casos_dataset_colega.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "manifest_rag.json"
SCHEMA_VERSION = "dataset-colega-rag-v1"
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_NAMES = {"fbclid", "gclid"}


def canonical_url(value: object) -> str:
    """Normaliza uma URL sem remover parâmetros que possam identificar a página."""
    raw = str(value).strip()
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"URL HTTP(S) inválida: {raw!r}")
    host = parsed.hostname.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    port = parsed.port
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(sorted(
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_NAMES
        and not key.lower().startswith(TRACKING_QUERY_PREFIXES)
    ))
    return urlunsplit((parsed.scheme.lower(), netloc, path, query, ""))


def publisher(domain: object) -> tuple[str, str]:
    host = str(domain or "").strip().lower().removeprefix("www.")
    ordered = (
        ("piaui.folha.uol.com.br", "Revista Piauí", "grupo_folha"),
        ("folha.uol.com.br", "Folha de S.Paulo", "grupo_folha"),
        ("g1.globo.com", "G1", "grupo_globo"),
        ("globo.com", "Grupo Globo", "grupo_globo"),
        ("uol.com.br", "UOL", "grupo_uol"),
        ("estadao.com.br", "Estadão", "grupo_estado"),
        ("r7.com", "R7", "grupo_record"),
        ("correiobraziliense.com.br", "Correio Braziliense", "grupo_diarios_associados"),
    )
    for suffix, source, group in ordered:
        if host == suffix or host.endswith(f".{suffix}"):
            return source, group
    safe = host or "fonte_desconhecida"
    return safe, safe.replace(".", "_")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "desconhecida"


def require_columns(frame: pd.DataFrame, expected: set[str], source: Path) -> None:
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: colunas ausentes: {', '.join(missing)}")


def title_map(interim_path: Path) -> dict[str, str]:
    if not interim_path.is_file():
        return {}
    frame = pd.read_parquet(interim_path, columns=["link", "titulo"])
    titles: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        if pd.isna(row.link) or pd.isna(row.titulo):
            continue
        title = str(row.titulo).strip()
        if not title:
            continue
        try:
            url = canonical_url(row.link)
        except ValueError:
            continue
        current = titles.get(url, "")
        if (len(title), title) > (len(current), current):
            titles[url] = title
    return titles


def year_text(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip()


def fallback_title(nature: str, domain: str, year: str) -> str:
    kind = "Checagem" if nature == "checagem" else "Notícia"
    suffix = f" — {year}" if year else ""
    return f"{kind} publicada por {domain}{suffix}"


def evidence_documents(
    frame: pd.DataFrame,
    titles: dict[str, str],
    evidence_path: Path,
    license_name: str,
) -> tuple[dict[str, dict], dict[str, str], list[dict]]:
    require_columns(
        frame,
        {"id_checagem", "natureza", "texto_limpo", "link", "dominio", "ano"},
        evidence_path,
    )
    candidates = []
    for number, row in enumerate(frame.itertuples(index=False), 1):
        nature = str(row.natureza).strip().lower()
        if nature not in {"checagem", "noticia"}:
            raise ValueError(f"{evidence_path}: linha {number} possui natureza inválida: {nature!r}")
        text = str(row.texto_limpo).strip() if not pd.isna(row.texto_limpo) else ""
        if not text:
            raise ValueError(f"{evidence_path}: linha {number} possui texto vazio")
        url = canonical_url(row.link)
        domain = str(row.dominio).strip().lower() if not pd.isna(row.dominio) else urlsplit(url).hostname or ""
        source, origin_group = publisher(domain)
        published_at = year_text(row.ano)
        digest = sha256_text(text)
        candidates.append({
            "id": stable_id(url),
            "kind": "evidencia",
            "source": source,
            "origin_group": origin_group,
            "url": url,
            "title": titles.get(url) or fallback_title(nature, domain, published_at),
            "published_at": published_at,
            "license": license_name,
            "content_hash": digest,
            "raw_path": f"{display_path(evidence_path)}#{row.id_checagem}",
            "text": text,
            "natureza": nature,
            "dataset_name": "FakeTrue.Br",
        })

    # Mantém uma URL canônica por conteúdo idêntico e conserva aliases para a avaliação.
    candidates.sort(key=lambda item: (
        item["content_hash"],
        0 if item["natureza"] == "checagem" else 1,
        -len(item["text"]),
        item["url"],
    ))
    by_hash: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    discarded: list[dict] = []
    for document in candidates:
        kept = by_hash.get(document["content_hash"])
        if kept is None:
            by_hash[document["content_hash"]] = document
            continue
        aliases[document["url"]] = kept["url"]
        discarded.append({
            "url_descartada": document["url"],
            "url_mantida": kept["url"],
            "motivo": "texto_exatamente_igual",
        })
    documents = {item["id"]: item for item in by_hash.values()}
    if len(documents) != len({item["url"] for item in documents.values()}):
        raise ValueError("A conversão produziu URLs duplicadas")
    return documents, aliases, discarded


def evaluation_cases(
    frame: pd.DataFrame,
    evaluation_path: Path,
    aliases: dict[str, str],
    evidence_urls: set[str],
) -> list[dict]:
    require_columns(
        frame,
        {"id_original", "titulo", "texto_limpo", "link_boato", "link_checagem_esperada", "checagem_na_base"},
        evaluation_path,
    )
    cases = []
    identifiers = set()
    for number, row in enumerate(frame.itertuples(index=False), 1):
        identifier = str(row.id_original).strip() or str(number)
        if identifier in identifiers:
            raise ValueError(f"{evaluation_path}: id_original repetido: {identifier}")
        identifiers.add(identifier)
        title = "" if pd.isna(row.titulo) else str(row.titulo).strip()
        text = "" if pd.isna(row.texto_limpo) else str(row.texto_limpo).strip()
        question = title if len(title) >= 10 else text
        if not question:
            raise ValueError(f"{evaluation_path}: linha {number} sem alegação")
        expected = canonical_url(row.link_checagem_esperada)
        in_base = bool(row.checagem_na_base)
        expected = aliases.get(expected, expected)
        if in_base and expected not in evidence_urls:
            raise ValueError(f"Caso {identifier}: URL esperada marcada na base não foi convertida: {expected}")
        if not in_base and expected in evidence_urls:
            raise ValueError(f"Caso {identifier}: URL reservada vazou para o corpus: {expected}")
        cases.append({
            "id": identifier,
            "pergunta": question,
            "avaliacao_revisada": None,
            "urls_esperadas": [expected] if in_base else [],
            "espera_sem_evidencia": not in_base,
            "url_checagem_reservada": expected if not in_base else "",
        })
    return cases


def atomic_write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def convert(
    evidence_path: Path = DEFAULT_EVIDENCE,
    evaluation_path: Path = DEFAULT_EVALUATION,
    interim_path: Path = DEFAULT_INTERIM,
    corpus_path: Path = CORPUS,
    cases_path: Path = DEFAULT_CASES,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    license_name: str = "verificar",
    preserve_educational: bool = True,
) -> dict:
    for path in (evidence_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset não encontrado: {path}")
    evidence_frame = pd.read_parquet(evidence_path)
    evaluation_frame = pd.read_parquet(evaluation_path)
    documents, aliases, discarded = evidence_documents(
        evidence_frame, title_map(interim_path), evidence_path, license_name.strip() or "verificar"
    )
    educational = {}
    if preserve_educational:
        educational = {
            key: value for key, value in read_corpus(corpus_path).items()
            if value.get("kind") == "educativo"
        }
    collisions = set(documents) & set(educational)
    if collisions:
        raise ValueError(f"IDs repetidos entre evidências e materiais educativos: {sorted(collisions)[:3]}")
    corpus = {**educational, **documents}

    evidence_urls = {item["url"] for item in documents.values()}
    cases = evaluation_cases(evaluation_frame, evaluation_path, aliases, evidence_urls)
    reserved_urls = {
        item["url_checagem_reservada"] for item in cases if item["espera_sem_evidencia"]
    }
    leaked = sorted(reserved_urls & evidence_urls)
    if leaked:
        raise ValueError(f"URLs reservadas presentes no corpus: {leaked[:3]}")

    # Só substitui os destinos depois que corpus e casos passam nas validações cruzadas.
    write_corpus(corpus, corpus_path)
    atomic_write_jsonl(cases, cases_path)

    license_status = license_name.strip() or "verificar"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "inputs": {
            display_path(evidence_path): {
                "sha256": sha256_file(evidence_path), "rows": len(evidence_frame),
            },
            display_path(evaluation_path): {
                "sha256": sha256_file(evaluation_path), "rows": len(evaluation_frame),
            },
            **({display_path(interim_path): {
                "sha256": sha256_file(interim_path),
                "rows": len(pd.read_parquet(interim_path, columns=["link"])),
            }} if interim_path.is_file() else {}),
        },
        "conversion": {
            "license": license_status,
            "publication_allowed": license_status.casefold() not in {"", "verificar", "não informada"},
            "preserve_educational": preserve_educational,
            "deduplication": "content_hash; checagem, maior texto e URL como desempate",
        },
        "corpus": {
            "path": display_path(corpus_path),
            "sha256": sha256_file(corpus_path),
            "documents": len(corpus),
            "evidence_documents": len(documents),
            "educational_documents": len(educational),
            "discarded_exact_duplicates": len(discarded),
            "discarded": discarded,
            "by_nature": dict(Counter(item["natureza"] for item in documents.values())),
            "by_origin_group": dict(Counter(item["origin_group"] for item in documents.values())),
            "by_source": dict(Counter(item["source"] for item in documents.values())),
        },
        "evaluation": {
            "path": display_path(cases_path),
            "sha256": sha256_file(cases_path),
            "cases": len(cases),
            "with_expected_url": sum(bool(item["urls_esperadas"]) for item in cases),
            "reserved_without_evidence": sum(item["espera_sem_evidencia"] for item in cases),
        },
        "validation": {
            "reserved_url_leaks": len(leaked),
            "empty_evidence_texts": sum(not item["text"].strip() for item in documents.values()),
            "empty_evidence_urls": sum(not item["url"].strip() for item in documents.values()),
        },
    }
    atomic_write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidencias", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--avaliacao", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--interim", type=Path, default=DEFAULT_INTERIM)
    parser.add_argument("--saida-corpus", type=Path, default=CORPUS)
    parser.add_argument("--saida-casos", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--manifesto", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--license", default="verificar", help="Licença ou condição de uso revisada")
    parser.add_argument(
        "--nao-preservar-educativos", action="store_true",
        help="Não copiar documentos educativos existentes para o corpus convertido",
    )
    args = parser.parse_args()
    manifest = convert(
        args.evidencias.resolve(), args.avaliacao.resolve(), args.interim.resolve(),
        args.saida_corpus.resolve(), args.saida_casos.resolve(), args.manifesto.resolve(),
        license_name=args.license,
        preserve_educational=not args.nao_preservar_educativos,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
