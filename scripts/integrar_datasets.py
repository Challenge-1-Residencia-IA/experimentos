"""Script de integração e tratamento de datasets públicos para o RAG.

Suporta:
- FakeRecogna (Hugging Face - 11.9k notícias checadas e apuradas em PT-BR)
- X-FACT (Hugging Face - split PT de verificação de fatos com evidências Google)
- AVeriTeC PT-BR (GitHub - alegações e evidências com URLs)
- Fake.br Local (/Users/aluno2/Fake.br-Corpus - corpus da USP com textos completos)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.corpus import CORPUS, ROOT, clean_text, read_corpus, stable_id, write_corpus
from rag.splitter import RecursiveTextSplitter, split_large_text

# URLs dos datasets públicos gratuitos
URL_FAKERECOGNA_CSV = "https://huggingface.co/datasets/recogna-nlp/FakeRecogna/resolve/main/FakeRecogna.csv"
URL_XFACT_PT_DEV = "https://huggingface.co/datasets/utahnlp/x-fact/raw/main/pt/dev-00000-of-00001.csv"
URL_XFACT_PT_TRAIN = "https://huggingface.co/datasets/utahnlp/x-fact/raw/main/pt/train-00000-of-00001.csv"
URL_AVERITEC_PTBR = "https://raw.githubusercontent.com/lucasfrag/automated-fact-checking-in-pt-br/main/averitec/pt-br/dataset/dev.json"
LOCAL_FAKEBR_PATH = Path("/Users/aluno2/Fake.br-Corpus/data/processed/train.jsonl")


def extract_origin_group(url: str, default: str = "fonte_externa") -> str:
    """Extrai o grupo de origem (domínio limpo) para manter diversidade de fontes."""
    if not url:
        return default
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not host:
        return default
    # Agrupa domínios conhecidos
    for domain in ("g1.globo.com", "uol.com.br", "boatos.org", "e-farsas.com", "afp.com", "aosfatos.org", "lupa.news"):
        if domain in host:
            return domain.split(".")[0]
    labels = host.split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def process_fakerecogna(
    limit: int = 50,
    splitter: RecursiveTextSplitter | None = None,
    inject_context: bool = False,
) -> list[dict]:
    """Baixa e processa amostras do FakeRecogna (checagens e notícias reais)."""
    print(f"\n[1/4] Baixando e processando FakeRecogna (limite: {limit})...")
    response = requests.get(URL_FAKERECOGNA_CSV, stream=True, timeout=60)
    response.raise_for_status()

    documents = []
    lines_buffer = []
    splitter = splitter or RecursiveTextSplitter(chunk_size=180, chunk_overlap=35)

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        lines_buffer.append(line)
        if len(lines_buffer) > (limit * 5) and len(documents) >= limit:
            break

        # Tenta processar em pequenos lotes para economizar memória
        if len(lines_buffer) >= 50:
            reader = csv.DictReader(StringIO("\n".join(lines_buffer)))
            lines_buffer = [lines_buffer[0]]  # mantém cabeçalho
            for row in reader:
                title = clean_text(row.get("Titulo") or "")
                body = clean_text(row.get("Noticia") or "")
                url = row.get("URL") or ""
                group = extract_origin_group(url, default="fakerecogna")
                published_at = row.get("Data") or ""
                classe = row.get("Classe", "")

                if len(body) < 120 or not title:
                    continue

                full_text = f"# {title}\n\n{body}"
                chunks = splitter.split_text(
                    full_text,
                    title=title,
                    source=group,
                    inject_context=inject_context,
                )

                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = stable_id(f"fakerecogna:{url}:{chunk_idx}")
                    digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    documents.append({
                        "id": doc_id,
                        "kind": "evidencia",
                        "source": f"FakeRecogna ({group})",
                        "origin_group": group,
                        "url": url or "https://huggingface.co/datasets/recogna-nlp/FakeRecogna",
                        "title": f"{title} (Trecho {chunk_idx + 1})" if len(chunks) > 1 else title,
                        "published_at": published_at,
                        "license": "FakeRecogna: MIT",
                        "content_hash": digest,
                        "raw_path": URL_FAKERECOGNA_CSV,
                        "text": chunk,
                    })

                if len(documents) >= limit:
                    break

    print(f"-> FakeRecogna: {len(documents)} trechos de evidência gerados.")
    return documents[:limit]


def process_xfact(
    limit: int = 50,
    splitter: RecursiveTextSplitter | None = None,
    inject_context: bool = False,
) -> list[dict]:
    """Baixa e processa amostras do X-FACT (split português com evidências do Google)."""
    print(f"\n[2/4] Baixando e processando X-FACT PT (limite: {limit})...")
    response = requests.get(URL_XFACT_PT_DEV, timeout=45)
    response.raise_for_status()

    reader = csv.DictReader(response.text.splitlines())
    documents = []
    splitter = splitter or RecursiveTextSplitter(chunk_size=180, chunk_overlap=35)

    for i, row in enumerate(reader):
        claim = clean_text(row.get("claim") or "")
        site = row.get("site") or "xfact"
        date = row.get("claimDate") or ""
        label = row.get("label") or ""

        # Coleta as 5 evidências disponíveis no X-FACT
        evidences = []
        for e_idx in range(1, 6):
            ev_text = clean_text(row.get(f"evidence_{e_idx}") or "")
            ev_link = row.get(f"link_{e_idx}") or ""
            if len(ev_text) >= 40 and ev_link.startswith("http"):
                evidences.append((ev_text, ev_link))

        if not claim or not evidences:
            continue

        for ev_idx, (ev_text, ev_link) in enumerate(evidences):
            group = extract_origin_group(ev_link, default=site)
            title = f"Checagem X-FACT: {claim[:80]}..."
            doc_text = f"Alegação avaliada: {claim}\nVeredito documentado: {label}\nEvidência apurada: {ev_text}"

            chunks = splitter.split_text(
                doc_text,
                title=title,
                source=group,
                inject_context=inject_context,
            )

            for chunk_idx, chunk in enumerate(chunks):
                doc_id = stable_id(f"xfact:{i}:{ev_idx}:{chunk_idx}")
                digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                documents.append({
                    "id": doc_id,
                    "kind": "evidencia",
                    "source": f"X-FACT ({group})",
                    "origin_group": group,
                    "url": ev_link,
                    "title": title,
                    "published_at": date[:10] if date else "",
                    "license": "X-FACT: MIT",
                    "content_hash": digest,
                    "raw_path": URL_XFACT_PT_DEV,
                    "text": chunk,
                })

        if len(documents) >= limit:
            break

    print(f"-> X-FACT: {len(documents)} trechos de evidência gerados.")
    return documents[:limit]


def process_averitec(
    limit: int = 50,
    splitter: RecursiveTextSplitter | None = None,
    inject_context: bool = False,
) -> list[dict]:
    """Baixa e processa amostras do benchmark AVeriTeC PT-BR."""
    print(f"\n[3/4] Baixando e processando AVeriTeC PT-BR (limite: {limit})...")
    response = requests.get(URL_AVERITEC_PTBR, timeout=60)
    response.raise_for_status()
    rows = response.json()

    documents = []
    splitter = splitter or RecursiveTextSplitter(chunk_size=180, chunk_overlap=35)

    for case_id, row in enumerate(rows):
        claim = clean_text(row.get("claim") or "")
        label = row.get("label") or ""
        questions = row.get("questions") or []

        for q_idx, q in enumerate(questions):
            q_text = clean_text(q.get("question") or "")
            for a_idx, a in enumerate(q.get("answers") or []):
                ans_text = clean_text(a.get("answer") or "")
                source_url = a.get("source_url") or ""
                group = extract_origin_group(source_url, default="averitec")

                if len(ans_text) < 30 or not source_url.startswith("http"):
                    continue

                full_text = (
                    f"Fato investigado: {claim}\n"
                    f"Pergunta de checagem: {q_text}\n"
                    f"Evidência documental: {ans_text}"
                )
                title = f"Evidência AVeriTeC: {claim[:70]}..."
                chunks = splitter.split_text(
                    full_text,
                    title=title,
                    source=group,
                    inject_context=inject_context,
                )

                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = stable_id(f"averitec:{case_id}:{q_idx}:{a_idx}:{chunk_idx}")
                    digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    documents.append({
                        "id": doc_id,
                        "kind": "evidencia",
                        "source": f"AVeriTeC ({group})",
                        "origin_group": group,
                        "url": source_url,
                        "title": title,
                        "published_at": "",
                        "license": "AVeriTeC: CC BY-NC 4.0",
                        "content_hash": digest,
                        "raw_path": URL_AVERITEC_PTBR,
                        "text": chunk,
                    })

                if len(documents) >= limit:
                    break
            if len(documents) >= limit:
                break
        if len(documents) >= limit:
            break

    print(f"-> AVeriTeC PT-BR: {len(documents)} trechos de evidência gerados.")
    return documents[:limit]


def process_fakebr_local(
    path: Path = LOCAL_FAKEBR_PATH,
    limit: int = 50,
    splitter: RecursiveTextSplitter | None = None,
    inject_context: bool = False,
) -> list[dict]:
    """Processa notícias completas do Fake.br existente localmente na máquina."""
    print(f"\n[4/4] Processando Fake.br local de {path} (limite: {limit})...")
    if not path.exists():
        print(f"Aviso: Arquivo {path} não encontrado. Pulando Fake.br local.")
        return []

    documents = []
    splitter = splitter or RecursiveTextSplitter(chunk_size=180, chunk_overlap=35)

    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            msgs = item.get("messages", [])
            # Busca a mensagem do usuário (que contém o texto da notícia)
            user_msg = next((m.get("content") for m in msgs if m.get("role") == "user"), "")
            # Remove o prefixo do prompt se existir
            clean_news = re.sub(r"^Mensagem a ser analisada:\s*", "", user_msg, flags=re.IGNORECASE).strip()
            if len(clean_news) < 150:
                continue

            first_line = clean_news.splitlines()[0] if clean_news.splitlines() else clean_news[:60]
            title = clean_text(first_line[:90])
            group = "fakebr_corpus"

            chunks = splitter.split_text(
                clean_news,
                title=title,
                source="Fake.br Corpus (USP)",
                inject_context=inject_context,
            )

            for chunk_idx, chunk in enumerate(chunks):
                doc_id = stable_id(f"fakebr:{i}:{chunk_idx}")
                digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                documents.append({
                    "id": doc_id,
                    "kind": "evidencia",
                    "source": "Fake.br Corpus (USP)",
                    "origin_group": group,
                    "url": "https://github.com/roneysco/Fake.br-Corpus",
                    "title": f"{title} (Parte {chunk_idx + 1}/{len(chunks)})",
                    "published_at": "",
                    "license": "Fake.br Corpus: Academic Use",
                    "content_hash": digest,
                    "raw_path": str(path),
                    "text": chunk,
                })

            if len(documents) >= limit:
                break

    print(f"-> Fake.br Local: {len(documents)} trechos de evidência gerados.")
    return documents[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Integração de múltiplos datasets e fragmentação de alta qualidade para o RAG.")
    parser.add_argument(
        "--dataset",
        choices=["fakerecogna", "xfact", "averitec", "fakebr", "todos"],
        default="todos",
        help="Dataset a ser integrado",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Limite de documentos/trechos por dataset selecionado",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=180,
        help="Tamanho máximo de palavras em cada chunk",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=35,
        help="Sobreposição de palavras entre chunks contíguos",
    )
    parser.add_argument(
        "--inject-context",
        action="store_true",
        help="Adicionar cabeçalho com título e fonte em cada chunk",
    )
    parser.add_argument(
        "--output-corpus",
        type=Path,
        default=CORPUS,
        help="Arquivo de destino do corpus (padrão: data/documents.jsonl)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescrever o corpus existente em vez de mesclar",
    )
    args = parser.parse_args()

    splitter = RecursiveTextSplitter(
        chunk_size=args.max_words,
        chunk_overlap=args.overlap,
        unit="words",
    )

    novos_documentos = []
    if args.dataset in {"fakerecogna", "todos"}:
        novos_documentos.extend(process_fakerecogna(limit=args.limit, splitter=splitter, inject_context=args.inject_context))
    if args.dataset in {"xfact", "todos"}:
        novos_documentos.extend(process_xfact(limit=args.limit, splitter=splitter, inject_context=args.inject_context))
    if args.dataset in {"averitec", "todos"}:
        novos_documentos.extend(process_averitec(limit=args.limit, splitter=splitter, inject_context=args.inject_context))
    if args.dataset in {"fakebr", "todos"}:
        novos_documentos.extend(process_fakebr_local(limit=args.limit, splitter=splitter, inject_context=args.inject_context))

    # Carrega ou cria corpus
    corpus = {} if args.overwrite else read_corpus(args.output_corpus)
    adicionados = 0
    atualizados = 0

    for doc in novos_documentos:
        doc_id = doc["id"]
        if doc_id in corpus:
            if corpus[doc_id]["content_hash"] != doc["content_hash"]:
                corpus[doc_id] = doc
                atualizados += 1
        else:
            corpus[doc_id] = doc
            adicionados += 1

    write_corpus(corpus, args.output_corpus)

    # Diagnóstico e estatísticas
    print("\n" + "=" * 60)
    print("RESUMO DA INTEGRAÇÃO DO RAG")
    print("=" * 60)
    print(f"Destino do corpus: {args.output_corpus}")
    print(f"Novos documentos adicionados: {adicionados}")
    print(f"Documentos atualizados: {atualizados}")
    print(f"Total geral no corpus: {len(corpus)}")

    grupos = Counter(d.get("origin_group") for d in corpus.values())
    tipos = Counter(d.get("kind") for d in corpus.values())
    print("\nDistribuição por tipo de documento:")
    for k, v in tipos.items():
        print(f" - {k}: {v}")

    print("\nTop 8 origens de informação no corpus:")
    for g, count in grupos.most_common(8):
        print(f" - {g}: {count} trechos")

    print("\n[OK] Concluído com sucesso! Agora você pode rodar a indexação vetorial:")
    print("  python scripts/indexar_rag.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
