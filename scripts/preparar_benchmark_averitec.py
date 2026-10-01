"""Baixa AVeriTeC PT-BR e cria corpus de evidências e casos de avaliação isolados."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests

from rag.corpus import ROOT, clean_text, stable_id, write_corpus

DATASET_URL = (
    "https://raw.githubusercontent.com/lucasfrag/automated-fact-checking-in-pt-br/"
    "main/averitec/pt-br/dataset/dev.json"
)
DATASET_SHA256 = "aca47877ee050909791b03436a3b68f4879a3b98de2ed756822ff7ceae7b5a29"
LABELS = {"Refuted": "FALSA", "Supported": "VERDADEIRA", "Not Enough Evidence": "INCONCLUSIVA"}
MEDIA = {"Web text", "PDF", "Web table", "Metadata"}
BASE = ROOT / "data" / "benchmark_averitec"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_dataset(path: Path) -> list[dict]:
    if path.exists() and sha256(path.read_bytes()) == DATASET_SHA256:
        return json.loads(path.read_text(encoding="utf-8"))
    response = requests.get(DATASET_URL, timeout=90)
    response.raise_for_status()
    if len(response.content) > 20_000_000 or sha256(response.content) != DATASET_SHA256:
        raise ValueError("O download não corresponde ao snapshot esperado do AVeriTeC PT-BR")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.json()


def source_group(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname == "web.archive.org":
        match = re.search(r"/(?:https?://)([^/]+)", parsed.path)
        if match:
            return match.group(1).lower().removeprefix("www.")
    return (parsed.hostname or "").lower().removeprefix("www.")


def evidence_for(row: dict, case_id: int) -> list[dict]:
    result = []
    for question_number, question in enumerate(row.get("questions", [])):
        prompt = clean_text(question.get("question") or "")
        for answer_number, answer in enumerate(question.get("answers", [])):
            url = answer.get("source_url") or ""
            answer_text = clean_text(answer.get("answer") or "")
            group = source_group(url)
            if (
                answer.get("answer_type") == "Unanswerable"
                or answer.get("source_medium") not in MEDIA
                or urlparse(url).scheme not in {"http", "https"}
                or not group or len(answer_text) < 12 or not prompt
            ):
                continue
            text = f"Pergunta de checagem: {prompt}\nResposta documentada: {answer_text}"
            result.append({"id": stable_id(f"averitec-ptbr-dev:{case_id}:{question_number}:{answer_number}"),
                           "kind": "evidencia", "source": group, "origin_group": group,
                           "url": url, "title": f"Pergunta e resposta de evidência {case_id}-{question_number + 1}",
                           "published_at": "",
                           "license": "AVeriTeC: CC BY-NC 4.0; URL original sob termos do publicador",
                           "content_hash": sha256(text.encode("utf-8")),
                           "raw_path": DATASET_URL, "text": text})
    return result


def prepare(per_class: int = 10, seed: int = 42, base: Path = BASE) -> dict:
    if per_class < 1:
        raise ValueError("--per-class deve ser positivo")
    base = base.resolve()
    raw_path = base / "dev.json"
    rows = fetch_dataset(raw_path)
    candidates = {label: [] for label in LABELS.values()}
    skipped = Counter()
    seen_claims = set()
    for number, row in enumerate(rows):
        label = LABELS.get(row.get("label"))
        if label is None:
            skipped["rotulo_ambiguo"] += 1
            continue
        claim = clean_text(row.get("claim") or "")
        claim_key = " ".join(claim.casefold().split())
        if len(claim) < 25 or claim_key in seen_claims:
            skipped["alegacao_curta_ou_repetida"] += 1
            continue
        evidence = evidence_for(row, number)
        groups = {item["origin_group"] for item in evidence}
        if len(groups) < (1 if label == "INCONCLUSIVA" else 2):
            skipped["fontes_insuficientes"] += 1
            continue
        seen_claims.add(claim_key)
        candidates[label].append((number, row, evidence))
    selected = []
    for label, group in candidates.items():
        ordered = sorted(group, key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest())
        if len(ordered) < per_class:
            raise ValueError(f"Somente {len(ordered)} casos elegíveis para {label}; reduza --per-class")
        selected.extend(ordered[:per_class])
    selected.sort(key=lambda item: item[0])
    documents = {}
    cases = []
    for number, row, evidence in selected:
        for item in evidence:
            documents[item["id"]] = item
        cases.append({"id": f"averitec-dev-{number}", "pergunta": row["claim"],
                      "urls_esperadas": sorted({item["url"] for item in evidence}),
                      "veredito_revisado": LABELS[row["label"]],
                      "justificativa_revisada": row.get("justification", ""),
                      "origem_rotulo": "AVeriTeC dev PT-BR; tradução automática"})
    base.mkdir(parents=True, exist_ok=True)
    write_corpus(documents, base / "documents.jsonl")
    with (base / "casos.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    manifest = {"dataset_url": DATASET_URL, "dataset_sha256": DATASET_SHA256,
                "licenca": "CC BY-NC 4.0", "seed": seed, "por_classe": per_class,
                "total_casos": len(cases), "total_evidencias": len(documents),
                "rotulos": dict(Counter(case["veredito_revisado"] for case in cases)),
                "descartes": dict(skipped),
                "protocolo": "Somente perguntas/respostas com URL entram no índice; rótulos e justificativas ficam fora."}
    (base / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=10, help="Casos por veredito; padrão: 10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=BASE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.per_class, args.seed, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
