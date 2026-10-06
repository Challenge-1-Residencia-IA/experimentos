"""Publica uma versão reproduzível dos datasets usados pelo RAG no DagsHub."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import mlflow

from rag.corpus import CORPUS, ROOT, read_corpus
from rag.dagshub_tracking import configure_tracking

DATASET_VERSION = "rag-datasets-v2"
BENCHMARK = ROOT / "data" / "benchmark_averitec"
COLLEAGUE_CASES = ROOT / "data" / "avaliacao" / "casos_dataset_colega.jsonl"
COLLEAGUE_MANIFEST = ROOT / "data" / "manifest_rag.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def main() -> None:
    files = {
        "corpus/documents.jsonl": CORPUS,
    }
    optional = {
        "avaliacao/casos_dataset_colega.jsonl": COLLEAGUE_CASES,
        "manifest_rag.json": COLLEAGUE_MANIFEST,
        "benchmark_averitec/casos.jsonl": BENCHMARK / "casos.jsonl",
        "benchmark_averitec/dev.json": BENCHMARK / "dev.json",
        "benchmark_averitec/documents.jsonl": BENCHMARK / "documents.jsonl",
        "benchmark_averitec/manifest.json": BENCHMARK / "manifest.json",
    }
    files.update({name: path for name, path in optional.items() if path.is_file()})
    missing = [str(path) for path in (CORPUS, COLLEAGUE_CASES, COLLEAGUE_MANIFEST) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Arquivos de dataset ausentes: " + ", ".join(missing))

    corpus = read_corpus(CORPUS)
    licenses = Counter(str(item.get("license", "não informada")) for item in corpus.values())
    unreviewed = {name for name in licenses if name.casefold() in {"", "verificar", "não informada"}}
    if unreviewed:
        raise ValueError(
            "Publicação bloqueada: revise a licença/condição de uso do corpus: "
            + ", ".join(sorted(unreviewed))
        )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "visibility_required": "private",
        "files": {
            name: {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                **({"rows": jsonl_rows(path)} if path.suffix == ".jsonl" else {}),
            }
            for name, path in files.items()
        },
        "corpus_documents": len(corpus),
        "corpus_kinds": dict(Counter(item["kind"] for item in corpus.values())),
        "corpus_licenses": dict(licenses),
        "notes": (
            "Dados para pesquisa e avaliação do RAG. Respeite as condições de uso "
            "registradas no corpus e mantenha o artefato privado quando exigido."
        ),
    }

    uri = configure_tracking(remote=True)
    with mlflow.start_run(run_name=f"dataset-{DATASET_VERSION}") as run:
        mlflow.log_params({
            "dataset_version": DATASET_VERSION,
            "corpus_documents": len(corpus),
            "corpus_sha256": manifest["files"]["corpus/documents.jsonl"]["sha256"],
            "evaluation_cases": manifest["files"]["avaliacao/casos_dataset_colega.jsonl"]["rows"],
        })
        for artifact_name, path in files.items():
            mlflow.log_artifact(str(path), artifact_path=str(Path("datasets") / Path(artifact_name).parent))
        mlflow.log_dict(manifest, "datasets/manifest.json")
        mlflow.set_tags({"tipo": "dataset_rag", "dataset_version": DATASET_VERSION,
                         "uso": "pesquisa_privada"})
        print(json.dumps({"tracking_uri": uri, "run_id": run.info.run_id,
                          "dataset_version": DATASET_VERSION,
                          "corpus_documents": len(corpus),
                          "evaluation_cases": manifest["files"]["avaliacao/casos_dataset_colega.jsonl"]["rows"]},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
