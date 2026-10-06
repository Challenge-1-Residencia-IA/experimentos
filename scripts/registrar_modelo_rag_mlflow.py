"""Empacota o RAG atual e cria uma versão no Model Registry do DagsHub."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from rag.corpus import CORPUS, ROOT, read_corpus
from rag.dagshub_tracking import REGISTERED_MODEL_NAME, configure_tracking, contract_fingerprint, git_revision
from rag.mlflow_model import RAGModel, ollama_digest
from rag.store import MODEL, STORE
from rag.tutor import ANSWER_SCHEMA, RESPONSE_VERSION, SYSTEM_PROMPT

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_fingerprint(path: Path) -> str:
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        raise ValueError("O índice Chroma está vazio")
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def embedding_revision() -> str:
    snapshots = list((Path.home() / ".cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots").glob("*"))
    revisions = [item.name for item in snapshots if item.is_dir() and len(item.name) == 40]
    if len(revisions) != 1:
        raise ValueError("Não foi possível identificar uma única revisão local do modelo de embeddings")
    return revisions[0]


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not STORE.is_dir() or not CORPUS.is_file():
        raise FileNotFoundError("Índice Chroma e corpus são necessários para registrar o RAG")
    documents = read_corpus()
    licenses = dict(Counter(str(item.get("license", "não informada")) for item in documents.values()))
    qwen = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    digest = ollama_digest(qwen)
    revision = embedding_revision()
    index_hash = index_fingerprint(STORE)
    corpus_hash = sha256(CORPUS)
    source_hashes = {file: sha256(ROOT / file) for file in (
        "rag/corpus.py", "rag/store.py", "rag/tutor.py", "rag/mlflow_model.py",
        "rag/dagshub_tracking.py", "rag/web.py", "web_sources.json",
        "prompts/tutora_rag.txt")}
    manifest = {
        "response_version": RESPONSE_VERSION,
        "ollama_model": qwen,
        "ollama_sha256": digest,
        "embedding_model": MODEL,
        "embedding_revision": revision,
        "index_sha256": index_hash,
        "corpus_sha256": corpus_hash,
        "web_sources_sha256": sha256(ROOT / "web_sources.json"),
        "document_count": len(documents),
        "licenses": licenses,
        "source_sha256": source_hashes,
        "git_revision": git_revision(),
        "contract": contract_fingerprint(SYSTEM_PROMPT, ANSWER_SCHEMA),
        "runtime": "Ollama local; pesos Qwen não incluídos no artefato",
    }
    configure_tracking()
    requirements = [f"{name}=={version(name)}" for name in (
        "mlflow", "chromadb", "sentence-transformers", "requests",
        "python-dotenv", "feedparser", "beautifulsoup4", "pypdf", "pandas",
        "ddgs", "trafilatura")]
    with mlflow.start_run(run_name=f"registro-rag-{RESPONSE_VERSION}") as run:
        mlflow.log_params({
            "response_version": RESPONSE_VERSION,
            "ollama_model": qwen,
            "ollama_sha256": digest,
            "embedding_model": MODEL,
            "embedding_revision": revision,
            "index_sha256": index_hash,
            "corpus_sha256": corpus_hash,
            "git_revision": manifest["git_revision"],
            **manifest["contract"],
        })
        mlflow.log_dict(manifest, "versao/manifest.json")
        mlflow.log_dict(ANSWER_SCHEMA, "versao/schema.json")
        mlflow.log_text(SYSTEM_PROMPT, "versao/prompt.txt")
        model = mlflow.pyfunc.log_model(
            artifact_path="modelo_rag",
            python_model=RAGModel(qwen, digest, MODEL, revision),
            artifacts={"index": str(STORE), "corpus": str(CORPUS),
                       "web_sources": str(ROOT / "web_sources.json")},
            code_paths=[str(ROOT / "rag"), str(ROOT / "prompts")],
            pip_requirements=requirements,
            metadata=manifest,
        )
        registered = mlflow.register_model(model.model_uri, REGISTERED_MODEL_NAME)
        client = MlflowClient()
        client.set_model_version_tag(REGISTERED_MODEL_NAME, registered.version, "run_id", run.info.run_id)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, registered.version, "contract_sha256", manifest["contract"]["contract_sha256"])
        mlflow.set_tags({"tipo": "modelo_rag", "registered_model_name": REGISTERED_MODEL_NAME,
                         "registered_model_version": str(registered.version)})
        print(json.dumps({"run_id": run.info.run_id, "model_uri": model.model_uri,
                          "registered_model": REGISTERED_MODEL_NAME, "version": registered.version,
                          "ollama_sha256": digest, "index_sha256": index_hash,
                          "corpus_sha256": corpus_hash}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
