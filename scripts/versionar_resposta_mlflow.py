"""Publica no MLflow do DagsHub o contrato exato de resposta."""

import hashlib
import os

import mlflow

from rag.corpus import CORPUS, ROOT
from rag.dagshub_tracking import configure_tracking, contract_fingerprint, git_revision
from rag.store import MODEL
from rag.tutor import ANSWER_SCHEMA, RESPONSE_VERSION, SYSTEM_PROMPT


def main() -> None:
    uri = configure_tracking()
    fingerprint = contract_fingerprint(SYSTEM_PROMPT, ANSWER_SCHEMA)
    with mlflow.start_run(run_name=f"versao-{RESPONSE_VERSION}-{fingerprint['contract_sha256'][:12]}") as run:
        mlflow.log_params({
            "response_version": RESPONSE_VERSION,
            **fingerprint,
            "git_revision": git_revision(),
            "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            "embedding_model": MODEL,
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest() if CORPUS.exists() else "ausente",
        })
        mlflow.log_text(SYSTEM_PROMPT, "versao/prompt.txt")
        mlflow.log_dict(ANSWER_SCHEMA, "versao/schema.json")
        mlflow.log_artifact(str(ROOT / "rag" / "tutor.py"), artifact_path="versao")
        mlflow.log_artifact(str(ROOT / "requirements-rag.txt"), artifact_path="versao")
        mlflow.set_tag("tipo", "contrato_resposta")
        print(f"DagsHub: {uri}\nVersão: {RESPONSE_VERSION} ({fingerprint['contract_sha256'][:12]})\nMLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
