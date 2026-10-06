"""Configura o MLflow hospedado pelo repositório DagsHub da equipe."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from rag.corpus import ROOT

REGISTERED_MODEL_NAME = "NexoVerdict-RAG-Analise-Evidencias"


def tracking_uri() -> str:
    load_dotenv(ROOT / ".env")
    explicit = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if explicit:
        return explicit
    repo = os.getenv("DAGSHUB_REPO", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Defina DAGSHUB_REPO=organizacao/repositorio no arquivo .env.")
    return f"https://dagshub.com/{repo}.mlflow"


def configure_tracking() -> str:
    """Usa o URI explícito para testes; caso contrário, usa o DagsHub."""
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    name = os.getenv("MLFLOW_EXPERIMENT_NAME", "NexoVerdict-RAG")
    if mlflow.get_experiment_by_name(name) is None:
        artifact_root = os.getenv("MLFLOW_LOCAL_ARTIFACT_ROOT", "").strip()
        mlflow.create_experiment(
            name,
            artifact_location=Path(artifact_root).resolve().as_uri() if artifact_root else None,
        )
    mlflow.set_experiment(name)
    return uri


def contract_fingerprint(prompt: str, schema: dict) -> dict[str, str]:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()
    validator_hash = hashlib.sha256((ROOT / "rag" / "tutor.py").read_bytes()).hexdigest()
    combined = hashlib.sha256(f"{prompt_hash}:{schema_hash}:{validator_hash}".encode("ascii")).hexdigest()
    return {"prompt_sha256": prompt_hash, "schema_sha256": schema_hash,
            "validator_sha256": validator_hash, "contract_sha256": combined}


def git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "desconhecida"


def matching_model_version(contract_sha256: str, corpus_sha256: str) -> str | None:
    """Retorna a versão registrada que corresponde ao contrato e ao corpus local."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    except Exception:
        return None
    matches = []
    for item in versions:
        run = client.get_run(item.run_id) if item.run_id else None
        params = run.data.params if run else {}
        if params.get("contract_sha256") == contract_sha256 and params.get("corpus_sha256") == corpus_sha256:
            matches.append(item)
    return str(max(matches, key=lambda item: int(item.version)).version) if matches else None
