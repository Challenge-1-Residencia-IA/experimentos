"""Configura MLflow local para testes e DagsHub para publicações compartilhadas."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException

from rag.corpus import ROOT

REGISTERED_MODEL_NAME = "NexoVerdict-RAG-Analise-Evidencias"


def tracking_uri() -> str:
    """Retorna o destino das execuções cotidianas, local por padrão."""
    load_dotenv(ROOT / ".env")
    explicit = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if explicit:
        return explicit
    return f"sqlite:///{ROOT / 'mlflow.db'}"


def dagshub_tracking_uri() -> str:
    """Retorna o servidor remoto usado para modelos e datasets publicados."""
    load_dotenv(ROOT / ".env")
    repo = os.getenv("DAGSHUB_REPO", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Defina DAGSHUB_REPO=organizacao/repositorio no arquivo .env.")
    return f"https://dagshub.com/{repo}.mlflow"


def configure_tracking(*, remote: bool = False) -> str:
    """Configura o MLflow local, ou o DagsHub quando ``remote=True``."""
    uri = dagshub_tracking_uri() if remote else tracking_uri()
    mlflow.set_tracking_uri(uri)
    name = os.getenv("MLFLOW_EXPERIMENT_NAME", "NexoVerdict-RAG")
    if mlflow.get_experiment_by_name(name) is None:
        artifact_root = os.getenv("MLFLOW_LOCAL_ARTIFACT_ROOT", "").strip()
        if not remote and not artifact_root:
            artifact_root = str(ROOT / "mlartifacts" / "rag")
        try:
            mlflow.create_experiment(
                name,
                artifact_location=Path(artifact_root).resolve().as_uri() if artifact_root else None,
            )
        except MlflowException:
            if mlflow.get_experiment_by_name(name) is None:
                raise
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


def tracked_registered_prediction(
    question: str,
    mode: str,
    version: str,
    predict: Callable[[], str],
) -> tuple[str, str]:
    """Executa uma versão registrada e salva toda tentativa no MLflow padrão."""
    configure_tracking()
    started = time.perf_counter()
    with mlflow.start_run(run_name="consulta-modelo-registrado") as run:
        mlflow.log_params({
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": version,
            "model_uri": f"models:/{REGISTERED_MODEL_NAME}/{version}",
            "modo": mode,
            "git_revision": git_revision(),
        })
        mlflow.log_dict({"pergunta": question, "modo": mode}, "consulta/entrada.json")
        try:
            output = predict()
            answer = json.loads(output)
        except Exception as error:
            mlflow.log_metric("latencia_s", time.perf_counter() - started)
            mlflow.set_tags({"resultado": "erro", "erro_tipo": type(error).__name__})
            mlflow.log_text(str(error), "consulta/erro.txt")
            raise
        mlflow.log_metric("latencia_s", time.perf_counter() - started)
        mlflow.log_dict(answer, "consulta/resposta.json")
        mlflow.set_tags({
            "resultado": "sucesso",
            "avaliacao_evidencias": str(answer.get("avaliacao_evidencias", "ausente")),
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": version,
        })
        return output, run.info.run_id
