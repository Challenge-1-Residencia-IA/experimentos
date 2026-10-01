"""Registra no MLflow a versão exata do contrato de resposta, sem chamar o modelo."""

import hashlib
import json
import os

import mlflow
from dotenv import load_dotenv

from rag.corpus import ROOT
from rag.tutor import ANSWER_SCHEMA, RESPONSE_VERSION, SYSTEM_PROMPT


def main() -> None:
    load_dotenv(ROOT / ".env")
    uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"
    name = os.getenv("MLFLOW_EXPERIMENT_NAME", "NexoVerdict-RAG")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlartifacts" / "rag").as_uri())
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"versao-{RESPONSE_VERSION}") as run:
        mlflow.log_params({
            "response_version": RESPONSE_VERSION,
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(json.dumps(ANSWER_SCHEMA, sort_keys=True).encode()).hexdigest(),
        })
        mlflow.log_text(SYSTEM_PROMPT, "versao/prompt.txt")
        mlflow.log_dict(ANSWER_SCHEMA, "versao/schema.json")
        mlflow.set_tag("tipo", "contrato_resposta")
        print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
