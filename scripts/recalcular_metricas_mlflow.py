"""Preenche métricas numéricas em consultas antigas do MLflow local."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from rag.dagshub_tracking import configure_tracking
from rag.metrics import answer_metrics


def main() -> None:
    configure_tracking()
    experiment = mlflow.get_experiment_by_name("NexoVerdict-RAG")
    if experiment is None:
        raise ValueError("Experimento local NexoVerdict-RAG não encontrado")
    client = MlflowClient()
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"])
    updated = 0
    skipped = 0
    for run in runs:
        if run.data.tags.get("mlflow.runName") != "consulta-analise":
            skipped += 1
            continue
        try:
            path = client.download_artifacts(run.info.run_id, "consulta/rastreio.json")
            trace = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        metrics = answer_metrics(
            trace.get("documentos", []),
            trace.get("resposta", {}),
            trace.get("contexto_execucao", {}),
        )
        if run.info.end_time and run.info.start_time:
            metrics.setdefault("duracao_run_s", (run.info.end_time - run.info.start_time) / 1000)
        for name, value in metrics.items():
            client.log_metric(run.info.run_id, name, value)
        client.set_tag(run.info.run_id, "metricas_recalculadas", "true")
        updated += 1
    print(json.dumps({"runs_atualizados": updated, "runs_ignorados": skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
