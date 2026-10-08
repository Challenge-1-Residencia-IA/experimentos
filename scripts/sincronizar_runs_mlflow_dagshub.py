"""Copia runs do MLflow local para o DagsHub sem reenviar artefatos com textos do corpus."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from mlflow.entities import Param, ViewType
from mlflow.tracking import MlflowClient

from rag.corpus import ROOT
from rag.dagshub_tracking import dagshub_tracking_uri

DEFAULT_DB = ROOT / "mlflow.db"
DEFAULT_EXPERIMENT = "NexoVerdict-RAG"
SYNC_TAG = "source_local_run_id"


def local_tracking_uri(database: Path) -> str:
    return f"sqlite:///{database.resolve()}"


def selected_runs(client: MlflowClient, experiment_id: str, run_ids: list[str]) -> list:
    runs = client.search_runs(
        [experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attribute.start_time ASC"],
        max_results=50_000,
    )
    if run_ids:
        selected = set(run_ids)
        available = {run.info.run_id for run in runs}
        missing = sorted(selected - available)
        if missing:
            raise ValueError("Runs locais não encontrados: " + ", ".join(missing))
        runs = [run for run in runs if run.info.run_id in selected]
    return runs


def local_experiments(database: Path) -> list[dict]:
    """Lista experimentos locais com runs, sem acessar o DagsHub."""
    if not database.is_file():
        raise FileNotFoundError(f"Banco MLflow local não encontrado: {database}")
    client = MlflowClient(tracking_uri=local_tracking_uri(database))
    experiments = []
    for experiment in client.search_experiments(view_type=ViewType.ACTIVE_ONLY, max_results=50_000):
        runs = selected_runs(client, experiment.experiment_id, [])
        if runs:
            experiments.append({
                "name": experiment.name,
                "runs": len(runs),
                "runs_with_metrics": sum(bool(run.data.metrics) for run in runs),
            })
    return sorted(experiments, key=lambda item: item["name"].casefold())


def public_tags(tags: dict[str, str]) -> dict[str, str]:
    """Remove tags internas recriadas pelo servidor remoto."""
    return {key: value for key, value in tags.items() if not key.startswith("mlflow.")}


def summary(run) -> dict:
    return {
        "source_local_run_id": run.info.run_id,
        "source_experiment_id": run.info.experiment_id,
        "source_status": run.info.status,
        "source_start_time": run.info.start_time,
        "source_end_time": run.info.end_time,
        "run_name": run.data.tags.get("mlflow.runName", run.info.run_id),
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": public_tags(dict(run.data.tags)),
        "artifacts_copied": False,
        "artifact_policy": (
            "Somente este resumo foi enviado. Artefatos locais não foram copiados "
            "porque podem conter trechos do corpus com licença ainda não confirmada."
        ),
    }


def already_synced(client: MlflowClient, experiment_id: str, local_run_id: str):
    matches = client.search_runs(
        [experiment_id],
        filter_string=f"tags.{SYNC_TAG} = '{local_run_id}'",
        max_results=1,
    )
    return matches[0] if matches else None


def chunks(values: list, size: int) -> list[list]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def copy_run(
    local: MlflowClient,
    remote: MlflowClient,
    remote_experiment_id: str,
    run,
    *,
    remote_run_id: str | None = None,
) -> str:
    name = run.data.tags.get("mlflow.runName", run.info.run_id)
    tags = {
        **public_tags(dict(run.data.tags)),
        SYNC_TAG: run.info.run_id,
        "source_experiment_id": str(run.info.experiment_id),
        "source_status": str(run.info.status),
        "sync_mode": "metrics_params_tags_summary_only",
        "artifacts_with_corpus_excluded": "true",
    }
    if remote_run_id is None:
        created = remote.create_run(
            remote_experiment_id,
            start_time=run.info.start_time,
            tags=tags,
            run_name=name,
        )
        remote_run_id = created.info.run_id
    try:
        params = [Param(key, value) for key, value in run.data.params.items()]
        histories = []
        for key in run.data.metrics:
            histories.extend(local.get_metric_history(run.info.run_id, key))
        first_metrics = histories[:1000]
        remote.log_batch(remote_run_id, metrics=first_metrics, params=params[:100])
        for batch in chunks(histories[1000:], 1000):
            remote.log_batch(remote_run_id, metrics=batch)
        for batch in chunks(params[100:], 100):
            remote.log_batch(remote_run_id, params=batch)
        payload = summary(run)
        with tempfile.TemporaryDirectory(prefix="nexoverdict-mlflow-sync-") as directory:
            path = Path(directory) / "resumo_run_local.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            remote.log_artifact(remote_run_id, str(path), artifact_path="importacao")
        status = run.info.status if run.info.status in {"FINISHED", "FAILED", "KILLED"} else "FINISHED"
        remote.set_terminated(remote_run_id, status=status, end_time=run.info.end_time)
    except Exception:
        remote.set_terminated(remote_run_id, status="FAILED")
        raise
    return remote_run_id


def sync(
    database: Path,
    experiment_name: str,
    run_ids: list[str],
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict]:
    if not database.is_file():
        raise FileNotFoundError(f"Banco MLflow local não encontrado: {database}")
    local = MlflowClient(tracking_uri=local_tracking_uri(database))
    local_experiment = local.get_experiment_by_name(experiment_name)
    if local_experiment is None:
        raise ValueError(f"Experimento local não encontrado: {experiment_name}")
    runs = selected_runs(local, local_experiment.experiment_id, run_ids)
    if dry_run:
        return [{
            "local_run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", run.info.run_id),
            "status": "seria_enviado",
        } for run in runs]

    load_dotenv(ROOT / ".env")
    remote_uri = dagshub_tracking_uri()
    remote = MlflowClient(tracking_uri=remote_uri)
    remote_experiment = remote.get_experiment_by_name(experiment_name)
    remote_experiment_id = (
        remote_experiment.experiment_id
        if remote_experiment is not None
        else remote.create_experiment(experiment_name)
    )
    results = []
    for run in runs:
        existing = already_synced(remote, remote_experiment_id, run.info.run_id)
        if existing and existing.info.status == "FINISHED" and not force:
            results.append({
                "local_run_id": run.info.run_id,
                "remote_run_id": existing.info.run_id,
                "status": "já_sincronizado",
            })
            continue
        resumable_id = existing.info.run_id if existing and not force else None
        remote_run_id = copy_run(
            local, remote, remote_experiment_id, run, remote_run_id=resumable_id
        )
        results.append({
            "local_run_id": run.info.run_id,
            "remote_run_id": remote_run_id,
            "status": "enviado",
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-id", action="append", default=[], help="Run específico; repita para vários")
    parser.add_argument("--dry-run", action="store_true", help="Lista os runs sem conectar ao DagsHub")
    parser.add_argument("--force", action="store_true", help="Cria outra cópia mesmo se o run já foi sincronizado")
    args = parser.parse_args()
    results = sync(
        args.local_db.resolve(), args.experiment_name, args.run_id,
        dry_run=args.dry_run, force=args.force,
    )
    print(json.dumps({
        "modo": "metricas_parametros_tags_e_resumo",
        "artefatos_com_texto_copiados": False,
        "runs": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
