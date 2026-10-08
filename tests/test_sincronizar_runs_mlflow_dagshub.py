"""Descoberta dos experimentos locais, sem qualquer acesso remoto."""

from mlflow.tracking import MlflowClient

from scripts.sincronizar_runs_mlflow_dagshub import local_experiments, local_tracking_uri


def test_local_experiments_ignora_vazios_e_conta_runs_com_metricas(tmp_path):
    database = tmp_path / "mlflow.db"
    client = MlflowClient(tracking_uri=local_tracking_uri(database))
    client.create_experiment("sem-runs")
    experiment_id = client.create_experiment("com-runs")
    run_with_metric = client.create_run(experiment_id)
    client.log_metric(run_with_metric.info.run_id, "score", 0.5)
    client.create_run(experiment_id)

    assert local_experiments(database) == [
        {"name": "com-runs", "runs": 2, "runs_with_metrics": 1}
    ]
