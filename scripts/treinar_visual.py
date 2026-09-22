import sys
import re
import subprocess
import argparse
import os
from pathlib import Path
import mlflow
from dotenv import load_dotenv
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data/processed"
ADAPTERS_DIR = ROOT_DIR / "models/adapters"

ITERS_TOTAIS = 400


def configurar_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configura o tracking local e cria o experimento quando necessário."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    if client.get_experiment_by_name(experiment_name) is None:
        if tracking_uri.startswith(("http://", "https://", "databricks")):
            client.create_experiment(experiment_name)
        else:
            artifact_location = (ROOT_DIR / "mlartifacts" / experiment_name).resolve().as_uri()
            client.create_experiment(experiment_name, artifact_location=artifact_location)

    mlflow.set_experiment(experiment_name)


def arquivo_uri(path: Path) -> str:
    """Retorna a URI SQLite absoluta para o backend local padrão."""
    return f"sqlite:///{path.resolve()}"


def montar_comando(data_dir: Path, adapters_dir: Path, modelo: str, iteracoes: int):
    return [
    sys.executable, "-m", "mlx_lm", "lora",
    "--model", modelo,
    "--data", str(data_dir),
    "--adapter-path", str(adapters_dir),
    "--train",
    "--iters", str(iteracoes),
    "--batch-size", "1",
    "--learning-rate", "2e-5",
    "--num-layers", "16",
    "--max-seq-length", "512",
    "--grad-checkpoint",
    "--val-batches", "5",
    "--steps-per-eval", "100",
    "--save-every", "100",
    "--steps-per-report", "1"
    ]

def main(
    data_dir: Path,
    adapters_dir: Path,
    modelo: str,
    iteracoes: int,
    tracking_uri: str,
    experiment_name: str,
):
    if not (data_dir / "train.jsonl").exists() or not (data_dir / "valid.jsonl").exists():
        raise FileNotFoundError(f"Esperados train.jsonl e valid.jsonl em '{data_dir}'.")

    configurar_mlflow(tracking_uri, experiment_name)
    comando = montar_comando(data_dir, adapters_dir, modelo, iteracoes)
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("• Faltam:"),
        TimeRemainingColumn(),
    )

    task_id = progress.add_task("Treinando Sift LLM...", total=iteracoes)

    metrics_text = Text("Iniciando carregamento do modelo...", style="yellow")

    def make_layout():
        layout = Layout()
        layout.split(
            Layout(Panel(progress, title="[bold green]Sift AI - Fine-Tuning Progress[/bold green]"), size=5),
            Layout(Panel(metrics_text, title="[bold blue]Métricas em Tempo Real[/bold blue]"), size=6)
        )
        return layout

    re_train = re.compile(r"Iter (\d+): Train loss ([\d\.]+).*?It/sec ([\d\.]+).*?Peak mem ([\d\.]+) GB")
    re_val = re.compile(r"Iter (\d+): Val loss ([\d\.]+)")

    with mlflow.start_run(run_name=f"lora-{modelo.rsplit('/', 1)[-1]}"):
        mlflow.log_params({
            "model": modelo,
            "iterations": iteracoes,
            "data_dir": str(data_dir.resolve()),
            "adapters_dir": str(adapters_dir.resolve()),
            "batch_size": 1,
            "learning_rate": "2e-5",
            "num_layers": 16,
            "max_seq_length": 512,
            "grad_checkpoint": True,
        })
        mlflow.set_tags({"framework": "mlx-lm", "task": "lora_fine_tuning"})

        with Live(make_layout(), refresh_per_second=4) as live:
            process = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            val_loss = "N/A"
            ultimas_linhas = []

            for line in iter(process.stdout.readline, ''):
                ultimas_linhas.append(line.rstrip())
                ultimas_linhas = ultimas_linhas[-20:]
                match_train = re_train.search(line)
                if match_train:
                    iter_atual = int(match_train.group(1))
                    train_loss = float(match_train.group(2))
                    it_sec = float(match_train.group(3))
                    mem = float(match_train.group(4))

                    progress.update(task_id, completed=iter_atual)
                    mlflow.log_metrics(
                        {"train_loss": train_loss, "iterations_per_second": it_sec, "peak_memory_gb": mem},
                        step=iter_atual,
                    )

                    metrics_text = Text.from_markup(
                        f" Iteração: [bold white]{iter_atual}/{iteracoes}[/bold white]\n"
                        f" Loss Treino: [bold green]{train_loss}[/bold green] | Loss Validação: [bold yellow]{val_loss}[/bold yellow]\n"
                        f" Velocidade: [bold cyan]{it_sec} iter/s[/bold cyan] | RAM Alocada: [bold magenta]{mem} GB[/bold magenta]"
                    )
                    live.update(make_layout())

                match_val = re_val.search(line)
                if match_val:
                    iter_atual = int(match_val.group(1))
                    val_loss = float(match_val.group(2))
                    mlflow.log_metric("val_loss", val_loss, step=iter_atual)

            retorno = process.wait()
            if retorno != 0:
                diagnostico = "\n".join(linha for linha in ultimas_linhas if linha)
                raise RuntimeError(
                    f"O treinamento MLX terminou com código {retorno}. "
                    f"Saída final:\n{diagnostico}"
                )

        if adapters_dir.exists():
            mlflow.log_artifacts(str(adapters_dir), artifact_path="adapters")

    print("\n[✓] Treinamento concluído com sucesso!")

if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Executa fine-tuning LoRA do Qwen com painel de progresso.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--adapters-dir", type=Path, default=ADAPTERS_DIR)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--iters", type=int, default=ITERS_TOTAIS)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI") or arquivo_uri(ROOT_DIR / "mlflow.db"),
        help="URI do servidor/backend MLflow (padrão: SQLite local).",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or "sift-lora-fine-tuning",
        help="Nome do experimento MLflow.",
    )
    args = parser.parse_args()
    if args.iters <= 0:
        parser.error("--iters deve ser positivo.")
    main(
        args.data_dir,
        args.adapters_dir,
        args.model,
        args.iters,
        args.tracking_uri,
        args.experiment_name,
    )
