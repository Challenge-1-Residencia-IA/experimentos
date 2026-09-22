import sys
import re
import subprocess
import argparse
from pathlib import Path
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data/processed"
ADAPTERS_DIR = ROOT_DIR / "models/adapters"

ITERS_TOTAIS = 400
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

def main(data_dir: Path, adapters_dir: Path, modelo: str, iteracoes: int):
    if not (data_dir / "train.jsonl").exists() or not (data_dir / "valid.jsonl").exists():
        raise FileNotFoundError(f"Esperados train.jsonl e valid.jsonl em '{data_dir}'.")

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

    with Live(make_layout(), refresh_per_second=4) as live:
        process = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        val_loss = "N/A"

        for line in iter(process.stdout.readline, ''):
            match_train = re_train.search(line)
            if match_train:
                iter_atual = int(match_train.group(1))
                train_loss = match_train.group(2)
                it_sec = match_train.group(3)
                mem = match_train.group(4)

                progress.update(task_id, completed=iter_atual)

                metrics_text = Text.from_markup(
                    f" Iteração: [bold white]{iter_atual}/{iteracoes}[/bold white]\n"
                    f" Loss Treino: [bold green]{train_loss}[/bold green] | Loss Validação: [bold yellow]{val_loss}[/bold yellow]\n"
                    f" Velocidade: [bold cyan]{it_sec} iter/s[/bold cyan] | RAM Alocada: [bold magenta]{mem} GB[/bold magenta]"
                )
                live.update(make_layout())

            match_val = re_val.search(line)
            if match_val:
                val_loss = match_val.group(2)

        retorno = process.wait()
        if retorno != 0:
            raise RuntimeError(f"O treinamento MLX terminou com código {retorno}.")

    print("\n[✓] Treinamento concluído com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa fine-tuning LoRA do Qwen com painel de progresso.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--adapters-dir", type=Path, default=ADAPTERS_DIR)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--iters", type=int, default=ITERS_TOTAIS)
    args = parser.parse_args()
    if args.iters <= 0:
        parser.error("--iters deve ser positivo.")
    main(args.data_dir, args.adapters_dir, args.model, args.iters)
