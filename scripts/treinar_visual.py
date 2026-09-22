import sys
import re
import subprocess
import argparse
import os
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import mlflow
from dotenv import load_dotenv
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data/evidencias"
ADAPTERS_DIR = ROOT_DIR / "models/adapters-evidencias"

ITERS_TOTAIS = 100


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


def validar_jsonl(caminho: Path) -> int:
    """Falha cedo para evitar iniciar um treino caro com dados malformados."""
    total = 0
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, 1):
            try:
                item = json.loads(linha)
            except json.JSONDecodeError as erro:
                raise ValueError(f"JSON invalido em {caminho}:{numero}: {erro}") from erro
            mensagens = item.get("messages")
            if not isinstance(mensagens, list) or len(mensagens) < 2:
                raise ValueError(f"Campo messages invalido em {caminho}:{numero}.")
            if mensagens[-1].get("role") != "assistant" or not mensagens[-1].get("content"):
                raise ValueError(f"Resposta assistant ausente em {caminho}:{numero}.")
            total += 1
    if total == 0:
        raise ValueError(f"Dataset vazio: '{caminho}'.")
    return total


def garantir_modelo_disponivel(modelo: str) -> None:
    """Baixa o snapshot antes do painel para exibir progresso e erros de rede."""
    if Path(modelo).expanduser().exists():
        print(f"[✓] Modelo local encontrado: {modelo}")
        return
    from huggingface_hub import snapshot_download

    print(f"[↓] Conferindo/baixando modelo '{modelo}' no cache do Hugging Face...", flush=True)
    caminho = snapshot_download(repo_id=modelo)
    print(f"[✓] Modelo disponível em: {caminho}", flush=True)


def montar_comando(args: argparse.Namespace):
    comando = [
    sys.executable, "-u", "-m", "mlx_lm", "lora",
    "--model", args.model,
    "--data", str(args.data_dir),
    "--adapter-path", str(args.adapters_dir),
    "--train",
    "--iters", str(args.iters),
    "--batch-size", str(args.batch_size),
    "--learning-rate", str(args.learning_rate),
    "--num-layers", str(args.num_layers),
    "--max-seq-length", str(args.max_seq_length),
    "--val-batches", str(args.val_batches),
    "--steps-per-eval", str(args.steps_per_eval),
    "--save-every", str(args.save_every),
    "--steps-per-report", str(args.steps_per_report),
    "--grad-accumulation-steps", str(args.grad_accumulation_steps),
    "--seed", str(args.seed),
    ]
    if args.grad_checkpoint:
        comando.append("--grad-checkpoint")
    if args.mask_prompt:
        comando.append("--mask-prompt")
    if args.resume_adapter_file:
        comando.extend(["--resume-adapter-file", str(args.resume_adapter_file)])
    if args.evaluate_test:
        comando.extend(["--test", "--test-batches", "-1"])
    return comando

def main(args: argparse.Namespace):
    if not (args.data_dir / "train.jsonl").exists() or not (args.data_dir / "valid.jsonl").exists():
        raise FileNotFoundError(f"Esperados train.jsonl e valid.jsonl em '{args.data_dir}'.")
    total_treino = validar_jsonl(args.data_dir / "train.jsonl")
    total_validacao = validar_jsonl(args.data_dir / "valid.jsonl")
    total_teste = validar_jsonl(args.data_dir / "test.jsonl") if (args.data_dir / "test.jsonl").exists() else 0
    adaptador_final = args.adapters_dir / "adapters.safetensors"
    if adaptador_final.exists() and not args.resume_adapter_file and not args.overwrite:
        raise FileExistsError(
            f"'{adaptador_final}' ja existe. Use outro --adapters-dir, --resume-adapter-file "
            "ou --overwrite conscientemente."
        )
    if args.resume_adapter_file and not args.resume_adapter_file.exists():
        raise FileNotFoundError(f"Checkpoint para retomada nao encontrado: '{args.resume_adapter_file}'.")
    if not args.skip_model_download:
        garantir_modelo_disponivel(args.model)

    configurar_mlflow(args.tracking_uri, args.experiment_name)
    comando = montar_comando(args)
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("• Faltam:"),
        TimeRemainingColumn(),
    )

    task_id = progress.add_task("Treinando Sift LLM...", total=args.iters)

    metrics_text = Text("Iniciando carregamento do modelo...", style="yellow")

    def make_layout():
        layout = Layout()
        layout.split(
            Layout(Panel(progress, title="[bold green]Sift AI - Fine-Tuning Progress[/bold green]"), size=5),
            Layout(Panel(metrics_text, title="[bold blue]Métricas em Tempo Real[/bold blue]"), size=6)
        )
        return layout

    numero = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    re_train = re.compile(
        rf"Iter (\d+): Train loss ({numero}).*?It/sec ({numero}).*?Peak mem ({numero}) GB"
    )
    re_val = re.compile(rf"Iter (\d+): Val loss ({numero})")

    historico_validacao = []
    with mlflow.start_run(run_name=f"lora-{args.model.rsplit('/', 1)[-1]}"):
        mlflow.log_params({
            "model": args.model,
            "iterations": args.iters,
            "data_dir": str(args.data_dir.resolve()),
            "adapters_dir": str(args.adapters_dir.resolve()),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_layers": args.num_layers,
            "max_seq_length": args.max_seq_length,
            "grad_checkpoint": args.grad_checkpoint,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "mask_prompt": args.mask_prompt,
            "seed": args.seed,
            "resume_adapter_file": str(args.resume_adapter_file or ""),
            "train_examples": total_treino,
            "validation_examples": total_validacao,
            "test_examples": total_teste,
        })
        mlflow.set_tags({"framework": "mlx-lm", "task": "lora_fine_tuning"})

        with Live(make_layout(), refresh_per_second=4) as live:
            ambiente = os.environ.copy()
            ambiente["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                comando,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=ambiente,
            )

            val_loss = "N/A"
            ultimas_linhas = []

            for line in iter(process.stdout.readline, ''):
                linha_limpa = line.strip()
                ultimas_linhas.append(linha_limpa)
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
                        f" Iteração: [bold white]{iter_atual}/{args.iters}[/bold white]\n"
                        f" Loss Treino: [bold green]{train_loss}[/bold green] | Loss Validação: [bold yellow]{val_loss}[/bold yellow]\n"
                        f" Velocidade: [bold cyan]{it_sec} iter/s[/bold cyan] | RAM Alocada: [bold magenta]{mem} GB[/bold magenta]"
                    )
                    live.update(make_layout())

                elif linha_limpa:
                    metrics_text = Text(
                        "Carregando/preparando o modelo...\n" + linha_limpa[-500:],
                        style="yellow",
                    )
                    live.update(make_layout())

                match_val = re_val.search(line)
                if match_val:
                    iter_atual = int(match_val.group(1))
                    val_loss = float(match_val.group(2))
                    historico_validacao.append({"iteracao": iter_atual, "val_loss": val_loss})
                    mlflow.log_metric("val_loss", val_loss, step=iter_atual)
                    metrics_text = Text(
                        f"Iteração: {iter_atual}/{args.iters}\nLoss de validação: {val_loss}",
                        style="yellow",
                    )
                    live.update(make_layout())

            retorno = process.wait()
            if retorno != 0:
                diagnostico = "\n".join(linha for linha in ultimas_linhas if linha)
                raise RuntimeError(
                    f"O treinamento MLX terminou com código {retorno}. "
                    f"Saída final:\n{diagnostico}"
                )

        melhor = min(historico_validacao, key=lambda item: item["val_loss"], default=None)
        melhor_checkpoint = None
        if melhor:
            candidato = args.adapters_dir / f"{melhor['iteracao']:07d}_adapters.safetensors"
            if candidato.exists():
                melhor_dir = args.adapters_dir / "best"
                melhor_dir.mkdir(parents=True, exist_ok=True)
                melhor_checkpoint = melhor_dir / "adapters.safetensors"
                shutil.copy2(candidato, melhor_checkpoint)
                config_adaptador = args.adapters_dir / "adapter_config.json"
                if config_adaptador.exists():
                    shutil.copy2(config_adaptador, melhor_dir / "adapter_config.json")
        resumo = {
            "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
            "comando": comando,
            "historico_validacao": historico_validacao,
            "melhor_validacao": melhor,
            "melhor_checkpoint": str(melhor_checkpoint) if melhor_checkpoint else None,
            "observacao": "Compare tambem as metricas de tarefa; menor val_loss nao garante melhor verificacao.",
        }
        args.adapters_dir.mkdir(parents=True, exist_ok=True)
        resumo_path = args.adapters_dir / "training_summary.json"
        resumo_path.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.adapters_dir.exists():
            mlflow.log_artifacts(str(args.adapters_dir), artifact_path="adapters")

    print("\n[✓] Treinamento concluído com sucesso!")

if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Executa fine-tuning LoRA do Qwen com painel de progresso.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--adapters-dir", type=Path, default=ADAPTERS_DIR)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--iters", type=int, default=ITERS_TOTAIS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=-1)
    parser.add_argument("--steps-per-eval", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--steps-per-report", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-adapter-file", type=Path)
    parser.add_argument("--no-mask-prompt", action="store_false", dest="mask_prompt")
    parser.add_argument("--no-grad-checkpoint", action="store_false", dest="grad_checkpoint")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Pula a etapa explícita de download; útil quando --model é gerenciado externamente.",
    )
    parser.set_defaults(mask_prompt=True, grad_checkpoint=True)
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
    positivos = {
        "--iters": args.iters,
        "--batch-size": args.batch_size,
        "--max-seq-length": args.max_seq_length,
        "--steps-per-eval": args.steps_per_eval,
        "--save-every": args.save_every,
        "--steps-per-report": args.steps_per_report,
        "--grad-accumulation-steps": args.grad_accumulation_steps,
    }
    invalidos = [nome for nome, valor in positivos.items() if valor <= 0]
    if invalidos or args.learning_rate <= 0 or args.val_batches == 0:
        parser.error(f"Parametros numericos invalidos: {', '.join(invalidos) or '--learning-rate/--val-batches'}")
    main(args)
