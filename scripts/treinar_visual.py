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
DATA_DIR = ROOT_DIR / "data/evidencias-por-url-v1"
ADAPTERS_DIR = ROOT_DIR / "models/adapters-evidencias-por-url-v1"

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


def montar_comando(args: argparse.Namespace) -> list[str]:
    """Monta o comando do MLX-LM a partir das opções do treinador."""
    comando = [
        sys.executable,
        "-u",
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        args.model,
        "--data",
        str(args.data_dir),
        "--adapter-path",
        str(args.adapters_dir),
        "--train",
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--num-layers",
        str(args.num_layers),
        "--max-seq-length",
        str(args.max_seq_length),
        "--val-batches",
        str(args.val_batches),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--save-every",
        str(args.save_every),
        "--steps-per-report",
        str(args.steps_per_report),
        "--grad-accumulation-steps",
        str(args.grad_accumulation_steps),
        "--seed",
        str(args.seed),
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


def criar_painel(progresso: Progress, texto_metricas: Text) -> Layout:
    """Cria a vista de progresso mostrada enquanto o subprocesso treina."""
    layout = Layout()
    layout.split(
        Layout(Panel(progresso, title="[bold green]NexoVerdict IA — Progresso do ajuste fino[/bold green]"), size=5),
        Layout(Panel(texto_metricas, title="[bold blue]Métricas em tempo real[/bold blue]"), size=6),
    )
    return layout


def regex_metricas() -> tuple[re.Pattern, re.Pattern]:
    """Lê a saída do MLX-LM, cujos nomes de métricas são emitidos em inglês."""
    numero = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    treino = re.compile(
        rf"Iter (\d+): Train loss ({numero}).*?It/sec ({numero}).*?Peak mem ({numero}) GB"
    )
    validacao = re.compile(rf"Iter (\d+): Val loss ({numero})")
    return treino, validacao


def limpar_saida_terminal(texto: str) -> str:
    """Remove controles ANSI/retornos usados pelas barras do MLX antes do parsing."""
    sem_ansi = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", texto)
    return sem_ansi.replace("\r", " ").strip()


def guardar_melhor_checkpoint(
    adapters_dir: Path, historico_validacao: list[dict]
) -> Path | None:
    """Copia o checkpoint com menor perda de validação para adapters_dir/best."""
    melhor = min(historico_validacao, key=lambda item: item["perda_validacao"], default=None)
    if melhor is None:
        return None

    checkpoint = adapters_dir / f"{melhor['iteracao']:07d}_adapters.safetensors"
    if not checkpoint.exists():
        return None

    melhor_dir = adapters_dir / "best"
    melhor_dir.mkdir(parents=True, exist_ok=True)
    destino = melhor_dir / "adapters.safetensors"
    shutil.copy2(checkpoint, destino)

    config = adapters_dir / "adapter_config.json"
    if config.exists():
        shutil.copy2(config, melhor_dir / "adapter_config.json")
    return destino


def salvar_resumo(
    args: argparse.Namespace,
    comando: list[str],
    historico_validacao: list[dict],
    melhor_checkpoint: Path | None,
) -> Path:
    """Salva parâmetros e histórico do treino ao lado dos adaptadores."""
    melhor = min(historico_validacao, key=lambda item: item["perda_validacao"], default=None)
    resumo = {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "comando": comando,
        "historico_validacao": historico_validacao,
        "melhor_validacao": melhor,
        "melhor_checkpoint": str(melhor_checkpoint) if melhor_checkpoint else None,
        "observacao": "Compare também as métricas da tarefa; menor perda de validação não garante melhor verificação.",
    }
    args.adapters_dir.mkdir(parents=True, exist_ok=True)
    caminho = args.adapters_dir / "training_summary.json"
    caminho.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho

def main(args: argparse.Namespace):
    if not (args.data_dir / "train.jsonl").exists() or not (args.data_dir / "valid.jsonl").exists():
        raise FileNotFoundError(f"Esperados train.jsonl e valid.jsonl em '{args.data_dir}'.")
    total_treino = validar_jsonl(args.data_dir / "train.jsonl")
    total_validacao = validar_jsonl(args.data_dir / "valid.jsonl")
    total_teste = validar_jsonl(args.data_dir / "test.jsonl") if (args.data_dir / "test.jsonl").exists() else 0
    lotes_validacao = total_validacao if args.val_batches == -1 else min(args.val_batches, total_validacao)
    print(
        f"[i] Dataset: {total_treino} treino, {total_validacao} validação, {total_teste} teste.\n"
        f"[i] Antes da primeira iteração, o MLX calculará a perda em {lotes_validacao} lote(s) de validação.",
        flush=True,
    )
    if args.val_batches == -1 and total_validacao > 100:
        print(
            "[!] --val-batches=-1 usa toda a validação e pode deixar o progresso em 0% por muito tempo. "
            "Interrompa e use --val-batches 5 para smoke test ou 25 para treino normal se isso não foi intencional.",
            flush=True,
        )
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
    progresso = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("• Faltam:"),
        TimeRemainingColumn(),
    )

    id_tarefa = progresso.add_task("Ajustando o modelo NexoVerdict...", total=args.iters)

    texto_metricas = Text(
        f"Carregando o modelo. Em seguida será feita a validação inicial em {lotes_validacao} lote(s); "
        "a barra de treino permanece em 0% durante essa etapa.",
        style="yellow",
    )

    padrao_treino, padrao_validacao = regex_metricas()
    historico_validacao = []
    with mlflow.start_run(run_name=f"ajuste-fino-lora-{args.model.rsplit('/', 1)[-1]}"):
        mlflow.log_params({
            "modelo": args.model,
            "iteracoes": args.iters,
            "diretorio_dados": str(args.data_dir.resolve()),
            "diretorio_adaptadores": str(args.adapters_dir.resolve()),
            "tamanho_lote": args.batch_size,
            "taxa_aprendizado": args.learning_rate,
            "numero_camadas": args.num_layers,
            "comprimento_maximo_sequencia": args.max_seq_length,
            "checkpoint_gradiente": args.grad_checkpoint,
            "passos_acumulacao_gradiente": args.grad_accumulation_steps,
            "mascarar_prompt": args.mask_prompt,
            "semente": args.seed,
            "arquivo_retomada_adaptador": str(args.resume_adapter_file or ""),
            "exemplos_treinamento": total_treino,
            "exemplos_validacao": total_validacao,
            "exemplos_teste": total_teste,
        })
        mlflow.set_tags({"biblioteca": "mlx-lm", "tarefa": "ajuste_fino_lora"})

        with Live(criar_painel(progresso, texto_metricas), refresh_per_second=4) as live:
            args.adapters_dir.mkdir(parents=True, exist_ok=True)
            log_treino = args.adapters_dir / "training_output.log"
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

            perda_validacao = "Ainda não calculada"
            ultimas_linhas = []
            erro_numerico = None

            with log_treino.open("w", encoding="utf-8") as arquivo_log:
                for line in iter(process.stdout.readline, ''):
                    arquivo_log.write(line)
                    arquivo_log.flush()
                    linha_limpa = limpar_saida_terminal(line)
                    ultimas_linhas.append(linha_limpa)
                    ultimas_linhas = ultimas_linhas[-20:]
                    if re.search(r"Iter\s+\d+:\s+(?:Train|Val) loss\s+(?:nan|inf)\b", linha_limpa, re.IGNORECASE):
                        erro_numerico = linha_limpa
                        process.terminate()
                        break
                    match_train = padrao_treino.search(linha_limpa)
                    if match_train:
                        iter_atual = int(match_train.group(1))
                        perda_treinamento = float(match_train.group(2))
                        iteracoes_por_segundo = float(match_train.group(3))
                        memoria_gb = float(match_train.group(4))

                        progresso.update(id_tarefa, completed=iter_atual, description="Ajustando o modelo NexoVerdict...")
                        mlflow.log_metrics(
                            {
                                "perda_treinamento": perda_treinamento,
                                "iteracoes_por_segundo": iteracoes_por_segundo,
                                "pico_memoria_gb": memoria_gb,
                            },
                            step=iter_atual,
                        )

                        texto_metricas = Text.from_markup(
                            f" Iteração: [bold white]{iter_atual}/{args.iters}[/bold white]\n"
                            f" Perda de treinamento: [bold green]{perda_treinamento}[/bold green] | "
                            f"Perda de validação: [bold yellow]{perda_validacao}[/bold yellow]\n"
                            f" Velocidade: [bold cyan]{iteracoes_por_segundo} iterações/s[/bold cyan] | "
                            f"Pico de memória: [bold magenta]{memoria_gb} GB[/bold magenta]"
                        )
                        live.update(criar_painel(progresso, texto_metricas))

                    elif linha_limpa:
                        texto_metricas = Text(
                            "Carregando/preparando o modelo...\n" + linha_limpa[-500:],
                            style="yellow",
                        )
                        live.update(criar_painel(progresso, texto_metricas))

                    match_val = padrao_validacao.search(linha_limpa)
                    if match_val:
                        iter_atual = int(match_val.group(1))
                        perda_validacao = float(match_val.group(2))
                        historico_validacao.append({"iteracao": iter_atual, "perda_validacao": perda_validacao})
                        mlflow.log_metric("perda_validacao", perda_validacao, step=iter_atual)
                        texto_metricas = Text(
                            f"Iteração: {iter_atual}/{args.iters}\nPerda de validação: {perda_validacao}",
                            style="yellow",
                        )
                        live.update(criar_painel(progresso, texto_metricas))

            retorno = process.wait()
            if erro_numerico:
                raise RuntimeError(
                    "O treinamento foi interrompido automaticamente porque a perda ficou inválida "
                    f"({erro_numerico}). Isso costuma indicar que o prompt foi truncado antes da resposta "
                    "quando --mask-prompt está ativo. Gere dados menores ou aumente --max-seq-length."
                )
            if retorno != 0:
                diagnostico = "\n".join(linha for linha in ultimas_linhas if linha)
                raise RuntimeError(
                    f"O treinamento MLX terminou com código {retorno}. "
                    f"Saída final:\n{diagnostico}"
                )
            progresso.update(id_tarefa, completed=args.iters)
            live.update(criar_painel(progresso, Text("Treinamento concluído; salvando artefatos...", style="green")))

        melhor_checkpoint = guardar_melhor_checkpoint(args.adapters_dir, historico_validacao)
        resumo_path = salvar_resumo(args, comando, historico_validacao, melhor_checkpoint)
        if args.adapters_dir.exists():
            mlflow.log_artifacts(str(args.adapters_dir), artifact_path="adapters")

    print("\n[✓] Treinamento concluído com sucesso!")

if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Executa o ajuste fino LoRA do Qwen com painel de progresso.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--adapters-dir", type=Path, default=ADAPTERS_DIR)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_REPO") or "mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--iters", type=int, default=ITERS_TOTAIS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=25)
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
        help="Endereço do servidor ou banco do MLflow (padrão: SQLite local).",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or "nexoverdict-lora-fine-tuning",
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
