"""Menu de terminal da tutora RAG."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "documents.jsonl"
INDEX = ROOT / "vector_store" / "chroma"
CANDIDATES = ROOT / "data" / "candidatas_checagens.jsonl"
console = Console()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def default_config() -> Path:
    custom = ROOT / "sources.json"
    return custom if custom.exists() else ROOT / "sources.example.json"


def corpus_counts() -> dict[str, int]:
    counts = {"educativo": 0, "evidencia": 0}
    if CORPUS.exists():
        with CORPUS.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    kind = json.loads(line).get("kind")
                    if kind in counts:
                        counts[kind] += 1
    return counts


def menu() -> None:
    counts = corpus_counts()
    title = Text("NexoVerdict  •  Tutora RAG", style="bold cyan")
    description = (
        "Aprenda a investigar informações com método e fontes.\n"
        f"Corpus: {counts['educativo']} guias · {counts['evidencia']} evidências"
    )
    console.print(Panel(description, title=title, border_style="cyan", padding=(1, 2)))
    table = Table(title="Menu principal", border_style="cyan", show_lines=True)
    table.add_column("Opção", style="bold cyan", width=5)
    table.add_column("Ação")
    table.add_column("O que faz")
    for number, (label, description, _) in ACTIONS.items():
        table.add_row(number, label, description)
    table.add_row("0", "Sair", "Encerra o menu")
    console.print(table)


def collect() -> None:
    default = default_config().relative_to(ROOT)
    chosen = Prompt.ask("Arquivo de fontes", default=str(default))
    path = resolve_path(chosen)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de fontes não encontrado: {path}")
    from rag.corpus import ingest

    with console.status("Coletando e normalizando documentos...", spinner="dots"):
        counts = ingest(path)
    table = Table(title="Coleta concluída")
    table.add_column("Novos", justify="right")
    table.add_column("Atualizados", justify="right")
    table.add_column("Sem mudança", justify="right")
    table.add_column("Falhas", justify="right")
    table.add_row(*(str(counts[key]) for key in ("new", "updated", "unchanged", "failed")))
    console.print(table)
    console.print("[dim]Use a opção 2 para atualizar o banco vetorial.[/dim]")


def index() -> None:
    if not CORPUS.exists():
        console.print("[yellow]O corpus ainda não existe. Use a opção 1 primeiro.[/yellow]")
        return
    from rag.store import VectorStore

    with console.status("Gerando embeddings e atualizando o índice...", spinner="dots"):
        counts = VectorStore().index()
    console.print(
        f"[green]Índice atualizado.[/green] {counts['indexed']} documentos indexados, "
        f"{counts['unchanged']} sem mudança, {counts['removed_chunks']} trechos antigos removidos."
    )


def render_answer(answer: dict) -> None:
    verdict = answer["veredito"]
    style = {
        "VERDADEIRA": "green",
        "FALSA": "red",
        "ENGANOSA": "yellow",
        "INCONCLUSIVA": "magenta",
    }.get(verdict, "white")
    console.print(Panel(Text(answer["como_verificar"]), title="Como verificar", border_style="cyan"))
    console.print(Panel(Text(answer["analise"]), title="Análise das evidências", border_style="blue"))
    console.print(Panel(Text(answer["justificativa"]), title=f"Veredito: {verdict}", border_style=style))
    if answer["fontes"]:
        table = Table(title="Fontes citadas", show_lines=True)
        table.add_column("Nº", justify="right", style="cyan", no_wrap=True)
        table.add_column("Tipo", no_wrap=True)
        table.add_column("Título / origem")
        table.add_column("Data", no_wrap=True)
        table.add_column("URL ou arquivo", overflow="fold")
        for source in answer["fontes"]:
            table.add_row(
                str(source["numero"]),
                source["tipo"],
                Text(f"{source['titulo']}\n{source['origem']}"),
                source.get("data") or "—",
                Text(source["url"]),
            )
        console.print(table)
    console.print(Text(answer["limite_da_validacao"], style="dim"))


def talk() -> None:
    if not INDEX.exists():
        console.print("[yellow]O banco vetorial ainda não existe. Use as opções 1 e 2 primeiro.[/yellow]")
        return
    question = Prompt.ask("Pergunta ou alegação").strip()
    if not question:
        console.print("[yellow]Digite uma pergunta para continuar.[/yellow]")
        return
    from rag.tutor import ask

    with console.status("Buscando fontes e preparando a explicação...", spinner="dots"):
        answer = ask(question)
    render_answer(answer)


def discover() -> None:
    query = Prompt.ask("Tema ou alegação para pesquisar").strip()
    if not query:
        console.print("[yellow]Digite um tema para continuar.[/yellow]")
        return
    pages = IntPrompt.ask("Páginas de resultados (1 a 10)", default=1)
    from scripts.descobrir_checagens import discover_claims

    with console.status("Consultando checagens...", spinner="dots"):
        added = discover_claims(query, pages, CANDIDATES)
    console.print(Text(f"{added} URLs candidatas novas. Arquivo: {CANDIDATES}", style="green"))
    console.print("[dim]Revise fonte e licença antes de incluir uma URL no corpus.[/dim]")


def evaluate() -> None:
    default = "data/avaliacao/casos.jsonl"
    path = resolve_path(Prompt.ask("Arquivo de casos revisados", default=default))
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de avaliação não encontrado: {path}")
    k = IntPrompt.ask("Quantos resultados examinar", default=5)
    from scripts.avaliar_recuperacao_rag import evaluate_retrieval

    with console.status("Avaliando recuperação...", spinner="dots"):
        details, summary = evaluate_retrieval(path, k)
    table = Table(title="Recuperação")
    table.add_column("Caso", justify="right")
    table.add_column("Posição da fonte esperada", justify="right")
    for item in details:
        table.add_row(str(item["caso"]), str(item["rank"]) if item["rank"] else "não encontrada")
    console.print(table)
    console.print(
        f"[bold]Casos:[/bold] {summary['casos']}   "
        f"[bold]Hit@{k}:[/bold] {summary[f'hit@{k}']:.1%}   "
        f"[bold]MRR:[/bold] {summary['mrr']:.3f}"
    )


def status() -> None:
    counts = corpus_counts()
    table = Table(title="Estado local")
    table.add_column("Item")
    table.add_column("Estado", style="cyan")
    table.add_row("Configuração", str(default_config()))
    table.add_row("Guias educativos", str(counts["educativo"]))
    table.add_row("Documentos de evidência", str(counts["evidencia"]))
    table.add_row("Banco vetorial", "presente" if INDEX.exists() else "ainda não criado")
    table.add_row("Checagens candidatas", "presente" if CANDIDATES.exists() else "nenhuma")
    benchmark_cases = ROOT / "data" / "benchmark_averitec" / "casos.jsonl"
    table.add_row("Benchmark AVeriTeC", "preparado" if benchmark_cases.exists() else "ainda não preparado")
    console.print(table)


def evaluate_mlflow() -> None:
    default = "data/avaliacao/casos.jsonl"
    path = resolve_path(Prompt.ask("Arquivo de casos", default=default))
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de avaliação não encontrado: {path}")
    from scripts.avaliar_rag_mlflow import evaluate as evaluate_full

    metrics, run_id, tracking_uri = evaluate_full(path)
    table = Table(title="Avaliação da tutora")
    table.add_column("Métrica")
    table.add_column("Valor", justify="right", style="cyan")
    for name, value in metrics.items():
        table.add_row(name, f"{value:.3f}")
    console.print(table)
    console.print(f"[green]Execução MLflow:[/green] {run_id}")
    console.print(f"[dim]Painel: {sys.executable} -m mlflow ui --backend-store-uri {tracking_uri}[/dim]")


def benchmark() -> None:
    per_class = IntPrompt.ask("Casos por veredito", default=10)
    seed = IntPrompt.ask("Semente da amostra", default=42)
    command = [sys.executable, "-m", "scripts.testar_benchmark_averitec",
               "--per-class", str(per_class), "--seed", str(seed)]
    console.print(Panel(" ".join(command), title="Benchmark AVeriTeC", border_style="blue"))
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        console.print(f"[red]Benchmark terminou com código {result.returncode}.[/red]")
    else:
        console.print("[green]Benchmark concluído. Abra a opção 9 para ver as métricas.[/green]")


def open_mlflow() -> None:
    load_dotenv(ROOT / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"
    port = IntPrompt.ask("Porta da interface", default=5000)
    command = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri",
               tracking_uri, "--port", str(port)]
    console.print(Panel(f"Acesse http://127.0.0.1:{port}\nPressione Ctrl+C para voltar.",
                        title="MLflow", border_style="green"))
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Servidor MLflow encerrado.[/yellow]")
    else:
        if result.returncode:
            console.print(f"[red]MLflow terminou com código {result.returncode}.[/red]")


ACTIONS: dict[str, tuple[str, str, Callable[[], None]]] = {
    "1": ("Coletar documentos", "Baixa e normaliza as fontes configuradas", collect),
    "2": ("Atualizar banco vetorial", "Gera embeddings e indexa os documentos", index),
    "3": ("Conversar com a tutora", "Busca evidências e explica a checagem", talk),
    "4": ("Descobrir checagens", "Pesquisa URLs candidatas para revisão", discover),
    "5": ("Avaliar recuperação", "Mede hit e MRR com URLs esperadas", evaluate),
    "6": ("Ver estado da base", "Mostra corpus e índice locais", status),
    "7": ("Avaliar respostas", "Registra vereditos e métricas no MLflow", evaluate_mlflow),
    "8": ("Benchmark AVeriTeC", "Baixa, prepara, indexa e avalia casos PT-BR", benchmark),
    "9": ("Abrir MLflow", "Inicia o painel com o Python deste ambiente", open_mlflow),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="Exibe detalhes técnicos em caso de erro")
    parser.add_argument("--listar", action="store_true", help="Mostra o menu e encerra")
    parser.add_argument("--opcao", choices=[*ACTIONS, "0"], help="Abre diretamente uma opção")
    args = parser.parse_args()
    if args.listar:
        menu()
        return
    first_choice = args.opcao
    try:
        while True:
            if console.is_terminal:
                console.clear()
            menu()
            choice = first_choice or Prompt.ask("Escolha uma opção", choices=[*ACTIONS, "0"])
            first_choice = None
            if choice == "0":
                console.print("[cyan]Até a próxima checagem.[/cyan]")
                break
            console.print()
            try:
                ACTIONS[choice][2]()
            except ModuleNotFoundError as error:
                console.print(f"[red]Dependência ausente:[/red] {error.name}. Instale requirements-rag.txt.")
            except requests.ConnectionError:
                console.print("[red]Falha de conexão.[/red] Confira a rede ou se o Ollama está em execução.")
            except requests.RequestException:
                console.print("[red]Falha na requisição remota.[/red] Confira a rede e as configurações da fonte ou do Ollama.")
                if args.debug:
                    console.print_exception()
            except (FileNotFoundError, OSError, ValueError) as error:
                console.print(Text(f"Não foi possível concluir: {error}", style="red"))
            except Exception as error:
                console.print(Text(f"Erro inesperado: {type(error).__name__}: {error}", style="red"))
                if args.debug:
                    console.print_exception()
            console.input("\n[dim]Pressione Enter para voltar ao menu...[/dim]")
    except (KeyboardInterrupt, EOFError):
        console.print("\n[cyan]Sessão encerrada.[/cyan]")


if __name__ == "__main__":
    main()
