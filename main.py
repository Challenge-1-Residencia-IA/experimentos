"""Menu de terminal da tutora RAG."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
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
EXAMPLE_CASES = ROOT / "docs" / "casos_avaliacao_exemplo.jsonl"
REVIEWED_CASES = ROOT / "data" / "avaliacao" / "casos.jsonl"
console = Console()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def default_config() -> Path:
    custom = ROOT / "sources.json"
    return custom if custom.exists() else ROOT / "sources.example.json"


def default_cases() -> Path:
    """Prefere casos revisados locais e usa o exemplo versionado como fallback."""
    return REVIEWED_CASES if REVIEWED_CASES.exists() else EXAMPLE_CASES


@contextmanager
def temporary_mlflow(path: Path):
    """Isola uma ação no MLflow local e restaura o ambiente ao terminar."""
    base = path.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    variables = {
        "MLFLOW_TRACKING_URI": f"sqlite:///{base / 'mlflow.db'}",
        "MLFLOW_LOCAL_ARTIFACT_ROOT": str(base / "artifacts"),
        "MLFLOW_EXPERIMENT_NAME": "NexoVerdict-Temporario",
    }
    previous = {name: os.environ.get(name) for name in variables}
    os.environ.update(variables)
    try:
        yield base
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def selected_mlflow_experiment(name: str):
    """Usa um experimento durante uma conversa e restaura o ambiente ao sair."""
    normalized = name.strip()
    if not normalized:
        raise ValueError("O nome do experimento não pode ficar vazio")
    variable = "MLFLOW_EXPERIMENT_NAME"
    previous = os.environ.get(variable)
    os.environ[variable] = normalized
    try:
        yield normalized
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


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
    from rag.tutor import EVALUATION_DESCRIPTIONS, VERDICT_TEXTS

    console.print("[bold]ANÁLISE[/bold]")
    console.print(Panel(Text(answer["afirmacao"]), title="Afirmação", border_style="cyan"))
    verdict = answer.get("veredito") or VERDICT_TEXTS[answer["avaliacao_evidencias"]]
    console.print(Panel(Text(verdict), title="Veredito", border_style="yellow"))
    if answer.get("fonte_enviada"):
        console.print(f"[dim]Publicação enviada: {answer['fonte_enviada']['url']}[/dim]")
    if answer.get("achados"):
        console.print(Panel(
            Text("\n\n".join(f"• {value}" for value in answer["achados"])),
            title="O que foi encontrado (trechos literais, não validados como conclusão)",
            border_style="cyan",
        ))
    for title, key, color in (("Evidências favoráveis", "evidencias_favoraveis", "green"),
                              ("Evidências contrárias", "evidencias_contrarias", "red"),
                              ("Pontos de atenção", "pontos_de_atencao", "yellow")):
        values = answer[key]
        console.print(Panel(Text("\n".join(f"• {value}" for value in values) if values else "Nenhuma encontrada."),
                            title=title, border_style=color))
    console.print(Panel(Text(answer["contexto"]), title="Contexto", border_style="blue"))
    console.print(Panel(Text(EVALUATION_DESCRIPTIONS[answer["avaliacao_evidencias"]]),
                        title="Avaliação das evidências", border_style="yellow"))
    console.print(Panel(Text(answer["limitacoes"]), title="Limitações", border_style="magenta"))
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
    found = answer.get("fontes_encontradas", answer.get("fontes_consultadas", []))
    consulted = [item for item in found
                 if item["url"] not in {source["url"] for source in answer["fontes"]}]
    if consulted:
        table = Table(title="Fontes encontradas, não usadas como evidência", show_lines=True)
        table.add_column("Relação")
        table.add_column("Origem")
        table.add_column("Título")
        table.add_column("URL", overflow="fold")
        for source in consulted:
            table.add_row({"direta": "Direta", "relacionada": "Relacionada",
                           "nao_relacionada": "Outra"}.get(source.get("relevancia"), "—"),
                          source["origem"], source["titulo"], source["url"])
        console.print(table)
    console.print(Text(answer["limite_da_validacao"], style="dim"))
    research = answer.get("pesquisa_web")
    if research is not None:
        console.print(f"[dim]Pesquisa web: {research['resultados']} resultados; "
                      f"{research['paginas_lidas']} páginas lidas; "
                      f"{research.get('fontes_qualificadas', 0)} fontes qualificadas; "
                      f"{research.get('evidencias_locais', 0)} evidências locais.[/dim]")
        if research.get("erro_busca"):
            console.print(f"[yellow]Busca indisponível: {research['erro_busca']}[/yellow]")
        if research.get("paginas_ignoradas"):
            console.print(f"[dim]{len(research['paginas_ignoradas'])} páginas não puderam ser lidas.[/dim]")


def talk() -> None:
    load_dotenv(ROOT / ".env")
    default_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "NexoVerdict-RAG")
    experiment_name = Prompt.ask(
        "Nome do experimento no MLflow",
        default=default_experiment,
    ).strip()
    with selected_mlflow_experiment(experiment_name):
        console.print(f"[dim]Experimento MLflow: {experiment_name}[/dim]")
        _talk()


def _talk() -> None:
    from rag.auto_refresh import refresh_if_due
    from rag.store import VectorStore
    from rag.tutor import ask_online
    from rag.web import WebResearcher

    store = VectorStore()
    try:
        with console.status("Atualizando fontes e banco vetorial, se necessário...", spinner="dots"):
            refresh = refresh_if_due(store, default_config())
        if refresh is not None:
            collected = refresh["collected"]
            console.print(
                f"[green]Base atualizada:[/green] {collected['new']} novos, "
                f"{collected['updated']} alterados, {collected['failed']} falhas."
            )
            if collected["failed"]:
                console.print("[yellow]Algumas fontes falharam; a próxima conversa tentará novamente.[/yellow]")
    except Exception as error:
        console.print(f"[yellow]Atualização automática indisponível: {error}. "
                      "A conversa continuará com as fontes disponíveis.[/yellow]")
    researcher = WebResearcher()
    history: list[dict] = []
    console.print("[dim]Envie uma pergunta, alegação ou link. Digite 'sair' para voltar ao menu.[/dim]")
    while True:
        question = Prompt.ask("Você").strip()
        if question.casefold() in {"sair", "voltar", "exit"}:
            return
        if not question:
            continue
        try:
            with console.status("Pesquisando na web e lendo fontes...", spinner="dots"):
                answer = ask_online(question, researcher, store=store, history=history)
        except Exception as error:
            console.print(f"[red]Não foi possível investigar:[/red] {type(error).__name__}: {error}")
            continue
        render_answer(answer)
        summary = " ".join(part for part in (
            answer.get("contexto", ""), answer.get("limitacoes", "")
        ) if part)
        history.append({"pergunta": answer.get("alegacao_verificada", question),
                        "resposta": summary[:500]})


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
    default = default_cases()
    path = resolve_path(Prompt.ask("Arquivo de casos revisados", default=str(default.relative_to(ROOT))))
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
    from rag.dagshub_tracking import dagshub_tracking_uri, tracking_uri

    table.add_row("MLflow padrão", tracking_uri())
    table.add_row("Publicação DagsHub", dagshub_tracking_uri())
    table.add_row("Modelo Ollama", os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
    console.print(table)


def evaluate_mlflow() -> None:
    default = default_cases()
    path = resolve_path(Prompt.ask("Arquivo de casos", default=str(default.relative_to(ROOT))))
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
    console.print(f"[dim]MLflow: {tracking_uri}[/dim]")


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
    from rag.dagshub_tracking import tracking_uri

    port = IntPrompt.ask("Porta da interface", default=5000)
    uri = tracking_uri()
    command = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri",
               uri, "--port", str(port)]
    console.print(Panel(
        f"Acesse http://127.0.0.1:{port}\nPressione Ctrl+C para voltar.",
        title="MLflow local",
        border_style="green",
    ))
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Servidor MLflow encerrado.[/yellow]")
    else:
        if result.returncode:
            console.print(f"[red]MLflow terminou com código {result.returncode}.[/red]")


def show_dagshub() -> None:
    from rag.dagshub_tracking import dagshub_tracking_uri

    console.print(Panel(f"Painel da equipe: {dagshub_tracking_uri()}", title="DagsHub", border_style="green"))


def temporary_test() -> None:
    """Executa uma pergunta rastreada somente em um diretório local descartável."""
    location = resolve_path(Prompt.ask(
        "Diretório do MLflow temporário", default="/tmp/nexoverdict-mlflow-teste"
    ))
    mode = Prompt.ask("Fontes do teste", choices=["web", "local"], default="local")
    question = Prompt.ask("Pergunta ou alegação").strip()
    if not question:
        raise ValueError("Digite uma pergunta ou alegação")

    from rag.store import VectorStore
    from rag.tutor import ask, ask_online
    from rag.web import WebResearcher

    with temporary_mlflow(location) as base:
        store = VectorStore()
        with console.status("Executando teste temporário...", spinner="dots"):
            if mode == "web":
                answer = ask_online(question, WebResearcher(), store=store)
            else:
                answer = ask(question, store=store,
                             tracking_context={"modo": "local_temporario"})
        render_answer(answer)
        console.print(Panel(
            f"Banco: {base / 'mlflow.db'}\nArtefatos: {base / 'artifacts'}\n"
            "Esta execução não foi enviada ao DagsHub.",
            title="MLflow temporário",
            border_style="green",
        ))


def test_registered_model() -> None:
    """Baixa e executa uma versão do RAG registrada no DagsHub."""
    import mlflow
    import pandas as pd
    from mlflow.tracking import MlflowClient

    from rag.dagshub_tracking import (
        REGISTERED_MODEL_NAME,
        configure_tracking,
        tracked_registered_prediction,
    )

    configure_tracking(remote=True)
    versions = MlflowClient().search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise ValueError("Nenhuma versão do RAG foi encontrada no Model Registry")
    latest = str(max(int(item.version) for item in versions))
    version = Prompt.ask("Versão registrada", default=latest)
    mode = Prompt.ask("Fontes do teste", choices=["web", "local"], default="web")
    question = Prompt.ask("Pergunta ou alegação").strip()
    if not question:
        raise ValueError("Digite uma pergunta ou alegação")

    uri = f"models:/{REGISTERED_MODEL_NAME}/{version}"
    with console.status(f"Carregando {REGISTERED_MODEL_NAME} v{version}...", spinner="dots"):
        model = mlflow.pyfunc.load_model(uri)
        output, run_id = tracked_registered_prediction(
            question,
            mode,
            version,
            lambda: model.predict(pd.DataFrame({"pergunta": [question], "modo": [mode]}))[0],
        )
    render_answer(json.loads(output))
    console.print(f"[green]Modelo testado:[/green] {REGISTERED_MODEL_NAME} v{version}")
    console.print(f"[green]Execução MLflow:[/green] {run_id}")


def publish_dagshub() -> None:
    """Publica modelo, datasets ou ambos no DagsHub configurado."""
    target = Prompt.ask(
        "O que publicar?", choices=["modelo", "datasets", "ambos"], default="modelo"
    )
    if target in {"modelo", "ambos"}:
        from scripts.registrar_modelo_rag_mlflow import main as register_model

        with console.status("Empacotando e registrando o RAG...", spinner="dots"):
            register_model()
    if target in {"datasets", "ambos"}:
        from scripts.publicar_dataset_rag_mlflow import main as publish_datasets

        with console.status("Publicando datasets e manifesto...", spinner="dots"):
            publish_datasets()
    console.print("[green]Publicação concluída no DagsHub.[/green]")


def integrate_datasets() -> None:
    dataset = Prompt.ask(
        "Qual dataset integrar?",
        choices=["todos", "fakerecogna", "xfact", "averitec", "fakebr"],
        default="todos",
    )
    limit = IntPrompt.ask("Limite de registros por fonte", default=40)
    max_words = IntPrompt.ask("Máximo de palavras por chunk", default=180)
    overlap = IntPrompt.ask("Sobreposição (overlap) de palavras", default=35)
    inject = Prompt.ask("Injetar cabeçalho de título e fonte nos chunks? (s/n)", default="s").lower() == "s"

    command = [
        sys.executable,
        str(ROOT / "scripts" / "integrar_datasets.py"),
        "--dataset",
        dataset,
        "--limit",
        str(limit),
        "--max-words",
        str(max_words),
        "--overlap",
        str(overlap),
    ]
    if inject:
        command.append("--inject-context")

    console.print(Panel(" ".join(command), title="Integrar Datasets e Fragmentar", border_style="cyan"))
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        console.print(f"[red]Processamento encerrou com código {result.returncode}.[/red]")
    else:
        console.print("[green]Integração concluída! Use a opção 2 para reindexar o banco vetorial.[/green]")


ACTIONS: dict[str, tuple[str, str, Callable[[], None]]] = {
    "1": ("Coletar documentos", "Baixa e normaliza as fontes configuradas", collect),
    "2": ("Atualizar banco vetorial", "Gera embeddings e indexa os documentos", index),
    "3": ("Conversar com a tutora", "Pesquisa na web, lê links e explica a checagem", talk),
    "4": ("Descobrir checagens", "Pesquisa URLs candidatas para revisão", discover),
    "5": ("Avaliar recuperação", "Mede hit e MRR com URLs esperadas", evaluate),
    "6": ("Ver estado da base", "Mostra corpus e índice locais", status),
    "7": ("Avaliar respostas", "Registra análises e métricas no MLflow", evaluate_mlflow),
    "8": ("Benchmark AVeriTeC", "Baixa, prepara, indexa e avalia casos PT-BR", benchmark),
    "9": ("Abrir MLflow local", "Mostra as execuções registradas neste computador", open_mlflow),
    "10": ("Integrar novos datasets", "Baixa FakeRecogna, X-FACT, Fake.br e fragmenta textos", integrate_datasets),
    "11": ("Teste temporário", "Rastreia uma pergunta somente no MLflow local", temporary_test),
    "12": ("Testar versão registrada", "Baixa e executa uma versão do Model Registry", test_registered_model),
    "13": ("Publicar no DagsHub", "Registra o RAG e/ou os datasets da versão atual", publish_dagshub),
    "14": ("Ver DagsHub", "Mostra o painel de versões compartilhadas", show_dagshub),
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
