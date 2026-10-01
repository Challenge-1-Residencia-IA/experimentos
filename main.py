#!/usr/bin/env python3
"""Central interativa para treino, avaliação e uso dos modelos do projeto."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table


RAIZ = Path(__file__).resolve().parent
SCRIPTS = RAIZ / "scripts"
load_dotenv(RAIZ / ".env")
MODELO_PADRAO = os.environ.get("HF_MODEL_REPO") or "mlx-community/Qwen2.5-7B-Instruct-4bit"
TRACKING_PADRAO = f"sqlite:///{(RAIZ / 'mlflow.db').resolve()}"
console = Console()


def caminho_relativo(caminho: Path) -> str:
    try:
        return str(caminho.resolve().relative_to(RAIZ))
    except ValueError:
        return str(caminho)


def perguntar_caminho(mensagem: str, padrao: Path) -> Path:
    resposta = Prompt.ask(mensagem, default=caminho_relativo(padrao)).strip()
    caminho = Path(resposta).expanduser()
    return caminho if caminho.is_absolute() else RAIZ / caminho


def perguntar_opcional(mensagem: str) -> str | None:
    resposta = Prompt.ask(mensagem, default="").strip()
    return resposta or None


def configuracao_mlflow(experimento_padrao: str = "nexoverdict-lora-fine-tuning") -> tuple[str, str]:
    experimento = Prompt.ask(
        "Nome do experimento no MLflow",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or experimento_padrao,
    )
    endereco = Prompt.ask(
        "Banco ou servidor do MLflow",
        default=os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_PADRAO,
    )
    return experimento, endereco


def adaptadores_disponiveis() -> list[Path]:
    modelos = RAIZ / "models"
    if not modelos.exists():
        return []
    return sorted({arquivo.parent for arquivo in modelos.rglob("adapters.safetensors")})


def selecionar_adaptador(padrao: Path | None = None) -> Path:
    adaptadores = adaptadores_disponiveis()
    if adaptadores:
        tabela = Table(title="Adaptadores encontrados", border_style="cyan")
        tabela.add_column("#", justify="right")
        tabela.add_column("Diretório")
        for indice, caminho in enumerate(adaptadores, 1):
            tabela.add_row(str(indice), caminho_relativo(caminho))
        console.print(tabela)
        escolha = Prompt.ask(
            "Número do adaptador ou caminho personalizado",
            default=str(len(adaptadores)),
        ).strip()
        if escolha.isdigit() and 1 <= int(escolha) <= len(adaptadores):
            return adaptadores[int(escolha) - 1]
        caminho = Path(escolha).expanduser()
        return caminho if caminho.is_absolute() else RAIZ / caminho
    return perguntar_caminho("Diretório do adaptador", padrao or RAIZ / "models/adapters-evidencias-v3/best")


def proximo_adaptador_evidencias() -> Path:
    numero = 1
    while (RAIZ / f"models/adapters-evidencias-v{numero}").exists():
        numero += 1
    return RAIZ / f"models/adapters-evidencias-v{numero}"


def executar_script(nome: str, argumentos: list[str], *, confirmar: bool = True) -> None:
    comando = [sys.executable, str(SCRIPTS / nome), *argumentos]
    console.print(Panel(" ".join(comando), title="Comando que será executado", border_style="blue"))
    if confirmar and not Confirm.ask("Executar agora?", default=True):
        console.print("[yellow]Operação cancelada.[/yellow]")
        return
    try:
        subprocess.run(comando, cwd=RAIZ, check=True)
    except subprocess.CalledProcessError as erro:
        console.print(Panel(f"O processo terminou com código {erro.returncode}.", title="Falha", border_style="red"))
    except KeyboardInterrupt:
        console.print("\n[yellow]Operação interrompida pelo usuário.[/yellow]")
    else:
        console.print("[bold green]✓ Operação concluída.[/bold green]")


def verificar_link() -> None:
    url = Prompt.ask("URL pública da notícia").strip()
    adaptador = selecionar_adaptador()
    saida_padrao = RAIZ / f"results/verificacao_link_{datetime.now():%Y%m%d_%H%M%S}.json"
    saida = perguntar_caminho("Arquivo JSON de saída", saida_padrao)
    experimento, tracking = configuracao_mlflow("verificacao-de-links")
    argumentos = [
        url, "--adapters-dir", str(adaptador), "--model", Prompt.ask("Modelo-base", default=MODELO_PADRAO),
        "--max-resultados", str(IntPrompt.ask("Máximo de fontes", default=5)),
        "--max-chars", str(IntPrompt.ask("Máximo de caracteres extraídos da página", default=12000)),
        "--max-claim-chars", str(IntPrompt.ask("Máximo de caracteres da alegação", default=500)),
        "--min-fontes-independentes", str(IntPrompt.ask("Mínimo de fontes independentes", default=2)),
        "--max-tokens", str(IntPrompt.ask("Máximo de tokens da resposta", default=220)),
        "--timeout", str(IntPrompt.ask("Tempo limite da página, em segundos", default=30)),
        "--output", str(saida), "--experiment-name", experimento, "--tracking-uri", tracking, "--visual",
    ]
    executar_script("verificar_link.py", argumentos)


def responder_pergunta() -> None:
    pergunta = Prompt.ask("Digite sua pergunta").strip()
    modo = Prompt.ask(
        "Modo de resposta",
        choices=["evidencias", "conversa"],
        default="evidencias",
    )
    argumentos = [
        pergunta,
        "--modo", modo,
        "--model", Prompt.ask("Modelo-base", default=MODELO_PADRAO),
        "--max-tokens", str(IntPrompt.ask("Máximo de tokens da resposta", default=350)),
        "--max-tentativas", str(IntPrompt.ask("Tentativas para corrigir formato/citações", default=2)),
        "--visual",
    ]
    if modo == "evidencias":
        argumentos.extend([
            "--max-resultados", str(IntPrompt.ask("Máximo de fontes", default=5)),
            "--min-fontes", str(IntPrompt.ask("Mínimo de fontes para responder", default=2)),
        ])
    if Confirm.ask("Usar um adapter de perguntas? Adapters de evidências/Fake.br serão recusados.", default=False):
        argumentos.extend(["--adapters-dir", str(selecionar_adaptador())])
    if Confirm.ask("Usar outra pasta de prompts internos?", default=False):
        argumentos.extend(["--prompts-dir", str(perguntar_caminho("Pasta dos prompts", RAIZ / "prompts"))])
    experimento, tracking = configuracao_mlflow("nexoverdict-perguntas")
    argumentos.extend(["--experiment-name", experimento, "--tracking-uri", tracking])
    if Confirm.ask("Salvar também um relatório JSON?", default=True):
        saida = RAIZ / f"results/pergunta_{datetime.now():%Y%m%d_%H%M%S}.json"
        argumentos.extend(["--output", str(perguntar_caminho("Arquivo JSON de saída", saida))])
    executar_script("responder_pergunta.py", argumentos)


def preparar_perguntas() -> None:
    dados = perguntar_caminho("Dataset de evidências preparado", RAIZ / "data/evidencias-por-url-v1")
    destino = perguntar_caminho("Novo dataset de perguntas", RAIZ / "data/perguntas-v1")
    argumentos = ["--evidence-data-dir", str(dados), "--output-dir", str(destino)]
    revisados = perguntar_opcional("JSONL de perguntas revisadas? Vazio = somente leitura de fontes")
    if revisados:
        argumentos.extend(["--curated-jsonl", revisados])
    if Confirm.ask("Usar outra pasta de prompts internos?", default=False):
        argumentos.extend(["--prompts-dir", str(perguntar_caminho("Pasta dos prompts", RAIZ / "prompts"))])
    if Confirm.ask("Substituir arquivos de dataset existentes no destino?", default=False):
        argumentos.append("--overwrite")
    executar_script("preparar_dataset_perguntas.py", argumentos)
    console.print("[dim]Para ajustar um adapter com estes dados, use a opção 3 e selecione o dataset de perguntas.[/dim]")


def baixar_perguntas_publicas() -> None:
    destino = perguntar_caminho("Pasta isolada para os dados públicos", RAIZ / "raw/publicos-perguntas-v1")
    argumentos = [
        "--output-dir", str(destino),
        "--rag-limit", str(IntPrompt.ask("Quantidade máxima do RAG PT-BR sintético", default=5000)),
    ]
    if destino.exists() and any(destino.iterdir()):
        argumentos.append("--resume")
    executar_script("baixar_datasets_perguntas.py", argumentos)


def mesclar_perguntas_publicas() -> None:
    argumentos = [
        "--base-data-dir", str(perguntar_caminho("Dataset de perguntas já preparado", RAIZ / "data/perguntas-v2")),
        "--public-raw-dir", str(perguntar_caminho("Dados públicos baixados", RAIZ / "raw/publicos-perguntas-v1")),
        "--output-dir", str(perguntar_caminho("Novo dataset mesclado", RAIZ / "data/perguntas-mescladas-v1")),
        "--seed", str(IntPrompt.ask("Semente da divisão", default=42)),
        "--max-por-origem", str(IntPrompt.ask("Máximo de exemplos por dataset público (0 = sem limite)", default=2500)),
        "--fracao-inconclusiva", str(FloatPrompt.ask("Fração de casos inconclusivos com distratores", default=0.12)),
    ]
    executar_script("preparar_dataset_perguntas_publicas.py", argumentos)


def avaliar_perguntas() -> None:
    teste = perguntar_caminho("Conjunto de teste de perguntas", RAIZ / "data/perguntas-mescladas-v1/test.jsonl")
    adaptador = selecionar_adaptador(RAIZ / "models/adapters-perguntas-v2/best")
    experimento, tracking = configuracao_mlflow("nexoverdict-perguntas-avaliacao")
    argumentos = [
        "--test-jsonl", str(teste), "--adapters-dir", str(adaptador),
        "--model", Prompt.ask("Modelo-base", default=MODELO_PADRAO),
        "--max-tokens", str(IntPrompt.ask("Máximo de tokens por resposta", default=350)),
        "--output", str(perguntar_caminho("Relatório JSON", RAIZ / "results/avaliacao_perguntas.json")),
        "--experiment-name", experimento, "--tracking-uri", tracking,
    ]
    limite = perguntar_opcional("Limitar exemplos? Vazio = avaliar todos")
    if limite:
        argumentos.extend(["--limit", limite])
    executar_script("avaliar_perguntas.py", argumentos)


def avaliar_evidencias() -> None:
    teste = perguntar_caminho("Conjunto de teste JSONL", RAIZ / "data/evidencias-por-url-v1/test.jsonl")
    adaptador = selecionar_adaptador()
    experimento, tracking = configuracao_mlflow()
    argumentos = [
        "--test-jsonl", str(teste), "--adapters-dir", str(adaptador),
        "--model", Prompt.ask("Modelo-base", default=MODELO_PADRAO),
        "--results-dir", str(perguntar_caminho("Diretório dos resultados", RAIZ / "results")),
        "--max-tokens", str(IntPrompt.ask("Máximo de tokens por resposta", default=220)),
        "--experiment-name", experimento, "--tracking-uri", tracking, "--visual",
    ]
    limite = perguntar_opcional("Limitar a quantidade de exemplos? Vazio = avaliar todos")
    if limite:
        argumentos.extend(["--limit", limite])
    executar_script("avaliar_evidencias.py", argumentos)


def treinar_adaptador() -> None:
    dados = perguntar_caminho("Diretório do dataset preparado", RAIZ / "data/evidencias-por-url-v1")
    destino = perguntar_caminho("Novo diretório do adaptador", proximo_adaptador_evidencias())
    experimento, tracking = configuracao_mlflow()
    argumentos = [
        "--data-dir", str(dados), "--adapters-dir", str(destino),
        "--model", Prompt.ask("Modelo-base", default=MODELO_PADRAO),
        "--iters", str(IntPrompt.ask("Iterações", default=100)),
        "--batch-size", str(IntPrompt.ask("Tamanho do lote", default=1)),
        "--learning-rate", str(FloatPrompt.ask("Taxa de aprendizado", default=1e-5)),
        "--num-layers", str(IntPrompt.ask("Camadas LoRA", default=16)),
        "--max-seq-length", str(IntPrompt.ask("Comprimento máximo da sequência", default=1024)),
        "--grad-accumulation-steps", str(IntPrompt.ask("Passos de acumulação de gradiente", default=1)),
        "--val-batches", str(IntPrompt.ask("Lotes usados na validação (-1 = todos; pode ser muito lento)", default=25)),
        "--steps-per-eval", str(IntPrompt.ask("Validar a cada N iterações", default=25)),
        "--save-every", str(IntPrompt.ask("Salvar checkpoint a cada N iterações", default=25)),
        "--steps-per-report", str(IntPrompt.ask("Atualizar métricas a cada N iterações", default=1)),
        "--seed", str(IntPrompt.ask("Semente aleatória", default=42)),
        "--experiment-name", experimento, "--tracking-uri", tracking,
    ]
    retomada = perguntar_opcional("Checkpoint para continuar um treino? Vazio = começar novo")
    if retomada:
        argumentos.extend(["--resume-adapter-file", retomada])
    if not Confirm.ask("Mascarar o prompt durante o cálculo da perda?", default=True):
        argumentos.append("--no-mask-prompt")
    if not Confirm.ask("Usar checkpoint de gradiente para economizar memória?", default=True):
        argumentos.append("--no-grad-checkpoint")
    if Confirm.ask("Executar também o teste interno do MLX ao finalizar?", default=False):
        argumentos.append("--evaluate-test")
    if Confirm.ask("Pular a conferência/download antecipado do modelo?", default=False):
        argumentos.append("--skip-model-download")
    if Confirm.ask("Substituir um adaptador final caso já exista no destino?", default=False):
        argumentos.append("--overwrite")
    executar_script("treinar_visual.py", argumentos)


def preparar_evidencias() -> None:
    argumentos = [
        "--source", str(perguntar_caminho("Arquivo FACTCK.BR TSV", RAIZ / "raw/factckbr/FACTCKBR.tsv")),
        "--output-dir", str(perguntar_caminho("Diretório de saída", RAIZ / "data/evidencias-por-url-v1")),
        "--seed", str(IntPrompt.ask("Semente aleatória", default=42)),
        "--max-claim-chars", str(IntPrompt.ask("Máximo de caracteres da alegação", default=500)),
        "--max-evidence-chars", str(IntPrompt.ask("Máximo de caracteres da evidência", default=700)),
        "--min-evidence-chars", str(IntPrompt.ask("Mínimo de caracteres da evidência", default=80)),
        "--max-justification-chars", str(IntPrompt.ask("Máximo de caracteres da justificativa", default=320)),
        "--balance-strategy", Prompt.ask("Balanceamento", choices=["none", "oversample", "undersample"], default="oversample"),
        "--synthetic-inconclusive", str(IntPrompt.ask("Exemplos inconclusivos sintéticos", default=30)),
        "--balance-target", str(IntPrompt.ask("Alvo mínimo por classe", default=150)),
    ]
    executar_script("preparar_dataset_evidencias.py", argumentos)


def pipeline_evidencias() -> None:
    experimento, tracking = configuracao_mlflow()
    argumentos = [
        "--source", str(perguntar_caminho("Arquivo FACTCK.BR TSV", RAIZ / "raw/factckbr/FACTCKBR.tsv")),
        "--data-dir", str(perguntar_caminho("Diretório do dataset preparado", RAIZ / "data/evidencias-por-url-v1")),
        "--adapters-dir", str(perguntar_caminho("Novo diretório do adaptador", proximo_adaptador_evidencias())),
        "--results-dir", str(perguntar_caminho("Diretório dos resultados", RAIZ / "results")),
        "--model", Prompt.ask("Modelo-base", default=MODELO_PADRAO),
        "--iters", str(IntPrompt.ask("Iterações", default=100)),
        "--learning-rate", str(FloatPrompt.ask("Taxa de aprendizado", default=1e-5)),
        "--max-seq-length", str(IntPrompt.ask("Comprimento máximo da sequência", default=1024)),
        "--steps-per-eval", str(IntPrompt.ask("Validar a cada N iterações", default=25)),
        "--seed", str(IntPrompt.ask("Semente aleatória", default=42)),
        "--balance-strategy", Prompt.ask("Balanceamento", choices=["none", "oversample", "undersample"], default="oversample"),
        "--synthetic-inconclusive", str(IntPrompt.ask("Exemplos inconclusivos sintéticos", default=30)),
        "--balance-target", str(IntPrompt.ask("Alvo mínimo por classe", default=150)),
        "--experiment-name", experimento, "--tracking-uri", tracking, "--visual",
    ]
    retomada = perguntar_opcional("Checkpoint para continuar um treino? Vazio = começar novo")
    if retomada:
        argumentos.extend(["--resume-adapter-file", retomada])
    avaliar_teste = Confirm.ask("Avaliar o teste reservado ao terminar?", default=False)
    if avaliar_teste:
        argumentos.append("--evaluate-test")
        limite = perguntar_opcional("Limitar a avaliação? Vazio = avaliar todos")
        if limite:
            argumentos.extend(["--limit-test", limite])
    if Confirm.ask("Substituir um adaptador final já existente?", default=False):
        argumentos.append("--overwrite")
    executar_script("executar_pipeline_evidencias.py", argumentos)


def pipeline_fakebr() -> None:
    experimento, tracking = configuracao_mlflow("fakebr-classificacao")
    argumentos = [
        "--input-csv", str(perguntar_caminho("CSV pré-processado do Fake.br", RAIZ / "raw/preprocessed/pre-processed.csv")),
        "--data-dir", str(perguntar_caminho("Diretório do dataset preparado", RAIZ / "data/fakebr")),
        "--adapters-dir", str(perguntar_caminho("Diretório do novo adaptador", RAIZ / "models/adapters-fakebr-v1")),
        "--model", Prompt.ask("Modelo-base", default=MODELO_PADRAO),
        "--iters", str(IntPrompt.ask("Iterações", default=400)),
        "--test-size", str(FloatPrompt.ask("Proporção de teste", default=0.15)),
        "--valid-size", str(FloatPrompt.ask("Proporção de validação", default=0.15)),
        "--seed", str(IntPrompt.ask("Semente aleatória", default=42)),
        "--results-dir", str(perguntar_caminho("Diretório dos resultados", RAIZ / "results")),
        "--experiment-name", experimento, "--tracking-uri", tracking, "--visual",
    ]
    limite = perguntar_opcional("Limitar avaliação final? Vazio = todos")
    if limite:
        argumentos.extend(["--limit-test", limite])
    executar_script("executar_pipeline.py", argumentos)


def preparar_fakebr() -> None:
    argumentos = [
        "--input-csv", str(perguntar_caminho("CSV pré-processado do Fake.br", RAIZ / "raw/preprocessed/pre-processed.csv")),
        "--output-dir", str(perguntar_caminho("Diretório de saída", RAIZ / "data/fakebr")),
        "--test-size", str(FloatPrompt.ask("Proporção de teste", default=0.15)),
        "--valid-size", str(FloatPrompt.ask("Proporção de validação", default=0.15)),
        "--seed", str(IntPrompt.ask("Semente aleatória", default=42)),
    ]
    executar_script("preparar_fakebr.py", argumentos)


def avaliar_fakebr() -> None:
    experimento, tracking = configuracao_mlflow("fakebr-classificacao")
    argumentos = [
        "--test-csv", str(perguntar_caminho("CSV reservado para teste", RAIZ / "data/fakebr/test.csv")),
        "--adapters-dir", str(selecionar_adaptador(RAIZ / "models/adapters-fakebr-v1")),
        "--model", Prompt.ask("Modelo-base", default=MODELO_PADRAO),
        "--results-dir", str(perguntar_caminho("Diretório dos resultados", RAIZ / "results")),
        "--experiment-name", experimento, "--tracking-uri", tracking, "--visual",
    ]
    limite = perguntar_opcional("Limitar a quantidade de exemplos? Vazio = avaliar todos")
    if limite:
        argumentos.extend(["--limit", limite])
    executar_script("avaliar_fakebr.py", argumentos)


def classificar_noticia() -> None:
    adaptador = selecionar_adaptador(RAIZ / "models/adapters-fakebr-v1")
    argumentos = ["--adapters-dir", str(adaptador), "--model", Prompt.ask("Modelo-base", default=MODELO_PADRAO), "--visual"]
    if Confirm.ask("A notícia está em um arquivo?", default=False):
        argumentos.extend(["--arquivo", str(perguntar_caminho("Arquivo UTF-8", RAIZ / "noticia.txt"))])
    else:
        argumentos.extend(["--texto", Prompt.ask("Texto da notícia")])
    executar_script("classificar_noticia.py", argumentos)


def amostrar_com_ollama() -> None:
    argumentos = [
        "--dataset", str(perguntar_caminho("Dataset CSV", RAIZ / "raw/preprocessed/pre-processed.csv")),
        "--quantidade", str(IntPrompt.ask("Quantidade de notícias", default=5)),
        "--seed", str(IntPrompt.ask("Semente aleatória", default=42)),
        "--modelo", Prompt.ask("Modelo no Ollama", default=os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b"),
        "--max-resultados", str(IntPrompt.ask("Fontes por notícia", default=5)),
        "--max-chars", str(IntPrompt.ask("Máximo de caracteres por notícia", default=1200)),
        "--timeout", str(IntPrompt.ask("Tempo limite, em segundos", default=120)),
        "--output", str(perguntar_caminho("Arquivo JSON de saída", RAIZ / "results/verificacao_amostral.json")),
        "--visual",
    ]
    executar_script("amostrar_e_verificar.py", argumentos)


def abrir_mlflow() -> None:
    endereco = os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_PADRAO
    porta = IntPrompt.ask("Porta da interface", default=5000)
    comando = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", endereco, "--port", str(porta)]
    console.print(Panel(f"Acesse http://127.0.0.1:{porta}\nPressione Ctrl+C para encerrar.", title="MLflow", border_style="green"))
    try:
        subprocess.run(comando, cwd=RAIZ, check=True)
    except (KeyboardInterrupt, subprocess.CalledProcessError):
        console.print("[yellow]Servidor MLflow encerrado.[/yellow]")


ACOES: dict[str, tuple[str, str, Callable[[], None]]] = {
    "1": ("Verificar notícia por link", "Busca evidências, aplica o adapter e salva auditoria", verificar_link),
    "2": ("Avaliar adapter de evidências", "Métricas, classes, MLflow, CSV e matriz de confusão", avaliar_evidencias),
    "3": ("Treinar novo adapter", "Ajuste fino usando um dataset já preparado", treinar_adaptador),
    "4": ("Pipeline completo de evidências", "Prepara dados, treina e opcionalmente avalia", pipeline_evidencias),
    "5": ("Preparar dataset de evidências", "Limpa, agrupa, divide e balanceia o FACTCK.BR", preparar_evidencias),
    "6": ("Pipeline completo Fake.br", "Prepara, treina e avalia classificação binária", pipeline_fakebr),
    "7": ("Classificar uma notícia", "Classificação direta de texto ou arquivo", classificar_noticia),
    "8": ("Amostragem com Ollama", "Amostra notícias, busca fontes e verifica", amostrar_com_ollama),
    "9": ("Abrir MLflow", "Inicia a interface local de experimentos", abrir_mlflow),
    "10": ("Preparar somente o Fake.br", "Cria os conjuntos de treino, validação e teste", preparar_fakebr),
    "11": ("Avaliar somente o Fake.br", "Calcula métricas e registra a avaliação no MLflow", avaliar_fakebr),
    "12": ("Fazer uma pergunta", "Responde com busca e fontes ou em modo de conversa local", responder_pergunta),
    "13": ("Preparar perguntas", "Cria exemplos de resposta e justificativa para ajuste fino", preparar_perguntas),
    "14": ("Baixar datasets públicos", "Baixa QA português em uma pasta raw isolada; não treina", baixar_perguntas_publicas),
    "15": ("Mesclar datasets de perguntas", "Normaliza e mescla em um novo dataset; não treina", mesclar_perguntas_publicas),
    "16": ("Avaliar adapter de perguntas", "Mede JSON, citações, status, resposta e abstenção", avaliar_perguntas),
}


def mostrar_menu() -> None:
    console.print(Panel("Treino, avaliação e verificação em um só lugar", title="NexoVerdict IA", border_style="bright_blue"))
    tabela = Table(title="Menu principal", border_style="bright_blue", show_lines=True)
    tabela.add_column("Opção", justify="center", style="bold cyan")
    tabela.add_column("O que fazer", style="bold")
    tabela.add_column("O que a opção executa")
    for codigo, (titulo, descricao, _) in ACOES.items():
        tabela.add_row(codigo, titulo, descricao)
    tabela.add_row("0", "Sair", "Encerra o menu")
    console.print(tabela)


def executar_menu(opcao_inicial: str | None = None) -> None:
    opcao = opcao_inicial
    while True:
        mostrar_menu()
        escolha = opcao or Prompt.ask("Escolha uma opção", default="1")
        opcao = None
        if escolha == "0":
            console.print("Até a próxima.")
            return
        acao = ACOES.get(escolha)
        if acao is None:
            console.print("[red]Opção inválida.[/red]")
            continue
        console.print(f"\n[bold cyan]{acao[0]}[/bold cyan]")
        try:
            acao[2]()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Operação cancelada.[/yellow]")
        if not Confirm.ask("Voltar ao menu principal?", default=True):
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Menu unificado do projeto NexoVerdict IA.")
    parser.add_argument("--listar", action="store_true", help="Mostra o menu e encerra sem solicitar entrada.")
    parser.add_argument("--opcao", choices=[*ACOES.keys(), "0"], help="Abre diretamente uma opção do menu.")
    argumentos = parser.parse_args()
    if argumentos.listar:
        mostrar_menu()
        return
    executar_menu(argumentos.opcao)


if __name__ == "__main__":
    main()
