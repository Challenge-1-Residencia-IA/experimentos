#!/usr/bin/env python3
"""Executa inferência no teste do Fake.br e registra métricas no MLflow."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from mlx_lm import generate, load
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from preparar_fakebr import PROMPT_SISTEMA
from ui_terminal import cabecalho, console, tabela_metricas


ROOT_DIR = Path(__file__).resolve().parents[1]
ROTULOS = ["fake", "true"]


def arquivo_uri(path: Path) -> str:
    return f"sqlite:///{path.resolve()}"


def configurar_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        if tracking_uri.startswith(("http://", "https://", "databricks")):
            client.create_experiment(experiment_name)
        else:
            artifacts = (ROOT_DIR / "mlartifacts" / experiment_name).resolve().as_uri()
            client.create_experiment(experiment_name, artifact_location=artifacts)
    mlflow.set_experiment(experiment_name)


def interpretar_resposta(resposta: str) -> str:
    texto = resposta.upper()
    if "VERDADEIRA" in texto or "TRUE" in texto:
        return "true"
    if "FALSA" in texto or "FAKE" in texto:
        return "fake"
    return "indefinido"


def montar_prompt(tokenizer, texto: str) -> str:
    mensagens = [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": f"Notícia:\n{texto}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    return f"{PROMPT_SISTEMA}\n\nNotícia:\n{texto}\nResposta:"


def main(args: argparse.Namespace) -> None:
    if not args.test_csv.exists():
        raise FileNotFoundError(f"Conjunto de teste não encontrado: '{args.test_csv}'.")
    if not (args.adapters_dir / "adapters.safetensors").exists():
        raise FileNotFoundError(f"Adaptador não encontrado em '{args.adapters_dir}'.")

    dados = pd.read_csv(args.test_csv).dropna(subset=["texto", "label"])
    if args.limit:
        dados = dados.head(args.limit).copy()
    if args.visual:
        cabecalho("Avaliação Fake.br", f"Amostras: {len(dados)} | Adaptador: {args.adapters_dir.name}")
        console.print("[dim]Carregando modelo local...[/dim]")
    modelo, tokenizer = load(args.model, adapter_path=str(args.adapters_dir))

    respostas, predicoes = [], []
    progresso = Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console, disable=not args.visual,
    )
    with progresso:
        tarefa = progresso.add_task("Avaliando notícias", total=len(dados))
        for indice, linha in enumerate(dados.itertuples(index=False), start=1):
            resposta = generate(modelo, tokenizer, prompt=montar_prompt(tokenizer, linha.texto), max_tokens=8, verbose=False)
            respostas.append(resposta)
            predicoes.append(interpretar_resposta(resposta))
            progresso.advance(tarefa)
            if not args.visual:
                print(f"[{indice}/{len(dados)}] esperado={linha.label} predito={predicoes[-1]}", flush=True)

    avaliaveis = [predicao if predicao in ROTULOS else "fake" for predicao in predicoes]
    precisao, recall, f1, _ = precision_recall_fscore_support(
        dados["label"], avaliaveis, labels=ROTULOS, average="macro", zero_division=0
    )
    rodada = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.results_dir / f"predicoes_fakebr_{rodada}.csv"
    report_path = args.results_dir / f"metricas_fakebr_{rodada}.json"
    dados = dados.assign(predicao=predicoes, resposta_modelo=respostas)
    dados.to_csv(predictions_path, index=False)

    metricas = {
        "accuracy": accuracy_score(dados["label"], avaliaveis),
        "precision_macro": precisao,
        "recall_macro": recall,
        "f1_macro": f1,
        "indefinidas": sum(predicao == "indefinido" for predicao in predicoes),
    }
    relatorio = {
        "metadados": {"gerado_em_utc": datetime.now(timezone.utc).isoformat(), "modelo": args.model, "amostras": len(dados)},
        "metricas": metricas,
        "por_classe": classification_report(dados["label"], avaliaveis, labels=ROTULOS, output_dict=True, zero_division=0),
    }
    report_path.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")

    configurar_mlflow(args.tracking_uri, args.experiment_name)
    with mlflow.start_run(run_name=f"evaluation-fakebr-{rodada}"):
        mlflow.set_tags({"dataset": "Fake.br", "stage": "evaluation"})
        mlflow.log_params({"model": args.model, "adapters_dir": str(args.adapters_dir.resolve()), "test_csv": str(args.test_csv.resolve())})
        mlflow.log_metrics(metricas)
        mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")

    if args.visual:
        tabela_metricas(metricas, "Métricas Fake.br")
        console.print(f"[green]✓[/green] Métricas: {report_path}")
    else:
        print(f"[✓] Métricas: {report_path}")
        print(json.dumps(metricas, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Avalia o modelo no teste estratificado do Fake.br.")
    parser.add_argument("--test-csv", type=Path, default=ROOT_DIR / "data/fakebr/test.csv")
    parser.add_argument("--adapters-dir", type=Path, default=ROOT_DIR / "models/adapters")
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_REPO") or "mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--limit", type=int, help="Limita a avaliação; útil para smoke test.")
    parser.add_argument("--visual", action="store_true", help="Mostra progresso e métricas em painel no terminal.")
    parser.add_argument("--experiment-name", default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or "sift-lora-fine-tuning")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI") or arquivo_uri(ROOT_DIR / "mlflow.db"))
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit deve ser positivo.")
    main(args)
