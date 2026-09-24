#!/usr/bin/env python3
"""Prepara dados, treina e avalia o verificador baseado em evidencias."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]


def executar(comando: list[str]) -> None:
    print(f"\n$ {' '.join(comando)}\n", flush=True)
    subprocess.run(comando, cwd=ROOT_DIR, check=True)


def main(args: argparse.Namespace) -> None:
    executar([
        sys.executable, "scripts/preparar_dataset_evidencias.py",
        "--source", str(args.source),
        "--output-dir", str(args.data_dir),
        "--seed", str(args.seed),
        "--experiment-name", args.experiment_name,
        "--tracking-uri", args.tracking_uri,
        "--balance-strategy", args.balance_strategy,
        "--synthetic-inconclusive", str(args.synthetic_inconclusive),
        "--balance-target", str(args.balance_target),
    ])
    treino = [
        sys.executable, "scripts/treinar_visual.py",
        "--data-dir", str(args.data_dir),
        "--adapters-dir", str(args.adapters_dir),
        "--model", args.model,
        "--iters", str(args.iters),
        "--learning-rate", str(args.learning_rate),
        "--max-seq-length", str(args.max_seq_length),
        "--steps-per-eval", str(args.steps_per_eval),
        "--save-every", str(args.steps_per_eval),
        "--seed", str(args.seed),
    ]
    if args.resume_adapter_file:
        treino.extend(["--resume-adapter-file", str(args.resume_adapter_file)])
    if args.overwrite:
        treino.append("--overwrite")
    executar(treino)
    if not args.evaluate_test:
        print("Teste reservado. Avalie-o uma vez ao finalizar os ajustes do modelo.")
        return
    avaliacao = [
        sys.executable, "scripts/avaliar_evidencias.py",
        "--test-jsonl", str(args.data_dir / "test.jsonl"),
        "--adapters-dir", str(args.adapters_dir),
        "--model", args.model,
        "--results-dir", str(args.results_dir),
        "--experiment-name", args.experiment_name,
        "--tracking-uri", args.tracking_uri,
    ]
    if args.limit_test:
        avaliacao.extend(["--limit", str(args.limit_test)])
    if args.visual:
        avaliacao.append("--visual")
    executar(avaliacao)


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Pipeline reproduzivel de verificacao com evidencias.")
    parser.add_argument("--source", type=Path, default=ROOT_DIR / "raw/factckbr/FACTCKBR.tsv")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data/evidencias-por-url-v1")
    parser.add_argument("--adapters-dir", type=Path, default=ROOT_DIR / "models/adapters-evidencias-por-url-v1")
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_REPO") or "mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--steps-per-eval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-strategy", choices=("none", "oversample", "undersample"), default="oversample")
    parser.add_argument("--synthetic-inconclusive", type=int, default=30)
    parser.add_argument("--balance-target", type=int, default=150)
    parser.add_argument("--resume-adapter-file", type=Path)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--evaluate-test", action="store_true", help="Avalia no teste reservado somente ao finalizar os ajustes.")
    parser.add_argument("--visual", action="store_true", help="Ativa o painel visual durante a avaliação final.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or "sift-lora-fine-tuning",
        help="Nome compartilhado pelo treino e pela avaliação no MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(ROOT_DIR / 'mlflow.db').resolve()}",
        help="Endereço do servidor ou banco do MLflow.",
    )
    argumentos = parser.parse_args()
    if any(valor <= 0 for valor in (argumentos.iters, argumentos.learning_rate, argumentos.max_seq_length, argumentos.steps_per_eval)):
        parser.error("Parametros de treino devem ser positivos.")
    if argumentos.limit_test is not None and (argumentos.limit_test <= 0 or not argumentos.evaluate_test):
        parser.error("--limit-test exige --evaluate-test e deve ser positivo.")
    main(argumentos)
