#!/usr/bin/env python3
"""Executa preparo, fine-tuning e avaliação do Fake.br em uma única rodada."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "raw/preprocessed/pre-processed.csv"
DEFAULT_DATA_DIR = ROOT_DIR / "data/fakebr"
DEFAULT_ADAPTERS_DIR = ROOT_DIR / "models/adapters"
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def executar(comando: list[str]) -> None:
    print(f"\n$ {' '.join(comando)}\n", flush=True)
    subprocess.run(comando, check=True, cwd=ROOT_DIR)


def main(args: argparse.Namespace) -> None:
    if not args.input_csv.exists():
        raise FileNotFoundError(f"CSV do Fake.br não encontrado: '{args.input_csv}'.")

    executar([
        sys.executable, "scripts/preparar_fakebr.py",
        "--input-csv", str(args.input_csv),
        "--output-dir", str(args.data_dir),
        "--test-size", str(args.test_size),
        "--valid-size", str(args.valid_size),
        "--seed", str(args.seed),
    ])
    executar([
        sys.executable, "scripts/treinar_visual.py",
        "--data-dir", str(args.data_dir),
        "--adapters-dir", str(args.adapters_dir),
        "--model", args.model,
        "--iters", str(args.iters),
        "--experiment-name", args.experiment_name,
        "--tracking-uri", args.tracking_uri,
    ])
    comando_avaliacao = [
        sys.executable, "scripts/avaliar_fakebr.py",
        "--test-csv", str(args.data_dir / "test.csv"),
        "--adapters-dir", str(args.adapters_dir),
        "--model", args.model,
        "--results-dir", str(args.results_dir),
        "--experiment-name", args.experiment_name,
        "--tracking-uri", args.tracking_uri,
    ]
    if args.limit_test:
        comando_avaliacao.extend(["--limit", str(args.limit_test)])
    if args.visual:
        comando_avaliacao.append("--visual")
    executar(comando_avaliacao)


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Pipeline completa de classificação Fake.br.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--adapters-dir", type=Path, default=DEFAULT_ADAPTERS_DIR)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_REPO") or DEFAULT_MODEL)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-test", type=int, help="Avalia só N exemplos; útil para smoke test.")
    parser.add_argument("--visual", action="store_true", help="Ativa o painel visual durante a avaliação final.")
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--experiment-name", default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or "fakebr-classificacao")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(ROOT_DIR / 'mlflow.db').resolve()}")
    args = parser.parse_args()
    if args.iters <= 0:
        parser.error("--iters deve ser positivo.")
    if not 0 < args.test_size < 1 or not 0 < args.valid_size < 1 or args.test_size + args.valid_size >= 1:
        parser.error("--test-size e --valid-size devem ser positivos e somar menos de 1.")
    main(args)
