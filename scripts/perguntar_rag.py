"""Consulta a tutora RAG com uma pergunta ou alegação."""

import argparse
import json
import os
from pathlib import Path

from rag.tutor import ask, ask_online
from rag.web import WebResearcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pergunta")
    parser.add_argument("--local", action="store_true", help="Usa somente o índice local, sem pesquisa na web")
    parser.add_argument("--mlflow-local", type=Path, metavar="DIRETORIO",
                        help="Grava esta consulta em um MLflow temporário local")
    args = parser.parse_args()
    if args.mlflow_local:
        base = args.mlflow_local.expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{base / 'mlflow.db'}"
        os.environ["MLFLOW_LOCAL_ARTIFACT_ROOT"] = str(base / "artifacts")
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "NexoVerdict-Temporario"
    answer = ask(args.pergunta) if args.local else ask_online(args.pergunta, WebResearcher())
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
