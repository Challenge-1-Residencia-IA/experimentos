"""Baixa uma versão registrada no DagsHub e executa uma pergunta."""

from __future__ import annotations

import argparse
import json

import mlflow
import pandas as pd

from rag.dagshub_tracking import REGISTERED_MODEL_NAME, configure_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pergunta")
    parser.add_argument("--version", default="1", help="Versão do Model Registry")
    parser.add_argument("--modo", choices=("web", "local"), default="web")
    args = parser.parse_args()

    configure_tracking()
    uri = f"models:/{REGISTERED_MODEL_NAME}/{args.version}"
    model = mlflow.pyfunc.load_model(uri)
    output = model.predict(pd.DataFrame({"pergunta": [args.pergunta], "modo": [args.modo]}))[0]
    print(json.dumps(json.loads(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
