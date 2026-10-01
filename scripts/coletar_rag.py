"""Coleta documentos descritos em um arquivo JSON de fontes."""

import argparse
from pathlib import Path

from rag.corpus import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("sources.example.json"))
    args = parser.parse_args()
    print(ingest(args.config.resolve()))


if __name__ == "__main__":
    main()
