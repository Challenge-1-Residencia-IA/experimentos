"""Consulta a tutora RAG com uma pergunta ou alegação."""

import argparse
import json

from rag.tutor import ask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pergunta")
    args = parser.parse_args()
    print(json.dumps(ask(args.pergunta), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
