"""Consulta a tutora RAG com uma pergunta ou alegação."""

import argparse
import json

from rag.tutor import ask, ask_online
from rag.web import WebResearcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pergunta")
    parser.add_argument("--local", action="store_true", help="Usa somente o índice local, sem pesquisa na web")
    args = parser.parse_args()
    answer = ask(args.pergunta) if args.local else ask_online(args.pergunta, WebResearcher())
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
