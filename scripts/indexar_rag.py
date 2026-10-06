import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.corpus import CORPUS
from rag.store import STORE, VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria ou atualiza o índice vetorial do RAG.")
    parser.add_argument("--corpus-path", type=Path, default=CORPUS)
    parser.add_argument("--index-path", type=Path, default=STORE)
    args = parser.parse_args()
    corpus_path = args.corpus_path.resolve()
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus não encontrado: {corpus_path}")
    store = VectorStore(path=args.index_path.resolve())
    counts = store.index(corpus_path)
    counts["evidence_chunks"] = store.collections["evidencia"].count()
    counts["educational_chunks"] = store.collections["educativo"].count()
    print(counts)


if __name__ == "__main__":
    main()
