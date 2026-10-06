"""Mede a recuperação em casos revisados por pessoas (JSONL)."""

import argparse
import json
from pathlib import Path

from rag.store import STORE, VectorStore


def evaluate_retrieval(casos: Path, k: int = 5, index_path: Path = STORE) -> tuple[list[dict], dict]:
    if k < 1:
        raise ValueError("k deve ser positivo")
    store = VectorStore(path=index_path)
    ranks = []
    details = []
    with casos.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            expected = set(case["urls_esperadas"])
            if not expected:
                continue
            found = store.search(case["pergunta"], "evidencia", limit=k, max_per_source=2)
            rank = next((position for position, item in enumerate(found, 1) if item["metadata"]["url"] in expected), None)
            ranks.append(rank)
            details.append({"caso": number, "rank": rank, "urls_recuperadas": [item["metadata"]["url"] for item in found]})
    if not ranks:
        raise ValueError("Nenhum caso encontrado")
    summary = {
        "casos": len(ranks),
        f"hit@{k}": sum(rank is not None for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / len(ranks),
    }
    return details, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("casos", type=Path, help="JSONL com pergunta e urls_esperadas")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index-path", type=Path, default=STORE)
    args = parser.parse_args()
    details, summary = evaluate_retrieval(args.casos, args.k, args.index_path.resolve())
    for detail in details:
        print(json.dumps(detail, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
