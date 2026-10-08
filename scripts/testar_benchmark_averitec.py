"""Prepara, indexa e avalia AVeriTeC PT-BR em um índice RAG separado."""

import argparse
import json

from rag.store import VectorStore
from scripts.avaliar_rag_mlflow import evaluate
from scripts.preparar_benchmark_averitec import BASE, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=10, help="Casos por classe; padrão: 10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true", help="Baixa e prepara, sem indexar nem chamar o Qwen")
    args = parser.parse_args()
    manifest = prepare(args.per_class, args.seed, BASE)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return
    index_path = BASE / "chroma"
    print("Indexando evidências do benchmark...", flush=True)
    print(VectorStore(path=index_path).index(BASE / "documents.jsonl"), flush=True)
    print("Avaliando com o Qwen e registrando no MLflow...", flush=True)
    metrics, run_id, uri = evaluate(
        BASE / "casos.jsonl", f"averitec-ptbr-{args.per_class}x3-seed{args.seed}",
        index_path=index_path, corpus_path=BASE / "documents.jsonl",
    )
    print(json.dumps({"metricas": metrics, "run_id": run_id, "tracking_uri": uri},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
