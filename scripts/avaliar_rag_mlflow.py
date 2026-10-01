"""Avalia respostas da tutora e registra uma execução no MLflow local."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from rag.corpus import CORPUS, ROOT
from rag.store import MODEL, STORE, VectorStore
from rag.tutor import SYSTEM_PROMPT, VERDICTS, ask

DEFAULT_CASES = ROOT / "data" / "avaliacao" / "casos.jsonl"
EVIDENCE_K = 6  # O mesmo limite usado pela tutora.


def read_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            if not isinstance(case, dict) or not isinstance(case.get("pergunta"), str) or not case["pergunta"].strip():
                raise ValueError(f"Linha {number}: informe uma pergunta não vazia")
            label = case.get("veredito_revisado")
            if label is not None and label not in VERDICTS:
                raise ValueError(f"Linha {number}: veredito_revisado inválido: {label}")
            urls = case.get("urls_esperadas", [])
            if not isinstance(urls, list) or any(not isinstance(url, str) or not url for url in urls):
                raise ValueError(f"Linha {number}: urls_esperadas deve ser uma lista de URLs")
            cases.append({"id": case.get("id", str(number)), "pergunta": case["pergunta"],
                          "veredito_revisado": label, "urls_esperadas": urls,
                          "justificativa_revisada": case.get("justificativa_revisada")})
    if not cases:
        raise ValueError("O arquivo de casos está vazio")
    return cases


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "ausente"


def git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "desconhecida"


def summarize(details: list[dict]) -> dict[str, float]:
    total = len(details)
    valid = [item for item in details if item["erro"] is None]
    labeled = [item for item in details if item["veredito_revisado"] is not None]
    retrieval = [item for item in details if item["urls_esperadas"]]
    metrics = {
        "casos": float(total),
        "respostas_validas": float(len(valid)),
        "taxa_respostas_validas": len(valid) / total,
        "latencia_media_s": sum(item["latencia_s"] for item in details) / total,
    }
    if valid:
        metrics["documentos_citados_media"] = sum(len(item["resposta"]["fontes"]) for item in valid) / len(valid)
        metrics["evidencias_citadas_media"] = sum(
            sum(source["tipo"] == "evidencia" for source in item["resposta"]["fontes"])
            for item in valid
        ) / len(valid)
    if labeled:
        metrics["casos_com_veredito_revisado"] = float(len(labeled))
        metrics["acuracia_veredito"] = sum(item["veredito"] == item["veredito_revisado"] for item in labeled) / len(labeled)
        f1_scores = []
        for label in sorted({item["veredito_revisado"] for item in labeled}):
            expected = sum(item["veredito_revisado"] == label for item in labeled)
            predicted = sum(item["veredito"] == label for item in labeled)
            correct = sum(item["veredito_revisado"] == item["veredito"] == label for item in labeled)
            metrics[f"recall_{label.lower()}"] = correct / expected
            f1_scores.append(2 * correct / (expected + predicted) if expected + predicted else 0.0)
        metrics["f1_macro"] = sum(f1_scores) / len(f1_scores)
        abstention = [item for item in labeled if item["veredito_revisado"] == "INCONCLUSIVA"]
        if abstention:
            metrics["taxa_abstencao_correta"] = sum(item["veredito"] == "INCONCLUSIVA" for item in abstention) / len(abstention)
    if retrieval:
        metrics[f"hit_at_{EVIDENCE_K}"] = sum(item["rank_url_esperada"] is not None for item in retrieval) / len(retrieval)
        metrics["mrr"] = sum(1 / item["rank_url_esperada"] for item in retrieval if item["rank_url_esperada"] is not None) / len(retrieval)
        metrics["casos_com_urls_esperadas"] = float(len(retrieval))
    return metrics


def evaluate(
    cases_path: Path = DEFAULT_CASES,
    run_name: str | None = None,
    *,
    index_path: Path = STORE,
    corpus_path: Path = CORPUS,
) -> tuple[dict, str, str]:
    load_dotenv(ROOT / ".env")
    cases_path = cases_path.resolve()
    index_path = index_path.resolve()
    corpus_path = corpus_path.resolve()
    cases = read_cases(cases_path)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "NexoVerdict-RAG")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=(ROOT / "mlartifacts" / "rag").as_uri())
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name or f"avaliacao-{cases_path.stem}") as run:
        mlflow.log_params({
            "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            "embedding_model": MODEL,
            "evidence_k": EVIDENCE_K,
            "teaching_k": 3,
            "casos_sha256": file_hash(cases_path),
            "corpus_sha256": file_hash(corpus_path),
            "index_path": str(index_path),
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "git_revision": git_revision(),
        })
        if not index_path.exists():
            raise FileNotFoundError("Banco vetorial ausente. Execute a indexação antes da avaliação.")
        store = VectorStore(path=index_path)
        details = []
        for case in cases:
            start = time.perf_counter()
            item = {**case, "rank_url_esperada": None, "urls_recuperadas": [],
                    "grupos_recuperados": [], "veredito": None, "resposta": None, "erro": None}
            try:
                evidence = store.search(case["pergunta"], "evidencia", limit=EVIDENCE_K, max_per_source=2)
                teaching = store.search(case["pergunta"], "educativo", limit=3, max_per_source=2)
                item["urls_recuperadas"] = [hit["metadata"]["url"] for hit in evidence]
                item["grupos_recuperados"] = [hit["metadata"]["origin_group"] for hit in evidence]
                expected = set(case["urls_esperadas"])
                item["rank_url_esperada"] = next(
                    (rank for rank, url in enumerate(item["urls_recuperadas"], 1) if url in expected), None
                )
                answer = ask(case["pergunta"], store, evidence=evidence, teaching=teaching)
                item["veredito"] = answer["veredito"]
                item["resposta"] = answer
            except Exception as error:
                item["erro"] = f"{type(error).__name__}: {error}"
            item["latencia_s"] = round(time.perf_counter() - start, 3)
            details.append(item)
            print(f"{case['id']}: {item['veredito'] or 'ERRO'} ({item['latencia_s']:.1f}s)", flush=True)
        metrics = summarize(details)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"metricas": metrics, "casos": details}, "avaliacao/relatorio.json")
        mlflow.set_tag("avaliacao", "rag_completo")
        return metrics, run.info.run_id, tracking_uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("casos", type=Path, nargs="?", default=DEFAULT_CASES,
                        help="JSONL revisado; padrão: data/avaliacao/casos.jsonl")
    parser.add_argument("--run-name", help="Nome da execução no MLflow")
    parser.add_argument("--index-path", type=Path, default=STORE, help="Índice Chroma da avaliação")
    parser.add_argument("--corpus-path", type=Path, default=CORPUS, help="Corpus usado para criar o índice")
    args = parser.parse_args()
    metrics, run_id, tracking_uri = evaluate(
        args.casos, args.run_name, index_path=args.index_path, corpus_path=args.corpus_path
    )
    print(json.dumps({"metricas": metrics, "run_id": run_id, "tracking_uri": tracking_uri}, ensure_ascii=False, indent=2))
    print(f"Abra o painel: {sys.executable} -m mlflow ui --backend-store-uri {tracking_uri}")


if __name__ == "__main__":
    main()
