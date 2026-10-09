"""Avalia respostas da tutora e registra uma execução no MLflow local."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from rag.corpus import CORPUS, ROOT
from rag.dagshub_tracking import REGISTERED_MODEL_NAME, configure_tracking, contract_fingerprint, matching_model_version
from rag.store import MODEL, STORE, VectorStore
from rag.metrics import citation_metrics
from rag.splitter import CHUNKING_VERSION
from rag.tutor import SYSTEM_PROMPT, ANSWER_SCHEMA, EVALUATIONS, RESPONSE_VERSION, ask

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
            label = case.get("avaliacao_revisada")
            if label is not None and label not in EVALUATIONS:
                raise ValueError(f"Linha {number}: avaliacao_revisada inválida: {label}")
            urls = case.get("urls_esperadas", [])
            if not isinstance(urls, list) or any(not isinstance(url, str) or not url for url in urls):
                raise ValueError(f"Linha {number}: urls_esperadas deve ser uma lista de URLs")
            reserved = case.get("espera_sem_evidencia", False)
            if not isinstance(reserved, bool):
                raise ValueError(f"Linha {number}: espera_sem_evidencia deve ser booleano")
            cases.append({"id": case.get("id", str(number)), "pergunta": case["pergunta"],
                          "avaliacao_revisada": label, "urls_esperadas": urls,
                          "espera_sem_evidencia": reserved})
    if not cases:
        raise ValueError("O arquivo de casos está vazio")
    return cases


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "ausente"


def git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "desconhecida"


def summarize(details: list[dict], *, retrieval_only: bool = False) -> dict[str, float]:
    total = len(details)
    if not total:
        raise ValueError("Nenhum caso para avaliar")
    completed = [item for item in details if item["erro"] is None]
    valid = [item for item in completed if item["resposta"] is not None]
    labeled = [item for item in valid if item["avaliacao_revisada"] is not None]
    retrieval = [item for item in completed if item["urls_esperadas"]]
    metrics = {
        "casos": float(total),
        "casos_com_erro": float(total - len(completed)),
        "taxa_erros": (total - len(completed)) / total,
        "latencia_media_s": sum(item["latencia_s"] for item in details) / total,
    }
    if not retrieval_only:
        metrics["respostas_validas"] = float(len(valid))
        metrics["taxa_respostas_validas"] = len(valid) / total
    if valid:
        metrics["documentos_citados_media"] = sum(len(item["resposta"]["fontes"]) for item in valid) / len(valid)
        metrics["evidencias_citadas_media"] = sum(
            sum(source["tipo"] == "evidencia" for source in item["resposta"]["fontes"])
            for item in valid
        ) / len(valid)
        metrics["fontes_independentes_citadas_media"] = sum(
            len({source["grupo_origem"] for source in item["resposta"]["fontes"]
                 if source["tipo"] == "evidencia"}) for item in valid
        ) / len(valid)
        citations = [citation_metrics(item["resposta"]["fontes"], item["resposta"]) for item in valid]
        evaluated = sum(item["citacoes_avaliadas"] for item in citations)
        resolved = sum(item["citacoes_resolvidas"] for item in citations)
        metrics["citacoes_avaliadas"] = evaluated
        metrics["taxa_respostas_com_citacoes"] = sum(item["resposta_com_citacoes"] for item in citations) / len(valid)
        if evaluated:
            metrics["taxa_citacoes_validas"] = resolved / evaluated
        for evaluation in sorted(EVALUATIONS):
            count = sum(item["avaliacao_evidencias"] == evaluation for item in valid)
            key = evaluation.casefold()
            metrics[f"avaliacoes_{key}"] = float(count)
            metrics[f"taxa_{key}"] = count / len(valid)
    if labeled:
        metrics["casos_com_avaliacao_revisada"] = float(len(labeled))
        metrics["concordancia_avaliacao"] = sum(item["avaliacao_evidencias"] == item["avaliacao_revisada"] for item in labeled) / len(labeled)
        expected_abstention = [item for item in labeled if item["avaliacao_revisada"] in {
            "EVIDENCIAS_INSUFICIENTES", "EVIDENCIAS_CONFLITANTES"}]
        if expected_abstention:
            metrics["taxa_abstencao_correta"] = sum(
                item["avaliacao_evidencias"] == item["avaliacao_revisada"]
                for item in expected_abstention
            ) / len(expected_abstention)
    reviewed = [item for item in details if item.get("avaliacao_revisada") is not None]
    metrics["casos_revisados_total"] = float(len(reviewed))
    metrics["casos_sem_gabarito_avaliacao"] = float(total - len(reviewed))
    if reviewed and not retrieval_only:
        def prediction(item):
            return item.get("avaliacao_evidencias") if item["erro"] is None else None

        metrics["concordancia_total_revisada"] = sum(
            prediction(item) == item["avaliacao_revisada"] for item in reviewed
        ) / len(reviewed)
        active = {item["avaliacao_revisada"] for item in reviewed} | {
            prediction(item) for item in reviewed if prediction(item) in EVALUATIONS
        }
        scores = []
        for label in sorted(active):
            tp = sum(prediction(item) == label == item["avaliacao_revisada"] for item in reviewed)
            predicted = sum(prediction(item) == label for item in reviewed)
            support = sum(item["avaliacao_revisada"] == label for item in reviewed)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            scores.append(f1)
            metrics.update({f"precision_{label.lower()}": precision,
                            f"recall_{label.lower()}": recall,
                            f"f1_{label.lower()}": f1})
        metrics["f1_macro_avaliacao"] = sum(scores) / len(scores)
        conclusive = {"EVIDENCIAS_SUFICIENTES_FAVORAVEIS", "EVIDENCIAS_SUFICIENTES_CONTRARIAS"}
        metrics["taxa_conclusao_incorreta_revisada"] = sum(
            prediction(item) in conclusive and prediction(item) != item["avaliacao_revisada"]
            for item in reviewed
        ) / len(reviewed)
    if valid:
        metrics["taxa_com_limitacoes"] = sum(bool(item["resposta"]["limitacoes"].strip()) for item in valid) / len(valid)
    generation = [item["diagnostico_geracao"] for item in valid if item.get("diagnostico_geracao")]
    if generation:
        metrics["tentativas_geracao_media"] = sum(item["tentativas_geracao"] for item in generation) / len(generation)
        metrics["taxa_primeira_resposta_valida"] = sum(item["primeira_resposta_valida"] for item in generation) / len(generation)
        metrics["taxa_correcao_automatica_sucesso"] = sum(item["correcao_automatica_sucesso"] for item in generation) / len(generation)
        metrics["taxa_fallback_seguro"] = sum(item["fallback_seguro"] for item in generation) / len(generation)
        metrics["taxa_afirmacao_original_preservada"] = sum(
            item["afirmacao_original_preservada"] for item in generation
        ) / len(generation)
    reserved = [item for item in valid if item.get("espera_sem_evidencia")]
    if reserved:
        metrics["casos_reservados_sem_evidencia"] = float(len(reserved))
        metrics["taxa_resposta_inconclusiva_casos_reservados"] = sum(
            item["avaliacao_evidencias"] in {"EVIDENCIAS_INSUFICIENTES", "EVIDENCIAS_CONFLITANTES"}
            for item in reserved
        ) / len(reserved)
    if retrieval:
        metrics[f"hit_at_{EVIDENCE_K}"] = sum(item["rank_url_esperada"] is not None for item in retrieval) / len(retrieval)
        metrics["mrr"] = sum(1 / item["rank_url_esperada"] for item in retrieval if item["rank_url_esperada"] is not None) / len(retrieval)
        metrics["casos_com_urls_esperadas"] = float(len(retrieval))
        metrics[f"precision_at_{EVIDENCE_K}"] = sum(item["precision_at_k"] for item in retrieval) / len(retrieval)
        metrics[f"recall_at_{EVIDENCE_K}"] = sum(item["recall_at_k"] for item in retrieval) / len(retrieval)
        metrics[f"ndcg_at_{EVIDENCE_K}"] = sum(item["ndcg_at_k"] for item in retrieval) / len(retrieval)
        metrics[f"taxa_documentos_irrelevantes_at_{EVIDENCE_K}"] = 1 - metrics[f"precision_at_{EVIDENCE_K}"]
    return metrics


def evaluate(
    cases_path: Path = DEFAULT_CASES,
    run_name: str | None = None,
    *,
    index_path: Path = STORE,
    corpus_path: Path = CORPUS,
    retrieval_only: bool = False,
) -> tuple[dict, str, str]:
    load_dotenv(ROOT / ".env")
    cases_path = cases_path.resolve()
    index_path = index_path.resolve()
    corpus_path = corpus_path.resolve()
    cases = read_cases(cases_path)
    tracking_uri = configure_tracking()
    with mlflow.start_run(run_name=run_name or f"avaliacao-{cases_path.stem}") as run:
        mlflow.log_params({
            "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            "embedding_model": MODEL,
            "evidence_k": EVIDENCE_K,
            "teaching_k": 3,
            "casos_sha256": file_hash(cases_path),
            "corpus_sha256": file_hash(corpus_path),
            "index_path": str(index_path),
            "retrieval_only": retrieval_only,
            **contract_fingerprint(SYSTEM_PROMPT, ANSWER_SCHEMA),
            "response_version": RESPONSE_VERSION,
            "chunking_version": CHUNKING_VERSION,
            "git_revision": git_revision(),
        })
        version = matching_model_version(contract_fingerprint(SYSTEM_PROMPT, ANSWER_SCHEMA)["contract_sha256"], file_hash(corpus_path))
        if version:
            mlflow.set_tags({"registered_model_name": REGISTERED_MODEL_NAME,
                             "registered_model_version": version})
        mlflow.log_text(SYSTEM_PROMPT, "versao/prompt.txt")
        mlflow.log_dict(ANSWER_SCHEMA, "versao/schema.json")
        mlflow.log_artifact(str(ROOT / "rag" / "tutor.py"), artifact_path="versao")
        if not index_path.exists():
            raise FileNotFoundError("Banco vetorial ausente. Execute a indexação antes da avaliação.")
        store = VectorStore(path=index_path)
        details = []
        for case in cases:
            start = time.perf_counter()
            item = {**case, "rank_url_esperada": None, "urls_recuperadas": [],
                    "grupos_recuperados": [], "avaliacao_evidencias": None, "resposta": None, "erro": None,
                    "diagnostico_geracao": {},
                    "precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}
            try:
                evidence = store.search(case["pergunta"], "evidencia", limit=EVIDENCE_K, max_per_source=2)
                teaching = store.search(case["pergunta"], "educativo", limit=3, max_per_source=2)
                item["urls_recuperadas"] = [hit["metadata"]["url"] for hit in evidence]
                item["grupos_recuperados"] = [hit["metadata"]["origin_group"] for hit in evidence]
                expected = set(case["urls_esperadas"])
                item["rank_url_esperada"] = next(
                    (rank for rank, url in enumerate(item["urls_recuperadas"], 1) if url in expected), None
                )
                relevant_positions = [
                    rank for rank, url in enumerate(item["urls_recuperadas"], 1) if url in expected
                ]
                relevant_urls = set(item["urls_recuperadas"]) & expected
                item["precision_at_k"] = len(relevant_positions) / len(item["urls_recuperadas"]) if item["urls_recuperadas"] else 0.0
                item["recall_at_k"] = len(relevant_urls) / len(expected) if expected else 0.0
                dcg = sum(1 / math.log2(rank + 1) for rank in relevant_positions)
                ideal = min(len(expected), len(item["urls_recuperadas"]))
                idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal + 1))
                item["ndcg_at_k"] = dcg / idcg if idcg else 0.0
                if not retrieval_only:
                    answer = ask(
                        case["pergunta"], store, evidence=evidence, teaching=teaching, track=False,
                        generation_diagnostics=item["diagnostico_geracao"],
                    )
                    item["avaliacao_evidencias"] = answer["avaliacao_evidencias"]
                    item["resposta"] = answer
            except Exception as error:
                item["erro"] = f"{type(error).__name__}: {error}"
            item["latencia_s"] = round(time.perf_counter() - start, 3)
            details.append(item)
            status = "RECUPERACAO" if retrieval_only and item["erro"] is None else item["avaliacao_evidencias"] or "ERRO"
            print(f"{case['id']}: {status} ({item['latencia_s']:.1f}s)", flush=True)
        metrics = summarize(details, retrieval_only=retrieval_only)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"metricas": metrics, "casos": details}, "avaliacao/relatorio.json")
        mlflow.set_tag("avaliacao", "recuperacao" if retrieval_only else "rag_completo")
        return metrics, run.info.run_id, tracking_uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("casos", type=Path, nargs="?", default=DEFAULT_CASES,
                        help="JSONL revisado; padrão: data/avaliacao/casos.jsonl")
    parser.add_argument("--run-name", help="Nome da execução no MLflow")
    parser.add_argument("--index-path", type=Path, default=STORE, help="Índice Chroma da avaliação")
    parser.add_argument("--corpus-path", type=Path, default=CORPUS, help="Corpus usado para criar o índice")
    parser.add_argument("--retrieval-only", action="store_true", help="Mede apenas a busca, sem chamar o Qwen")
    args = parser.parse_args()
    metrics, run_id, tracking_uri = evaluate(
        args.casos, args.run_name, index_path=args.index_path, corpus_path=args.corpus_path,
        retrieval_only=args.retrieval_only,
    )
    print(json.dumps({"metricas": metrics, "run_id": run_id, "tracking_uri": tracking_uri}, ensure_ascii=False, indent=2))
    print(f"MLflow: {tracking_uri}")


if __name__ == "__main__":
    main()
