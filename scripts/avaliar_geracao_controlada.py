"""Smoke de geração real com fontes sintéticas; não mede busca web nem acurácia geral."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from rag.mlflow_model import ollama_digest
from rag.tutor import RESPONSE_VERSION, SYSTEM_PROMPT, ask


CLAIM = "O projeto Aurora foi aprovado?"


def source(number, text, group=None):
    return {"id": str(number), "text": text, "distance": 0.1, "metadata": {
        "kind": "evidencia", "title": CLAIM, "source": "Fonte sintética",
        "origin_group": group or f"fonte-{number}", "url": f"https://fonte-{number}.example/aurora",
        "published_at": "", "conclusive_eligible": True, "relevancia": "direta"}}


def run(model):
    supporting = [source(1, "O projeto Aurora foi aprovado pelo conselho nesta sessão."),
                  source(2, "A ata confirma que o projeto Aurora recebeu aprovação unânime.")]
    opposing = [source(1, "O projeto Aurora não foi aprovado pelo conselho nesta sessão."),
                source(2, "É falso que o projeto Aurora foi aprovado; a proposta foi rejeitada.")]
    cases = [
        ("duas_fontes_favoraveis", supporting, "EVIDENCIAS_SUFICIENTES_FAVORAVEIS"),
        ("duas_fontes_contrarias", opposing, "EVIDENCIAS_SUFICIENTES_CONTRARIAS"),
        ("mesma_origem", [source(i + 1, item["text"], "mesma-origem")
                          for i, item in enumerate(supporting)], "EVIDENCIAS_INSUFICIENTES"),
        ("sem_evidencias", [], "EVIDENCIAS_INSUFICIENTES"),
        ("contraria_sem_qualificacao", [
            {**opposing[0], "metadata": {**opposing[0]["metadata"], "conclusive_eligible": False}}
        ], "EVIDENCIAS_INSUFICIENTES"),
        ("fontes_conflitantes", [supporting[0], opposing[1]], "EVIDENCIAS_CONFLITANTES"),
    ]
    report = {"modelo": model, "digest": ollama_digest(model), "versao": RESPONSE_VERSION,
              "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              "escopo": "Fontes sintéticas; geração real; sem busca web ou medição de generalização.",
              "casos": []}
    for name, evidence, expected in cases:
        started = time.perf_counter()
        diagnostics = {}
        answer = ask(CLAIM, evidence=evidence, teaching=[], track=False, model_name=model,
                     generation_diagnostics=diagnostics)
        synthesis_ok = bool(answer.get("sintese")) and answer.get("aviso_ia") == "Sou uma IA e posso cometer erros."
        if name == "contraria_sem_qualificacao":
            synthesis_ok = synthesis_ok and "contradizem a afirmação" in answer["sintese"]
        result = {"caso": name, "esperado": expected,
                  "correto": answer["avaliacao_evidencias"] == expected and synthesis_ok,
                  "sintese_verificada": synthesis_ok,
                  "latencia_s": round(time.perf_counter() - started, 3),
                  "diagnostico": diagnostics, "resposta": answer}
        report["casos"].append(result)
        print(f'{name}: {answer["avaliacao_evidencias"]} ({result["latencia_s"]}s)', flush=True)
    report["acertos"] = sum(item["correto"] for item in report["casos"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--output", type=Path, default=Path("results/geracao-controlada-rodada2.json"))
    args = parser.parse_args()
    report = run(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'Resultado: {report["acertos"]}/{len(report["casos"])}; relatório: {args.output}')


if __name__ == "__main__":
    main()
