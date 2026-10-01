#!/usr/bin/env python3
"""Avalia um adapter de perguntas no conjunto reservado e registra no MLflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from perguntas_utils import montar_prompt, validar_resposta_factual
from responder_pergunta import carregar_modelo, configurar_mlflow, validar_adapter_perguntas

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def ler_jsonl(caminho: Path):
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, 1):
            if linha.strip():
                try:
                    yield json.loads(linha)
                except json.JSONDecodeError as erro:
                    raise ValueError(f"JSON inválido em {caminho}:{numero}") from erro


def normalizar_tokens(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.casefold())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", re.sub(r"\[\d+\]", " ", texto))


def f1_tokens(esperado: str, previsto: str) -> float:
    a, b = Counter(normalizar_tokens(esperado)), Counter(normalizar_tokens(previsto))
    comuns = sum((a & b).values())
    if not a or not b or not comuns:
        return 0.0
    precisao, revocacao = comuns / sum(b.values()), comuns / sum(a.values())
    return 2 * precisao * revocacao / (precisao + revocacao)


def contrato_exemplo(item: dict) -> tuple[list[dict], dict, int, int]:
    mensagens = item.get("messages")
    if not isinstance(mensagens, list) or len(mensagens) < 3 or mensagens[-1].get("role") != "assistant":
        raise ValueError("Exemplo sem conversa completa ou resposta de referência.")
    referencia = json.loads(mensagens[-1]["content"])
    texto_usuario = next((m.get("content", "") for m in reversed(mensagens[:-1]) if m.get("role") == "user"), "")
    total_fontes = len({int(i) for i in re.findall(r"(?m)^\[(\d+)\]", texto_usuario)})
    sistema = next((m.get("content", "") for m in mensagens if m.get("role") == "system"), "")
    achado = re.search(r"pelo menos\s+(\d+)\s+fonte", sistema)
    minimo = int(achado.group(1)) if achado else 1
    return mensagens[:-1], referencia, total_fontes, minimo


def avaliar(args: argparse.Namespace) -> dict:
    from mlx_lm import generate

    validar_adapter_perguntas(args.adapters_dir)
    exemplos = list(ler_jsonl(args.test_jsonl))
    if args.limit:
        exemplos = exemplos[:args.limit]
    if not exemplos:
        raise ValueError("O conjunto de teste está vazio.")
    modelo, tokenizer = carregar_modelo(args.model, args.adapters_dir)
    detalhes = []
    for numero, item in enumerate(exemplos, 1):
        mensagens, referencia, total_fontes, minimo = contrato_exemplo(item)
        prompt = montar_prompt(tokenizer, mensagens)
        bruto = generate(modelo, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False).strip()
        previsto = validar_resposta_factual(bruto, total_fontes, minimo)
        detalhes.append({
            "exemplo": numero, "referencia": referencia, "previsto": previsto,
            "resposta_bruta": bruto,
            "status_correto": previsto["status"] == referencia.get("status"),
            "f1_resposta": f1_tokens(str(referencia.get("resposta", "")), previsto.get("resposta", "")),
        })
        print(f"[{numero}/{len(exemplos)}] status={previsto['status']} json={previsto['json_valido']}", flush=True)
    n = len(detalhes)
    metricas = {
        "exemplos": n,
        "taxa_json_valido": sum(d["previsto"]["json_valido"] for d in detalhes) / n,
        "taxa_citacoes_validas": sum(d["previsto"]["citacoes_validas"] for d in detalhes) / n,
        "acuracia_status": sum(d["status_correto"] for d in detalhes) / n,
        "f1_resposta_medio": sum(d["f1_resposta"] for d in detalhes) / n,
        "taxa_abstencao": sum(d["previsto"]["status"] == "INCONCLUSIVA" for d in detalhes) / n,
    }
    return {"metricas": metricas, "detalhes": detalhes}


def main(args: argparse.Namespace) -> None:
    resultado = avaliar(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    configurar_mlflow(args.tracking_uri, args.experiment_name)
    with mlflow.start_run(run_name="avaliacao-perguntas") as run:
        mlflow.log_params({"modelo": args.model, "adaptador": str(args.adapters_dir),
                           "teste": str(args.test_jsonl), "limite": args.limit or 0,
                           "max_tokens": args.max_tokens})
        mlflow.set_tags({"tarefa": "avaliacao_perguntas", "biblioteca": "mlx-lm"})
        mlflow.log_metrics(resultado["metricas"])
        mlflow.log_artifact(str(args.output), artifact_path="avaliacao")
        resultado["mlflow_run_id"] = run.info.run_id
    args.output.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado["metricas"], ensure_ascii=False, indent=2))
    print(f"Relatório: {args.output}\nMLflow run: {resultado['mlflow_run_id']}")


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, default=ROOT_DIR / "data/perguntas-mescladas-v1/test.jsonl")
    parser.add_argument("--adapters-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_REPO") or DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "results/avaliacao_perguntas.json")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(ROOT_DIR / 'mlflow.db').resolve()}")
    parser.add_argument("--experiment-name", default="nexoverdict-perguntas-avaliacao")
    args = parser.parse_args()
    if (args.limit is not None and args.limit <= 0) or args.max_tokens <= 0:
        parser.error("--limit e --max-tokens devem ser positivos.")
    main(args)
