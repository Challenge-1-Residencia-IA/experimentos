#!/usr/bin/env python3
"""Avalia o adaptador de verificacao no conjunto de teste nunca usado no treino."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from evidencias_utils import ROTULOS_EVIDENCIAS, extrair_json, normalizar_veredito, validar_resposta


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def carregar_jsonl(caminho: Path) -> list[dict]:
    registros = []
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, 1):
            try:
                item = json.loads(linha)
            except json.JSONDecodeError as erro:
                raise ValueError(f"JSON invalido em {caminho}:{numero}: {erro}") from erro
            mensagens = item.get("messages", [])
            if len(mensagens) < 3 or mensagens[-1].get("role") != "assistant":
                raise ValueError(f"Exemplo sem resposta assistant em {caminho}:{numero}")
            registros.append(item)
    return registros


def total_fontes(mensagens: list[dict]) -> int:
    texto_usuario = next((m.get("content", "") for m in mensagens if m.get("role") == "user"), "")
    indices = [int(valor) for valor in re.findall(r"(?m)^\[(\d+)\]", texto_usuario)]
    return max(indices, default=0)


def main(args: argparse.Namespace) -> None:
    if not args.test_jsonl.exists():
        raise FileNotFoundError(f"Teste nao encontrado: '{args.test_jsonl}'.")
    if not (args.adapters_dir / "adapters.safetensors").exists():
        raise FileNotFoundError(f"Adaptador nao encontrado em '{args.adapters_dir}'.")

    from mlx_lm import generate, load

    dados = carregar_jsonl(args.test_jsonl)
    if args.limit:
        dados = dados[:args.limit]
    modelo, tokenizer = load(args.model, adapter_path=str(args.adapters_dir))
    linhas = []
    for indice, item in enumerate(dados, 1):
        mensagens = item["messages"]
        alvo = extrair_json(mensagens[-1]["content"]) or {}
        esperado = normalizar_veredito(alvo.get("veredito"))
        prompt = tokenizer.apply_chat_template(mensagens[:-1], tokenize=False, add_generation_prompt=True)
        resposta = generate(modelo, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        qtd_fontes = total_fontes(mensagens)
        predito = validar_resposta(resposta, qtd_fontes)
        fontes_esperadas = alvo.get("fontes_usadas", []) if isinstance(alvo.get("fontes_usadas", []), list) else []
        abstencao_esperada = esperado == "INCONCLUSIVA"
        linhas.append({
            "id": indice,
            "esperado": esperado,
            "predito": predito["veredito"],
            "acertou": esperado == predito["veredito"],
            "json_valido": predito["json_valido"],
            "citacoes_validas": predito["citacoes_validas"],
            "fontes_esperadas": json.dumps(fontes_esperadas),
            "fontes_preditas": json.dumps(predito["fontes_usadas"]),
            "abstencao_esperada": abstencao_esperada,
            "abstencao_correta": (not abstencao_esperada) or predito["veredito"] == "INCONCLUSIVA",
            "resposta_bruta": resposta,
        })
        print(f"[{indice}/{len(dados)}] esperado={esperado} predito={predito['veredito']}", flush=True)

    quadro = pd.DataFrame(linhas)
    y_true, y_pred = quadro["esperado"], quadro["predito"]
    presentes = [rotulo for rotulo in ROTULOS_EVIDENCIAS if rotulo in set(y_true)]
    relatorio_classes = classification_report(
        y_true, y_pred, labels=presentes, output_dict=True, zero_division=0
    )
    matriz = confusion_matrix(y_true, y_pred, labels=list(ROTULOS_EVIDENCIAS)).tolist()
    inconclusivos = quadro[quadro["abstencao_esperada"]]
    metricas = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(relatorio_classes["macro avg"]["precision"]),
        "recall_macro": float(relatorio_classes["macro avg"]["recall"]),
        "f1_macro": float(relatorio_classes["macro avg"]["f1-score"]),
        "taxa_json_valido": float(quadro["json_valido"].mean()),
        "taxa_citacoes_validas": float(quadro["citacoes_validas"].mean()),
        "taxa_abstencao_correta": float(inconclusivos["abstencao_correta"].mean()) if len(inconclusivos) else 0.0,
    }
    rodada = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / f"predicoes_evidencias_{rodada}.csv"
    json_path = args.results_dir / f"metricas_evidencias_{rodada}.json"
    quadro.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "metadados": {
            "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "adapters_dir": str(args.adapters_dir.resolve()),
            "test_jsonl": str(args.test_jsonl.resolve()),
            "amostras": len(quadro),
        },
        "metricas": metricas,
        "rotulos_matriz_confusao": list(ROTULOS_EVIDENCIAS),
        "matriz_confusao": matriz,
        "por_classe": relatorio_classes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metricas, ensure_ascii=False, indent=2))
    print(f"Resultados: {json_path} e {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia o modelo de alegacao + evidencias.")
    parser.add_argument("--test-jsonl", type=Path, default=ROOT_DIR / "data/evidencias/test.jsonl")
    parser.add_argument("--adapters-dir", type=Path, default=ROOT_DIR / "models/adapters-evidencias")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int)
    argumentos = parser.parse_args()
    if argumentos.max_tokens <= 0 or (argumentos.limit is not None and argumentos.limit <= 0):
        parser.error("--max-tokens e --limit devem ser positivos.")
    main(argumentos)
