#!/usr/bin/env python3
"""Gera gráficos para uma execução de amostragem e verificação web."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "results/verificacao_amostral.json"
DEFAULT_OUTPUT = ROOT_DIR / "results/dashboard_verificacao_amostral.png"


def percentual(parte: int, total: int) -> float:
    return (parte / total * 100) if total else 0.0


def main(entrada: Path, saida: Path) -> None:
    dados = json.loads(entrada.read_text(encoding="utf-8"))
    resultados = dados.get("resultados", [])
    if not resultados:
        raise ValueError("O JSON não contém resultados para visualizar.")

    total = len(resultados)
    classificacoes = Counter(r["classificacao_modelo"].get("classificacao", "INCONCLUSIVA") for r in resultados)
    confiancas = Counter(r["classificacao_modelo"].get("confianca", "NÃO INFORMADA") for r in resultados)
    fontes_por_caso = [len(r.get("fontes", [])) for r in resultados]
    com_url = sum(1 for r in resultados for fonte in r.get("fontes", []) if fonte.get("url"))
    total_fontes = sum(fontes_por_caso)
    esperada = {"fake": "FALSA", "true": "VERDADEIRA"}
    conclusivos = [r for r in resultados if r["classificacao_modelo"].get("classificacao") in {"FALSA", "VERDADEIRA"}]
    concordancias = sum(
        r["classificacao_modelo"].get("classificacao") == esperada.get(r.get("rotulo_dataset", "").lower())
        for r in conclusivos
    )

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=180)
    fig.suptitle("Verificação amostral com busca web e Qwen", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    nomes = ["VERDADEIRA", "FALSA", "INCONCLUSIVA"]
    valores = [classificacoes[nome] for nome in nomes]
    barras = ax.bar(nomes, valores, color=["#10b981", "#ef4444", "#f59e0b"])
    ax.set_title("Vereditos do modelo")
    ax.set_ylabel("Mensagens")
    for barra in barras:
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.05, str(int(barra.get_height())), ha="center")

    ax = axes[0, 1]
    nomes = ["ALTA", "MÉDIA", "BAIXA", "NÃO INFORMADA"]
    valores = [confiancas[nome] for nome in nomes]
    ax.bar(nomes, valores, color="#3b82f6")
    ax.set_title("Confiança declarada")
    ax.set_ylabel("Mensagens")

    ax = axes[1, 0]
    ax.bar(range(1, total + 1), fontes_por_caso, color="#8b5cf6")
    ax.set_title("Fontes recuperadas por mensagem")
    ax.set_xlabel("ID da amostra")
    ax.set_ylabel("Fontes")
    ax.set_xticks(range(1, total + 1))

    ax = axes[1, 1]
    cobertura = percentual(com_url, total_fontes)
    taxa_conclusiva = percentual(len(conclusivos), total)
    acordo = percentual(concordancias, len(conclusivos))
    texto = (
        "RESUMO DA EXECUÇÃO\n"
        "────────────────────────\n\n"
        f"Modelo: {dados.get('modelo', 'não informado')}\n"
        f"Amostras: {total}\n"
        f"Fontes recuperadas: {total_fontes}\n"
        f"Fontes com URL: {cobertura:.1f}%\n"
        f"Vereditos conclusivos: {taxa_conclusiva:.1f}%\n"
        f"Concordância com o corpus*: {acordo:.1f}%\n\n"
        "*Calculada apenas para FALSA e VERDADEIRA.\n"
        "O rótulo do corpus é referência de avaliação,\n"
        "não evidência usada pelo modelo."
    )
    ax.axis("off")
    ax.text(0.05, 0.94, texto, va="top", transform=ax.transAxes, fontfamily="monospace", fontsize=11,
            bbox={"boxstyle": "round,pad=1", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"})

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=250, bbox_inches="tight")
    print(f"Dashboard salvo em: {saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dashboard da verificação amostral.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.input_json, args.output)
