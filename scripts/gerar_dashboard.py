#!/usr/bin/env python3
"""Gera um dashboard a partir de um relatório produzido por metricas_tecnicas.py."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT_DIR / "results/relatorio_metricas_tecnicas.json"
DEFAULT_OUTPUT = ROOT_DIR / "results/dashboard_metricas.png"


def percentual(valor: float) -> float:
    return float(valor) * 100


def main(json_path: Path, output_path: Path) -> None:
    with json_path.open("r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    extracao, recuperacao = dados["extracao"], dados["recuperacao"]
    divergencia, classificacao, geracao = dados["divergencia"], dados["classificacao"], dados["geracao"]
    metricas = [
        ("Precisão macro (classificação)", percentual(classificacao["precisao_macro"])),
        ("Recall macro (classificação)", percentual(classificacao["recall_macro"])),
        ("F1 macro (classificação)", percentual(classificacao["f1_score_macro"])),
        ("Extração correta de afirmações", percentual(extracao["taxa_identificacao_correta"])),
        ("Qualidade das fontes", percentual(recuperacao["qualidade_fontes_score"])),
        ("Taxa de alucinação", percentual(geracao["taxa_alucinacao"])),
        ("Respostas conclusivas sem evidência", percentual(recuperacao["taxa_respostas_sem_evidencia"])),
        ("Precisão de divergências", percentual(divergencia["precisao_identificacao_divergencias"])),
    ]
    classes = ["FALSA", "VERDADEIRA", "FORA DE CONTEXTO", "OPINIÃO", "SÁTIRA"]
    f1_por_classe = [percentual(classificacao["por_classe"][classe]["f1_score"]) for classe in classes]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=180)
    fig.suptitle("Copiloto Sift AI — Métricas técnicas por etapa", fontsize=16, fontweight="bold", y=0.98)

    ax1 = axes[0, 0]
    nomes, valores = zip(*metricas)
    cores = ["#ef4444" if "alucinação" in nome.lower() or "sem evidência" in nome.lower() else "#10b981" if valor >= 70 else "#f59e0b" if valor >= 30 else "#ef4444" for nome, valor in metricas]
    barras = ax1.barh(np.arange(len(nomes)), valores, color=cores, edgecolor="#334155")
    ax1.set_yticks(np.arange(len(nomes)), nomes, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Percentual (%)")
    ax1.set_title("As 8 métricas definidas")
    for barra in barras:
        ax1.text(barra.get_width() + 1, barra.get_y() + barra.get_height() / 2, f"{barra.get_width():.1f}%", va="center", fontsize=9)

    ax2 = axes[0, 1]
    barras = ax2.bar(classes, f1_por_classe, color="#3b82f6", edgecolor="#334155")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("F1-score (%)")
    ax2.set_title("F1 por classe")
    ax2.tick_params(axis="x", rotation=20)
    for barra in barras:
        ax2.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 2, f"{barra.get_height():.1f}%", ha="center")

    ax3 = axes[1, 0]
    matriz = classificacao["matriz_confusao"]
    matriz_array = np.array([[matriz[real][predita] for predita in classes] for real in classes])
    imagem = ax3.imshow(matriz_array, cmap="Blues")
    ax3.set_xticks(range(len(classes)), classes, rotation=30, ha="right", fontsize=8)
    ax3.set_yticks(range(len(classes)), classes, fontsize=8)
    ax3.set_xlabel("Predição")
    ax3.set_ylabel("Gabarito")
    ax3.set_title("Matriz de confusão")
    for linha in range(len(classes)):
        for coluna in range(len(classes)):
            ax3.text(coluna, linha, str(matriz_array[linha, coluna]), ha="center", va="center")
    fig.colorbar(imagem, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = axes[1, 1]
    ax4.axis("off")
    cobertura = 100 - percentual(recuperacao["taxa_respostas_sem_evidencia"])
    texto = (
        "DIAGNÓSTICO GERADO A PARTIR DO JSON\n"
        "────────────────────────────────────\n\n"
        f"Casos avaliados: {classificacao['total_amostras']}\n"
        f"Acurácia global: {percentual(classificacao['acuracia_global']):.1f}%\n"
        f"Extração correta: {percentual(extracao['taxa_identificacao_correta']):.1f}%\n"
        f"Fontes confiáveis: {percentual(recuperacao['qualidade_fontes_score']):.1f}%\n"
        f"Cobertura com evidência: {cobertura:.1f}%\n"
        f"Precisão / recall de divergência: {percentual(divergencia['precisao_identificacao_divergencias']):.1f}% / {percentual(divergencia['recall_divergencias']):.1f}%\n"
        f"Alucinação lexical: {percentual(geracao['taxa_alucinacao']):.1f}%\n\n"
        "A métrica de alucinação é lexical; valide-a\n"
        "com revisão humana ou suporte semântico."
    )
    ax4.text(0.04, 0.95, texto, transform=ax4.transAxes, va="top", fontfamily="monospace", fontsize=10,
             bbox={"boxstyle": "round,pad=1", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"})

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Dashboard salvo em: '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dashboard de métricas técnicas.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input_json.exists():
        parser.error(f"Relatório não encontrado: {args.input_json}")
    main(args.input_json, args.output)
