#!/usr/bin/env python3
"""Amostra notícias do corpus, busca evidências web e as verifica com Ollama."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from ddgs import DDGS
from dotenv import load_dotenv
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ui_terminal import cabecalho, console, painel_veredito, tabela_fontes


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "raw/preprocessed/pre-processed.csv"
DEFAULT_OUTPUT = ROOT_DIR / "results/verificacao_amostral.json"
OLLAMA_URL = "http://localhost:11434/api/chat"

PROMPT_EXTRACAO = """Extraia a principal alegação factual do texto abaixo.
Responda somente com a alegação, em uma frase curta. Não conclua se ela é verdadeira ou falsa.

TEXTO:
{texto}"""

PROMPT_VERIFICACAO = """Você é um verificador de fatos. Avalie a alegação usando SOMENTE as evidências fornecidas.
Retorne apenas um JSON válido neste formato:
{{
  "classificacao": "VERDADEIRA|FALSA|INCONCLUSIVA",
  "confianca": "ALTA|MÉDIA|BAIXA",
  "justificativa": "até duas frases, citando os índices das fontes",
  "fontes_usadas": [1, 2]
}}

Se as evidências não forem suficientes, use INCONCLUSIVA. Não use o seu conhecimento prévio.

ALEGAÇÃO:
{alegacao}

EVIDÊNCIAS:
{evidencias}"""


def chamar_ollama(modelo: str, prompt: str, timeout: int) -> str:
    resposta = requests.post(
        OLLAMA_URL,
        json={"model": modelo, "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=timeout,
    )
    resposta.raise_for_status()
    return resposta.json()["message"]["content"].strip()


def buscar_evidencias(alegacao: str, max_resultados: int) -> list[dict[str, str]]:
    try:
        with DDGS() as busca:
            resultados = busca.text(alegacao[:300], region="br-pt", max_results=max_resultados)
            return [
                {"titulo": item.get("title", ""), "url": item.get("href", ""), "trecho": item.get("body", "")}
                for item in resultados
            ]
    except Exception as erro:
        return [{"titulo": "Erro na busca", "url": "", "trecho": str(erro)}]


def formatar_evidencias(fontes: list[dict[str, str]]) -> str:
    if not fontes:
        return "Nenhuma evidência recuperada."
    return "\n\n".join(
        f"[{indice}] {fonte['titulo']}\nURL: {fonte['url']}\nTrecho: {fonte['trecho']}"
        for indice, fonte in enumerate(fontes, 1)
    )


def json_do_modelo(texto: str) -> dict[str, Any]:
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim == -1:
        return {"classificacao": "INCONCLUSIVA", "confianca": "BAIXA", "justificativa": texto, "fontes_usadas": []}
    try:
        return json.loads(texto[inicio : fim + 1])
    except json.JSONDecodeError:
        return {"classificacao": "INCONCLUSIVA", "confianca": "BAIXA", "justificativa": texto, "fontes_usadas": []}


def main(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {args.dataset}")

    dados = pd.read_csv(args.dataset)
    for coluna in ("label", "preprocessed_news"):
        if coluna not in dados.columns:
            raise ValueError(f"Coluna obrigatória ausente: {coluna}")
    if args.quantidade <= 0 or args.quantidade > len(dados):
        raise ValueError(f"--quantidade deve estar entre 1 e {len(dados)}")

    amostra = dados.sample(n=args.quantidade, random_state=args.seed)
    resultados = []
    if args.visual:
        cabecalho("Amostragem e verificação com evidências", f"Modelo Ollama: {args.modelo} | Amostras: {args.quantidade}")
    progresso = Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console, disable=not args.visual,
    )
    with progresso:
        tarefa = progresso.add_task("Verificando notícias", total=args.quantidade)
        for indice, (_, linha) in enumerate(amostra.iterrows(), 1):
            texto = str(linha["preprocessed_news"])
            if not args.visual:
                print(f"[{indice}/{args.quantidade}] Extraindo alegação...", flush=True)
            alegacao = chamar_ollama(args.modelo, PROMPT_EXTRACAO.format(texto=texto[:args.max_chars]), args.timeout)
            if not args.visual:
                print(f"[{indice}/{args.quantidade}] Pesquisando evidências...", flush=True)
            fontes = buscar_evidencias(alegacao, args.max_resultados)
            if not args.visual:
                print(f"[{indice}/{args.quantidade}] Classificando com as evidências...", flush=True)
            verificacao = json_do_modelo(
                chamar_ollama(
                    args.modelo,
                    PROMPT_VERIFICACAO.format(alegacao=alegacao, evidencias=formatar_evidencias(fontes)),
                    args.timeout,
                )
            )
            resultado = {
                "id_amostra": indice,
                "rotulo_dataset": str(linha["label"]),
                "alegacao_extraida": alegacao,
                "classificacao_modelo": verificacao,
                "fontes": fontes,
            }
            resultados.append(resultado)
            progresso.advance(tarefa)
            if args.visual:
                painel_veredito(verificacao.get("classificacao", "INCONCLUSIVA"), verificacao.get("justificativa", ""), f"Amostra {indice}: rótulo do dataset = {linha['label']}")
                tabela_fontes(fontes, verificacao.get("fontes_usadas", []))
            else:
                print(f"[{indice}/{args.quantidade}] {verificacao.get('classificacao', 'INCONCLUSIVA')}: {alegacao}", flush=True)

    saida = {"modelo": args.modelo, "seed": args.seed, "quantidade": args.quantidade, "resultados": resultados}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.visual:
        console.print(f"[green]✓[/green] Resultado salvo em: {args.output}")
    else:
        print(f"Resultado salvo em: {args.output}")


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Pesquisa e verifica uma amostra aleatória do corpus com Ollama.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--quantidade", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modelo", default=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
    parser.add_argument("--max-resultados", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--visual", action="store_true", help="Mostra progresso, vereditos e fontes em painéis no terminal.")
    main(parser.parse_args())
