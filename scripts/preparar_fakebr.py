#!/usr/bin/env python3
"""Prepara o Fake.br para fine-tuning e avaliação estratificada."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "raw/preprocessed/pre-processed.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data/fakebr"
PROMPT_SISTEMA = """Classifique a notícia como FALSA ou VERDADEIRA.
Responda exclusivamente com uma dessas duas palavras, sem justificativa."""


def normalizar_rotulo(valor: object) -> str:
    rotulo = str(valor).strip().lower()
    mapa = {"fake": "fake", "falsa": "fake", "true": "true", "verdadeira": "true"}
    if rotulo not in mapa:
        raise ValueError(f"Rótulo não suportado no Fake.br: {valor!r}")
    return mapa[rotulo]


def exemplo_treino(texto: str, rotulo: str) -> dict:
    resposta = "FALSA" if rotulo == "fake" else "VERDADEIRA"
    return {
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": f"Notícia:\n{texto}"},
            {"role": "assistant", "content": resposta},
        ]
    }


def salvar_jsonl(dados: pd.DataFrame, destino: Path) -> None:
    with destino.open("w", encoding="utf-8") as arquivo:
        for linha in dados.itertuples(index=False):
            arquivo.write(json.dumps(exemplo_treino(linha.texto, linha.label), ensure_ascii=False) + "\n")


def main(args: argparse.Namespace) -> None:
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Arquivo do Fake.br não encontrado: '{args.input_csv}'.")

    dados = pd.read_csv(args.input_csv, usecols=["label", "preprocessed_news"])
    dados = dados.rename(columns={"preprocessed_news": "texto"}).dropna(subset=["label", "texto"])
    dados["texto"] = dados["texto"].astype(str).str.strip()
    dados = dados[dados["texto"] != ""].copy()
    dados["label"] = dados["label"].map(normalizar_rotulo)

    treino, teste = train_test_split(
        dados, test_size=args.test_size, random_state=args.seed, stratify=dados["label"]
    )
    tamanho_validacao_relativo = args.valid_size / (1 - args.test_size)
    treino, validacao = train_test_split(
        treino,
        test_size=tamanho_validacao_relativo,
        random_state=args.seed,
        stratify=treino["label"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    salvar_jsonl(treino, args.output_dir / "train.jsonl")
    salvar_jsonl(validacao, args.output_dir / "valid.jsonl")
    teste.to_csv(args.output_dir / "test.csv", index=False)

    print(f"[✓] Fake.br preparado em '{args.output_dir}'.")
    print(f"  Treino: {len(treino)} | Validação: {len(validacao)} | Teste: {len(teste)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cria splits estratificados do Fake.br.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_size < 1 or not 0 < args.valid_size < 1 or args.test_size + args.valid_size >= 1:
        parser.error("--test-size e --valid-size devem ser positivos e somar menos de 1.")
    main(args)
