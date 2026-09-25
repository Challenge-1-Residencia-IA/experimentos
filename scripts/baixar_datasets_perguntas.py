#!/usr/bin/env python3
"""Baixa datasets públicos de QA para uma pasta isolada; não prepara nem treina."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from pathlib import Path

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOGO = {
    "faquad": {
        "repo": "liafacom/faquad",
        "license": "CC-BY-4.0",
        "homepage": "https://github.com/liafacom/faquad",
        "arquivos": {
            "train": "https://raw.githubusercontent.com/liafacom/faquad/master/data/train.json",
            "valid": "https://raw.githubusercontent.com/liafacom/faquad/master/data/dev.json",
        },
    },
    "pira": {
        "repo": "paulopirozelli/pira",
        "license": "CC-BY-4.0",
        "homepage": "https://huggingface.co/datasets/paulopirozelli/pira",
        "config": "default",
        "splits": ("train", "validation", "test"),
    },
    "rag_ptbr": {
        "repo": "Madras1/rag-qa-fulltext-ptbr",
        "license": "ODC-By-1.0",
        "homepage": "https://huggingface.co/datasets/Madras1/rag-qa-fulltext-ptbr",
        "config": "qa_pairs",
        "splits": ("train",),
    },
}


def baixar_json(url: str, destino: Path, timeout: int) -> None:
    resposta = requests.get(url, timeout=timeout, headers={"User-Agent": "sift-ai-dataset-importer/1.0"})
    resposta.raise_for_status()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(resposta.content)


def baixar_linhas_hf(
    repo: str, config: str, split: str, destino: Path, limite: int, timeout: int, retomar: bool = False
) -> int:
    destino.parent.mkdir(parents=True, exist_ok=True)
    total = sum(1 for _ in destino.open(encoding="utf-8")) if retomar and destino.exists() else 0
    with destino.open("a" if total else "w", encoding="utf-8") as arquivo:
        while total < limite:
            tamanho = min(100, limite - total)
            consulta = urllib.parse.urlencode({
                "dataset": repo, "config": config, "split": split,
                "offset": total, "length": tamanho,
            })
            url = f"https://datasets-server.huggingface.co/rows?{consulta}"
            for tentativa in range(6):
                resposta = requests.get(
                    url, timeout=timeout, headers={"User-Agent": "sift-ai-dataset-importer/1.0"}
                )
                if resposta.status_code != 429:
                    break
                time.sleep(min(60, 2 ** (tentativa + 2)))
            resposta.raise_for_status()
            pagina = resposta.json()
            linhas = pagina.get("rows", [])
            if not linhas:
                break
            for item in linhas:
                arquivo.write(json.dumps(item.get("row", {}), ensure_ascii=False) + "\n")
            total += len(linhas)
            if len(linhas) < tamanho:
                break
            time.sleep(0.05)
    return total


def main(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"'{args.output_dir}' já contém arquivos. Use outro caminho ou --resume.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contagens: dict[str, int] = {}
    selecionados = args.datasets or list(CATALOGO)
    for nome in selecionados:
        info = CATALOGO[nome]
        pasta = args.output_dir / nome
        pasta.mkdir(parents=True, exist_ok=True)
        if nome == "faquad":
            for split, url in info["arquivos"].items():
                destino = pasta / f"{split}.json"
                if not destino.exists() or destino.stat().st_size == 0:
                    baixar_json(url, destino, args.timeout)
                contagens[f"{nome}/{split}"] = -1
            continue
        limite = args.rag_limit if nome == "rag_ptbr" else args.pira_limit
        for split in info["splits"]:
            destino = pasta / f"{split}.jsonl"
            contagens[f"{nome}/{split}"] = baixar_linhas_hf(
                info["repo"], info["config"], split, destino, limite, args.timeout, args.resume
            )
    manifesto = {
        "datasets": {nome: CATALOGO[nome] for nome in selecionados},
        "contagens": contagens,
        "observacao": "Arquivos brutos públicos isolados. Nenhum treino foi iniciado.",
    }
    (args.output_dir / "manifesto.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifesto, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "raw/publicos-perguntas-v1")
    parser.add_argument("--datasets", nargs="+", choices=sorted(CATALOGO))
    parser.add_argument("--rag-limit", type=int, default=5000, help="Amostra do RAG PT-BR sintético.")
    parser.add_argument("--pira-limit", type=int, default=10000, help="Limite por split do Pirá.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    argumentos = parser.parse_args()
    if argumentos.rag_limit <= 0 or argumentos.pira_limit <= 0 or argumentos.timeout <= 0:
        parser.error("Limites e timeout devem ser positivos.")
    main(argumentos)
