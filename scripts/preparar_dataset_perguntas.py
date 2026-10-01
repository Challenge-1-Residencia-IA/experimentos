#!/usr/bin/env python3
"""Cria exemplos de perguntas com explicações a partir de fontes ou exemplos revisados."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from perguntas_utils import DEFAULT_PROMPTS_DIR, formatar_evidencias, montar_mensagens_pergunta


ROOT_DIR = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def ler_jsonl(caminho: Path):
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, 1):
            if linha.strip():
                try:
                    yield json.loads(linha)
                except json.JSONDecodeError as erro:
                    raise ValueError(f"JSON inválido em {caminho}:{numero}") from erro


def exemplo_leitura(item: dict, prompts_dir: Path = DEFAULT_PROMPTS_DIR) -> dict | None:
    """Ensina leitura da fonte; não converte rótulos de checagem em verdades."""
    try:
        usuario = next(m["content"] for m in item["messages"] if m["role"] == "user")
        alegacao = usuario.split("ALEGACAO:\n", 1)[1].split("\n\nEVIDENCIAS:\n", 1)[0].strip()
        bloco = usuario.split("\n\nEVIDENCIAS:\n", 1)[1]
        titulo = bloco.split("[1] Fonte: ", 1)[1].split("\n", 1)[0]
        url = bloco.split("\nURL: ", 1)[1].split("\n", 1)[0]
        trecho = bloco.split("\nTrecho: ", 1)[1].strip()
    except (KeyError, IndexError, StopIteration, TypeError):
        return None
    if not (alegacao and url and trecho and len(trecho) >= 40):
        return None
    fonte = {"titulo": titulo, "url": url, "trecho": trecho}
    pergunta = f"O que o trecho da fonte informa sobre a alegação: {alegacao}?"
    resposta = f"O trecho informa: {trecho[:360].rstrip()} [1]"
    justificativa = f"Esta é uma descrição do conteúdo do trecho fornecido em [1], não uma confirmação independente da alegação."
    mensagens = montar_mensagens_pergunta(pergunta, "evidencias", formatar_evidencias([fonte]), 1, prompts_dir)
    mensagens.append({"role": "assistant", "content": json.dumps({
        "status": "RESPONDIDA", "resposta": resposta,
        "justificativa": justificativa, "fontes_usadas": [1],
    }, ensure_ascii=False)})
    return {"messages": mensagens}


def exemplo_revisado(item: dict, prompts_dir: Path = DEFAULT_PROMPTS_DIR) -> tuple[str, dict]:
    split = item.get("split")
    modo = item.get("modo")
    pergunta = item.get("pergunta")
    resposta = item.get("resposta")
    justificativa = item.get("justificativa")
    if split not in SPLITS or modo not in {"conversa", "evidencias"}:
        raise ValueError("Exemplo revisado exige split train|valid|test e modo conversa|evidencias.")
    if not all(isinstance(v, str) and v.strip() for v in (pergunta, resposta, justificativa)):
        raise ValueError("Exemplo revisado exige pergunta, resposta e justificativa não vazias.")
    if modo == "conversa":
        mensagens = montar_mensagens_pergunta(pergunta, modo, prompts_dir=prompts_dir)
        saida = {"resposta": resposta, "justificativa": justificativa, "fontes_usadas": []}
    else:
        fontes = item.get("fontes")
        usadas = item.get("fontes_usadas")
        status = item.get("status")
        if not isinstance(fontes, list) or not fontes or status not in {"RESPONDIDA", "INCONCLUSIVA"}:
            raise ValueError("Exemplo de evidências exige fontes e status RESPONDIDA|INCONCLUSIVA.")
        if not all(isinstance(f, dict) and f.get("url") and f.get("trecho") for f in fontes):
            raise ValueError("Cada fonte exige url e trecho.")
        if not isinstance(usadas, list) or any(type(i) is not int or i < 1 or i > len(fontes) for i in usadas):
            raise ValueError("fontes_usadas deve conter apenas índices válidos.")
        minimo_fontes = item.get("minimo_fontes", 2 if len(fontes) >= 2 else 1)
        if type(minimo_fontes) is not int or minimo_fontes < 1 or minimo_fontes > len(fontes):
            raise ValueError("minimo_fontes deve ser um inteiro entre 1 e a quantidade de fontes.")
        if status == "RESPONDIDA" and (
            len(usadas) < minimo_fontes
            or any(f"[{i}]" not in resposta or f"[{i}]" not in justificativa for i in usadas)
        ):
            raise ValueError("Resposta conclusiva exige citações na resposta e na justificativa.")
        mensagens = montar_mensagens_pergunta(
            pergunta, modo, formatar_evidencias(fontes), minimo_fontes, prompts_dir
        )
        saida = {"status": status, "resposta": resposta, "justificativa": justificativa, "fontes_usadas": usadas}
    mensagens.append({"role": "assistant", "content": json.dumps(saida, ensure_ascii=False)})
    return split, {"messages": mensagens}


def preparar(args: argparse.Namespace) -> dict:
    if any((args.output_dir / f"{split}.jsonl").exists() for split in SPLITS) and not args.overwrite:
        raise FileExistsError(f"Dataset já existe em {args.output_dir}; use outro diretório ou --overwrite.")
    por_split = {split: [] for split in SPLITS}
    contagem = Counter()
    if args.evidence_data_dir:
        for split in SPLITS:
            caminho = args.evidence_data_dir / f"{split}.jsonl"
            if not caminho.exists():
                raise FileNotFoundError(caminho)
            for item in ler_jsonl(caminho):
                convertido = exemplo_leitura(item, args.prompts_dir)
                if convertido:
                    por_split[split].append(convertido)
                    contagem[f"leitura_{split}"] += 1
    if args.curated_jsonl:
        for item in ler_jsonl(args.curated_jsonl):
            split, convertido = exemplo_revisado(item, args.prompts_dir)
            por_split[split].append(convertido)
            contagem[f"revisados_{split}"] += 1
    if not por_split["train"] or not por_split["valid"]:
        raise ValueError("São necessários exemplos em train e valid.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, itens in por_split.items():
        destino = args.output_dir / f"{split}.jsonl"
        if itens:
            with destino.open("w", encoding="utf-8") as arquivo:
                for item in itens:
                    arquivo.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif args.overwrite and destino.exists():
            destino.unlink()
    resumo = {
        "contagens": dict(contagem),
        "aviso": "Exemplos automáticos ensinam a descrever trechos, não a determinar veracidade nem a responder perguntas gerais. Para isso, acrescente exemplos revisados de perguntas reais.",
        "fonte_evidencias": str(args.evidence_data_dir or ""),
        "fonte_revisados": str(args.curated_jsonl or ""),
        "diretorio_prompts": str(args.prompts_dir.resolve()),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-data-dir", type=Path, default=ROOT_DIR / "data/evidencias-por-url-v1")
    parser.add_argument("--curated-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "data/perguntas-v1")
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--no-evidence-reading", action="store_const", const=None, dest="evidence_data_dir")
    parser.add_argument("--overwrite", action="store_true")
    print(json.dumps(preparar(parser.parse_args()), ensure_ascii=False, indent=2))
