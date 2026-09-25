#!/usr/bin/env python3
"""Normaliza e mescla QA públicos com um dataset preparado, sem alterar as fontes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from perguntas_utils import DEFAULT_PROMPTS_DIR, formatar_evidencias, montar_mensagens_pergunta


ROOT_DIR = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def destino_split(chave: str, seed: int) -> str:
    valor = int(hashlib.sha256(f"{seed}:{chave}".encode()).hexdigest()[:8], 16) % 100
    return "train" if valor < 80 else "valid" if valor < 90 else "test"


def conversa(pergunta: str, resposta: str, contexto: str, fonte: str, prompts_dir: Path) -> dict:
    evidencia = {"titulo": fonte, "url": fonte, "trecho": contexto.strip()[:5000]}
    mensagens = montar_mensagens_pergunta(
        pergunta.strip(), "evidencias", formatar_evidencias([evidencia]), 1, prompts_dir
    )
    saida = {
        "status": "RESPONDIDA",
        "resposta": f"{resposta.strip()} [1]",
        "justificativa": "A resposta está apoiada pelo texto fornecido na fonte [1].",
        "fontes_usadas": [1],
    }
    mensagens.append({"role": "assistant", "content": json.dumps(saida, ensure_ascii=False)})
    return {"messages": mensagens}


def iter_faquad(pasta: Path, prompts_dir: Path):
    for nome, split in (("train.json", "train"), ("valid.json", "valid")):
        caminho = pasta / nome
        if not caminho.exists():
            continue
        for artigo in json.loads(caminho.read_text(encoding="utf-8")).get("data", []):
            for paragrafo in artigo.get("paragraphs", []):
                contexto = paragrafo.get("context", "")
                grupo = hashlib.sha256(contexto.encode()).hexdigest()
                for qa in paragrafo.get("qas", []):
                    respostas = qa.get("answers", [])
                    if contexto and qa.get("question") and respostas and respostas[0].get("text"):
                        yield split, grupo, conversa(
                            qa["question"], respostas[0]["text"], contexto,
                            "https://github.com/liafacom/faquad", prompts_dir,
                        )


def iter_jsonl(caminho: Path):
    with caminho.open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            if linha.strip():
                yield json.loads(linha)


def iter_pira(pasta: Path, prompts_dir: Path):
    mapa = {"train": "train", "validation": "valid", "test": "test"}
    for origem, split in mapa.items():
        caminho = pasta / f"{origem}.jsonl"
        if not caminho.exists():
            continue
        for item in iter_jsonl(caminho):
            pergunta = item.get("question_pt_origin") or item.get("question_pt_paraphase")
            resposta = item.get("answer_pt_validate") or item.get("answer_pt_origin")
            contexto = item.get("abstract_translated_pt") or item.get("text_excerpts_un_reports") or item.get("abstract")
            if pergunta and resposta and contexto and item.get("answer_in_text", 1) != 0:
                yield split, str(item.get("id_qa", pergunta)), conversa(
                    pergunta, resposta, contexto,
                    "https://huggingface.co/datasets/paulopirozelli/pira", prompts_dir,
                )


def iter_rag(pasta: Path, prompts_dir: Path, seed: int):
    caminho = pasta / "train.jsonl"
    if not caminho.exists():
        return
    for item in iter_jsonl(caminho):
        pergunta, resposta = item.get("question"), item.get("answer")
        contexto = item.get("evidence_text") or item.get("context")
        grupo = str(item.get("document_id") or item.get("qa_id") or pergunta)
        if pergunta and resposta and contexto:
            yield destino_split(grupo, seed), grupo, conversa(
                pergunta, resposta, contexto,
                "https://huggingface.co/datasets/Madras1/rag-qa-fulltext-ptbr", prompts_dir,
            )


def main(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Destino não vazio: '{args.output_dir}'. Escolha um caminho novo.")
    saida = {split: [] for split in SPLITS}
    vistos: set[str] = set()
    contagens = Counter()

    for split in SPLITS:
        origem = args.base_data_dir / f"{split}.jsonl"
        if origem.exists():
            for item in iter_jsonl(origem):
                chave = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if chave not in vistos:
                    vistos.add(chave); saida[split].append(item); contagens[f"base_{split}"] += 1

    fontes = [
        ("faquad", iter_faquad(args.public_raw_dir / "faquad", args.prompts_dir)),
        ("pira", iter_pira(args.public_raw_dir / "pira", args.prompts_dir)),
        ("rag_ptbr", iter_rag(args.public_raw_dir / "rag_ptbr", args.prompts_dir, args.seed)),
    ]
    grupos_por_split = {split: set() for split in SPLITS}
    for nome, registros in fontes:
        for split, grupo, item in registros:
            chave = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if chave in vistos or any(grupo in grupos for outro, grupos in grupos_por_split.items() if outro != split):
                contagens[f"{nome}_descartados"] += 1
                continue
            vistos.add(chave); grupos_por_split[split].add(grupo)
            saida[split].append(item); contagens[f"{nome}_{split}"] += 1

    if not saida["train"] or not saida["valid"] or not saida["test"]:
        raise ValueError("A mesclagem precisa produzir train, valid e test não vazios.")
    args.output_dir.mkdir(parents=True)
    for split, itens in saida.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as arquivo:
            for item in itens:
                arquivo.write(json.dumps(item, ensure_ascii=False) + "\n")
    metadata = {
        "contagens": dict(contagens), "totais": {s: len(v) for s, v in saida.items()},
        "base_preservada": str(args.base_data_dir.resolve()),
        "fontes_publicas_preservadas": str(args.public_raw_dir.resolve()),
        "treino_executado": False,
        "limitacoes": [
            "RAG PT-BR contém anotações sintéticas; revise uma amostra antes de treinar.",
            "A justificativa indica apoio textual e não prova a veracidade externa da fonte.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, default=ROOT_DIR / "data/perguntas-v2")
    parser.add_argument("--public-raw-dir", type=Path, default=ROOT_DIR / "raw/publicos-perguntas-v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "data/perguntas-mescladas-v1")
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
