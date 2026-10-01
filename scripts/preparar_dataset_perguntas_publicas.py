#!/usr/bin/env python3
"""Normaliza e mescla QA públicos, com distratores e casos inconclusivos."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from perguntas_utils import DEFAULT_PROMPTS_DIR, formatar_evidencias, montar_mensagens_pergunta

ROOT_DIR = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def destino_split(chave: str, seed: int) -> str:
    valor = int(hashlib.sha256(f"{seed}:{chave}".encode()).hexdigest()[:8], 16) % 100
    return "train" if valor < 80 else "valid" if valor < 90 else "test"


def _palavras(texto: str) -> set[str]:
    return {p for p in re.findall(r"[\wÀ-ÿ]+", texto.casefold()) if len(p) > 2}


def trecho_que_sustenta(resposta: str, contexto: str, limite: int = 420) -> str:
    """Seleciona uma frase observável para evitar justificativas genéricas."""
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+|\n+", contexto) if f.strip()]
    termos = _palavras(resposta)
    melhor = max(frases, key=lambda f: len(termos & _palavras(f)), default=contexto.strip())
    return melhor[:limite].rstrip()


def janela_relevante(resposta: str, contexto: str, limite: int = 2200) -> str:
    """Recorta o contexto ao redor da resposta sem encher o prompt com o documento inteiro."""
    contexto = contexto.strip()
    if len(contexto) <= limite:
        return contexto
    posicao = contexto.casefold().find(resposta.strip().casefold())
    if posicao < 0:
        frase = trecho_que_sustenta(resposta, contexto, limite)
        posicao = contexto.find(frase)
    metade = limite // 2
    inicio = max(0, posicao - metade)
    fim = min(len(contexto), inicio + limite)
    inicio = max(0, fim - limite)
    recorte = contexto[inicio:fim].strip()
    return ("… " if inicio else "") + recorte + (" …" if fim < len(contexto) else "")


def conversa(pergunta: str, resposta: str, contexto: str, fonte: str, prompts_dir: Path,
             distrator: dict[str, str] | None = None) -> dict:
    contexto_relevante = janela_relevante(resposta, contexto)
    evidencias = [{"titulo": fonte, "url": fonte, "trecho": contexto_relevante}]
    if distrator:
        evidencias.append(distrator)
    mensagens = montar_mensagens_pergunta(
        pergunta.strip(), "evidencias", formatar_evidencias(evidencias), 1, prompts_dir
    )
    apoio = trecho_que_sustenta(resposta, contexto_relevante)
    saida = {
        "status": "RESPONDIDA",
        "resposta": f"{resposta.strip()} [1]",
        "justificativa": f'A fonte [1] informa: "{apoio}". Esse trecho sustenta diretamente a resposta.',
        "fontes_usadas": [1],
    }
    mensagens.append({"role": "assistant", "content": json.dumps(saida, ensure_ascii=False)})
    return {"messages": mensagens}


def conversa_inconclusiva(pergunta: str, evidencias: list[dict[str, str]], prompts_dir: Path) -> dict:
    mensagens = montar_mensagens_pergunta(
        pergunta.strip(), "evidencias", formatar_evidencias(evidencias), 1, prompts_dir
    )
    saida = {
        "status": "INCONCLUSIVA",
        "resposta": "As evidências fornecidas não permitem responder à pergunta com segurança.",
        "justificativa": "Os trechos disponíveis não apresentam a informação solicitada; por isso nenhuma fonte foi usada como apoio.",
        "fontes_usadas": [],
    }
    mensagens.append({"role": "assistant", "content": json.dumps(saida, ensure_ascii=False)})
    return {"messages": mensagens}


def iter_jsonl(caminho: Path):
    with caminho.open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            if linha.strip():
                yield json.loads(linha)


def iter_faquad(pasta: Path):
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
                        yield split, grupo, qa["question"], respostas[0]["text"], contexto


def iter_pira(pasta: Path):
    for origem, split in {"train": "train", "validation": "valid", "test": "test"}.items():
        caminho = pasta / f"{origem}.jsonl"
        if not caminho.exists():
            continue
        for item in iter_jsonl(caminho):
            pergunta = item.get("question_pt_origin") or item.get("question_pt_paraphase")
            resposta = item.get("answer_pt_validate") or item.get("answer_pt_origin")
            contexto = item.get("abstract_translated_pt") or item.get("text_excerpts_un_reports") or item.get("abstract")
            if pergunta and resposta and contexto and item.get("answer_in_text", 1) != 0:
                yield split, str(item.get("id_qa", pergunta)), pergunta, resposta, contexto


def iter_rag(pasta: Path, seed: int):
    caminho = pasta / "train.jsonl"
    if not caminho.exists():
        return
    for item in iter_jsonl(caminho):
        pergunta, resposta = item.get("question"), item.get("answer")
        contexto = item.get("evidence_text") or item.get("context")
        grupo = str(item.get("document_id") or item.get("qa_id") or pergunta)
        if pergunta and resposta and contexto:
            yield destino_split(grupo, seed), grupo, pergunta, resposta, contexto


def _ordenar_e_limitar(registros: list[dict], limite: int, seed: int) -> list[dict]:
    ordenados = sorted(registros, key=lambda r: hashlib.sha256(
        f"{seed}:{r['origem']}:{r['grupo']}:{r['pergunta']}".encode()).hexdigest())
    return ordenados[:limite] if limite > 0 else ordenados


def _contar_contrato(item: dict, contagens: Counter, prefixo: str) -> None:
    try:
        resposta = json.loads(item["messages"][-1]["content"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return
    contagens[f"{prefixo}_{str(resposta.get('status', 'desconhecido')).lower()}"] += 1
    if len(resposta.get("fontes_usadas", [])) >= 2:
        contagens[f"{prefixo}_multifonte"] += 1


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
                    vistos.add(chave)
                    saida[split].append(item)
                    contagens[f"base_{split}"] += 1
                    _contar_contrato(item, contagens, "base")

    especificacoes = [
        ("faquad", "https://github.com/liafacom/faquad", list(iter_faquad(args.public_raw_dir / "faquad"))),
        ("pira", "https://huggingface.co/datasets/paulopirozelli/pira", list(iter_pira(args.public_raw_dir / "pira"))),
        ("rag_ptbr", "https://huggingface.co/datasets/Madras1/rag-qa-fulltext-ptbr", list(iter_rag(args.public_raw_dir / "rag_ptbr", args.seed))),
    ]
    registros = []
    for nome, url, brutos in especificacoes:
        normalizados = [
            {"origem": nome, "url": url, "split": split, "grupo": grupo, "pergunta": pergunta,
             "resposta": resposta, "contexto": contexto}
            for split, grupo, pergunta, resposta, contexto in brutos
        ]
        selecionados = _ordenar_e_limitar(normalizados, args.max_por_origem, args.seed)
        registros.extend(selecionados)
        contagens[f"{nome}_limitados"] += len(normalizados) - len(selecionados)

    por_split = {split: [r for r in registros if r["split"] == split] for split in SPLITS}
    for split, registros_split in por_split.items():
        for indice, registro in enumerate(registros_split):
            candidatos = [r for r in registros_split if r["grupo"] != registro["grupo"]]
            outro = candidatos[(indice * 7919 + args.seed) % len(candidatos)] if candidatos else None
            distrator = None if not outro else {
                "titulo": f"Distrator — {outro['origem']}", "url": outro["url"], "trecho": outro["contexto"][:700]
            }
            item = conversa(registro["pergunta"], registro["resposta"], registro["contexto"],
                            registro["url"], args.prompts_dir, distrator)
            chave = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if chave not in vistos:
                vistos.add(chave)
                saida[split].append(item)
                contagens[f"{registro['origem']}_{split}"] += 1
                contagens["respondidas_com_distrator"] += int(distrator is not None)
            marca = int(hashlib.sha256(
                f"inconclusiva:{args.seed}:{registro['grupo']}:{registro['pergunta']}".encode()
            ).hexdigest()[:8], 16)
            if distrator and marca / 0xFFFFFFFF < args.fracao_inconclusiva:
                inc = conversa_inconclusiva(registro["pergunta"], [distrator], args.prompts_dir)
                chave_inc = hashlib.sha256(json.dumps(inc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if chave_inc not in vistos:
                    vistos.add(chave_inc)
                    saida[split].append(inc)
                    contagens[f"inconclusivas_{split}"] += 1

    if not saida["train"] or not saida["valid"] or not saida["test"]:
        raise ValueError("A mesclagem precisa produzir train, valid e test não vazios.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, itens in saida.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as arquivo:
            for item in itens:
                arquivo.write(json.dumps(item, ensure_ascii=False) + "\n")
    metadata = {
        "contagens": dict(contagens), "totais": {s: len(v) for s, v in saida.items()},
        "configuracao": {"max_por_origem": args.max_por_origem, "fracao_inconclusiva": args.fracao_inconclusiva, "seed": args.seed},
        "base_preservada": str(args.base_data_dir.resolve()),
        "fontes_publicas_preservadas": str(args.public_raw_dir.resolve()), "treino_executado": False,
        "limitacoes": [
            "Os exemplos públicos têm uma fonte de apoio; uma segunda evidência, quando presente, é distratora e não é citada.",
            "Exemplos realmente multifonte devem vir do JSONL revisado e são contabilizados no metadata.",
            "RAG PT-BR contém anotações sintéticas; revise uma amostra antes de treinar.",
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
    parser.add_argument("--max-por-origem", type=int, default=2500, help="Limite por dataset público; 0 desativa o limite.")
    parser.add_argument("--fracao-inconclusiva", type=float, default=0.12)
    args = parser.parse_args()
    if args.max_por_origem < 0 or not 0 <= args.fracao_inconclusiva <= 1:
        parser.error("--max-por-origem deve ser >= 0 e --fracao-inconclusiva deve estar entre 0 e 1.")
    main(args)
