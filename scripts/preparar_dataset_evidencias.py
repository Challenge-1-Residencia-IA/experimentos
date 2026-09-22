#!/usr/bin/env python3
"""Cria dataset balanceado de verificação com evidências a partir do FACTCK.BR."""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from evidencias_utils import PROMPT_SISTEMA_EVIDENCIAS, montar_mensagens_evidencias


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT_DIR / "raw/factckbr/FACTCKBR.tsv"
DEFAULT_OUTPUT = ROOT_DIR / "data/evidencias"

PROMPT_SISTEMA = PROMPT_SISTEMA_EVIDENCIAS

MAPA_ROTULOS = {
    "falso": "FALSA",
    "exagerado": "FALSA",
    "distorcido": "FALSA",
    "sem contexto": "FORA_DE_CONTEXTO",
    "insustentavel": "FALSA",
    "impreciso": "FALSA",
    "verdadeiro": "VERDADEIRA",
    "verdadeiro, mas": "VERDADEIRA",
    "impossivel provar": "INCONCLUSIVA",
    "discutivel": "INCONCLUSIVA",
    "ainda e cedo para dizer": "INCONCLUSIVA",
    "de olho": "INCONCLUSIVA",
}


def normalizar(texto: str) -> str:
    import unicodedata

    texto = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", texto).strip().lower()


def limitar(texto: str, tamanho: int) -> str:
    texto = re.sub(r"\s+", " ", texto or "").strip()
    if len(texto) <= tamanho:
        return texto
    corte = texto[:tamanho].rsplit(" ", 1)[0]
    return f"{corte}…"


PADROES_VAZAMENTO = (
    r"\b(?:esta|está|e|é|foi)\s+(?:informacao\s+|informação\s+)?(?:e\s+|é\s+)?fals[ao]\b",
    r"\b(?:esta|está|e|é|foi)\s+(?:informacao\s+|informação\s+)?verdadeir[ao]\b",
    r"\b(?:classificad[ao]|rotulad[ao])\s+como\s+(?:fals[ao]|verdadeir[ao])\b",
)


def reduzir_vazamento_rotulo(texto: str) -> str:
    """Remove conclusoes explicitas que entregam o rotulo sem apagar os fatos."""
    saida = texto or ""
    for padrao in PADROES_VAZAMENTO:
        saida = re.sub(padrao, "", saida, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", saida).strip()


def carregar_fonte(
    caminho: Path,
    max_claim_chars: int,
    max_evidence_chars: int,
    min_evidence_chars: int,
) -> list[dict]:
    vistos, registros = set(), []
    with caminho.open(encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo, delimiter="\t"):
            claim = limitar(linha.get("claimReviewed", ""), max_claim_chars)
            evidencia = limitar(reduzir_vazamento_rotulo(linha.get("reviewBody", "")), max_evidence_chars)
            url = linha.get("URL", "").strip()
            rotulo_original = normalizar(linha.get("alternativeName", ""))
            rotulo = MAPA_ROTULOS.get(rotulo_original)
            chave = normalizar(claim)
            palavras_claim = set(normalizar(claim).split())
            palavras_evidencia = set(normalizar(evidencia).split())
            sobreposicao = (
                len(palavras_claim & palavras_evidencia) / len(palavras_claim)
                if palavras_claim else 1.0
            )
            evidencia_repete_claim = sobreposicao > 0.8 and len(evidencia) < max(250, len(claim) * 2)
            if (
                not rotulo
                or not claim
                or len(evidencia) < min_evidence_chars
                or evidencia_repete_claim
                or not url
                or chave in vistos
            ):
                continue
            vistos.add(chave)
            registros.append({
                "id": f"factckbr-{len(registros) + 1:04d}",
                "claim": claim,
                "evidence": evidencia,
                "url": url,
                "publisher": linha.get("Author", "").strip(),
                "date": linha.get("datePublished", "").strip(),
                "label": rotulo,
                "original_label": linha.get("alternativeName", "").strip(),
                "synthetic": False,
            })
    return registros


def separar(registros: list[dict], seed: int) -> dict[str, list[dict]]:
    por_rotulo = defaultdict(list)
    for registro in registros:
        por_rotulo[registro["label"]].append(registro)
    rng = random.Random(seed)
    splits = {"train": [], "valid": [], "test": []}
    for itens in por_rotulo.values():
        rng.shuffle(itens)
        n = len(itens)
        n_test = max(1, round(n * 0.15))
        n_valid = max(1, round(n * 0.15))
        splits["test"].extend(itens[:n_test])
        splits["valid"].extend(itens[n_test : n_test + n_valid])
        splits["train"].extend(itens[n_test + n_valid :])
    return splits


def adicionar_inconclusivas_sem_evidencia(registros: list[dict], seed: int, quantidade: int) -> list[dict]:
    """Cria contrafactuais apenas no treino para ensinar abstencao sem fontes."""
    rng = random.Random(seed)
    saida = list(registros)
    base_sintetica = [r for r in registros if r["label"] != "INCONCLUSIVA"]
    rng.shuffle(base_sintetica)
    for origem in base_sintetica[:quantidade]:
        saida.append({
            **origem,
            "id": f"{origem['id']}-sem-evidencia",
            "evidence": "Nenhuma evidência verificável foi fornecida para esta alegação.",
            "url": "",
            "publisher": "",
            "label": "INCONCLUSIVA",
            "original_label": "sem evidência suficiente",
            "synthetic": True,
        })
    rng.shuffle(saida)
    return saida


def balancear_treino(
    registros: list[dict], seed: int, estrategia: str, tamanho_alvo: int | None = None
) -> list[dict]:
    """Balanceia somente treino; nunca remove exemplos de validacao ou teste."""
    if estrategia == "none":
        return list(registros)
    rng = random.Random(seed)
    por_rotulo: dict[str, list[dict]] = defaultdict(list)
    for registro in registros:
        por_rotulo[registro["label"]].append(registro)
    tamanhos = [len(itens) for itens in por_rotulo.values() if itens]
    if not tamanhos:
        return []
    alvo = (tamanho_alvo or max(tamanhos)) if estrategia == "oversample" else min(tamanhos)
    saida = []
    for itens in por_rotulo.values():
        if len(itens) >= alvo:
            saida.extend(rng.sample(itens, alvo))
        else:
            saida.extend(itens)
            saida.extend(rng.choice(itens) for _ in range(alvo - len(itens)))
    rng.shuffle(saida)
    return saida


def mensagens(registro: dict, max_justification_chars: int) -> dict:
    evidencia = f"[1] Fonte: {registro['publisher']}\nURL: {registro['url']}\nTrecho: {registro['evidence']}"
    if registro["synthetic"]:
        evidencia = registro["evidence"]
        justificativa = "Não há evidências suficientes para confirmar ou refutar a alegação."
        fontes = []
    else:
        justificativa = limitar(registro["evidence"], max_justification_chars)
        fontes = [1]
    resposta = {
        "veredito": registro["label"],
        "justificativa": justificativa,
        "fontes_usadas": fontes,
    }
    mensagens_base = montar_mensagens_evidencias(registro["claim"], evidencia)
    return {"messages": mensagens_base + [
        {"role": "assistant", "content": json.dumps(resposta, ensure_ascii=False)},
    ]}


def salvar_jsonl(registros: list[dict], caminho: Path, max_justification_chars: int) -> None:
    with caminho.open("w", encoding="utf-8") as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(mensagens(registro, max_justification_chars), ensure_ascii=False) + "\n")


def main(args: argparse.Namespace) -> None:
    if not args.source.exists():
        raise FileNotFoundError(f"Fonte não encontrada: '{args.source}'.")
    registros = carregar_fonte(
        args.source,
        args.max_claim_chars,
        args.max_evidence_chars,
        args.min_evidence_chars,
    )
    splits = separar(registros, args.seed)
    if args.synthetic_inconclusive > 0:
        splits["train"] = adicionar_inconclusivas_sem_evidencia(
            splits["train"], args.seed + 10, args.synthetic_inconclusive
        )
    splits["train"] = balancear_treino(
        splits["train"], args.seed + 20, args.balance_strategy, args.balance_target
    )
    for indice, nome in enumerate(splits):
        random.Random(args.seed + indice).shuffle(splits[nome])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for nome, itens in splits.items():
        salvar_jsonl(itens, args.output_dir / f"{nome}.jsonl", args.max_justification_chars)
        with (args.output_dir / f"{nome}_audit.jsonl").open("w", encoding="utf-8") as arquivo:
            for item in itens:
                arquivo.write(json.dumps(item, ensure_ascii=False) + "\n")
    metadados = {
        "source": "FACTCK.BR",
        "source_url": "https://github.com/jghm-f/FACTCK.BR",
        "license": "MIT",
        "seed": args.seed,
        "source_records_after_quality_filter": len(registros),
        "labels": ["FALSA", "VERDADEIRA", "FORA_DE_CONTEXTO", "INCONCLUSIVA"],
        "balance_strategy_train_only": args.balance_strategy,
        "balance_target_per_class": args.balance_target,
        "splits": {
            nome: {
                "total": len(itens),
                "labels": dict(Counter(i["label"] for i in itens)),
                "synthetic_inconclusive": sum(i["synthetic"] for i in itens),
            }
            for nome, itens in splits.items()
        },
        "limitations": [
            "A evidência é o texto da checagem jornalística, não necessariamente uma fonte primária.",
            "Casos INCONCLUSIVA derivados ensinam abstinência quando nenhuma evidência é fornecida.",
            "Os rótulos originais foram normalizados para quatro classes.",
            "Apenas o treino pode conter oversampling e casos inconclusivos sinteticos.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadados["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset de verificação com evidências para MLX.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-claim-chars", type=int, default=500)
    parser.add_argument("--max-evidence-chars", type=int, default=700)
    parser.add_argument("--min-evidence-chars", type=int, default=80)
    parser.add_argument("--max-justification-chars", type=int, default=320)
    parser.add_argument(
        "--balance-strategy",
        choices=("none", "oversample", "undersample"),
        default="oversample",
        help="Balanceamento aplicado somente ao treino; oversample preserva todos os registros.",
    )
    parser.add_argument(
        "--synthetic-inconclusive",
        type=int,
        default=30,
        help="Casos sem evidencia adicionados somente ao treino.",
    )
    parser.add_argument(
        "--balance-target",
        type=int,
        default=150,
        help="Quantidade alvo por classe no treino balanceado.",
    )
    args = parser.parse_args()
    if args.synthetic_inconclusive < 0 or args.balance_target <= 0:
        parser.error("--synthetic-inconclusive e --balance-target devem ser validos.")
    main(args)
