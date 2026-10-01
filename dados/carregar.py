from pathlib import Path

import pandas as pd

from dados.caminhos import FAKEBR, FAKETRUE_CSV
from dados.texto import parse_data

# Ordem das 25 linhas dos arquivos *-meta.txt, conforme o README do Fake.br-Corpus.
CAMPOS_META_FAKEBR = [
    "autor", "link", "categoria", "data_bruta",
    "n_tokens", "n_palavras", "n_types", "n_links", "n_maiusculas",
    "n_verbos", "n_verb_subj_imp", "n_substantivos", "n_adjetivos",
    "n_adverbios", "n_verb_modais", "n_pron_1_2_sing", "n_pron_1_plural",
    "n_pronomes", "pausality", "n_caracteres", "comp_medio_sentenca",
    "comp_medio_palavra", "pct_erros_ortograficos", "emotiveness", "diversity",
]


def _ler_textos(pasta: Path) -> dict[int, str]:
    return {int(p.stem): p.read_text(encoding="utf-8", errors="ignore") for p in pasta.glob("*.txt")}


def carregar_fakebr(base: Path = FAKEBR) -> pd.DataFrame:
    """Uma linha por texto de `full_texts`, com os metadados de link, categoria e data."""
    linhas = []
    for classe in ["fake", "true"]:
        textos = _ler_textos(base / "full_texts" / classe)
        normalizados = set(_ler_textos(base / "size_normalized_texts" / classe))
        for arquivo in (base / "full_texts" / f"{classe}-meta-information").glob("*-meta.txt"):
            id_ = int(arquivo.stem.split("-")[0])
            if id_ not in textos:
                continue
            meta = dict(zip(CAMPOS_META_FAKEBR,
                            [l.strip() for l in arquivo.read_text(encoding="utf-8", errors="ignore").splitlines()]))
            linhas.append({
                "dataset": "fakebr",
                "id_original": str(id_),
                "classe": classe,
                "grupo": f"fakebr-{id_}",
                "titulo": None,
                "texto_original": textos[id_],
                "link": meta.get("link"),
                "categoria": meta.get("categoria"),
                "data": parse_data(meta.get("data_bruta")),
                "id_inconsistente": id_ not in normalizados,
            })
    return pd.DataFrame(linhas)


def carregar_faketrue(csv: Path = FAKETRUE_CSV) -> pd.DataFrame:
    """Uma linha por texto. O grupo é o link do texto verdadeiro, que se repete em vários pares."""
    bruto = pd.read_csv(csv)
    comum = {"dataset": "faketrue", "categoria": None, "data": pd.NaT, "id_inconsistente": False}
    fake = pd.DataFrame({
        **comum, "id_original": bruto.index.astype(str), "classe": "fake", "grupo": bruto["link_t"],
        "titulo": bruto["title_fake"].str.strip(), "texto_original": bruto["fake"].str.strip(),
        "link": bruto["link_f"],
    })
    true = pd.DataFrame({
        **comum, "id_original": bruto.index.astype(str), "classe": "true", "grupo": bruto["link_t"],
        "titulo": None, "texto_original": bruto["true"].str.strip(), "link": bruto["link_t"],
    })
    return pd.concat([fake, true], ignore_index=True)
