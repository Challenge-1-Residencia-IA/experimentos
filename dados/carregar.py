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
CAMPOS_NUMERICOS_FAKEBR = CAMPOS_META_FAKEBR[4:]


def ler_textos(pasta: Path) -> dict[int, str]:
    """Textos de uma pasta do Fake.br-Corpus, indexados pelo id e em ordem crescente de id."""
    arquivos = sorted(pasta.glob("*.txt"), key=lambda p: int(p.stem))
    return {int(p.stem): p.read_text(encoding="utf-8", errors="ignore") for p in arquivos}


def ler_meta(arquivo: Path) -> dict[str, str]:
    linhas = [l.strip() for l in arquivo.read_text(encoding="utf-8", errors="ignore").splitlines()]
    return dict(zip(CAMPOS_META_FAKEBR, linhas))


def truncar_pares(textos_a: pd.Series, textos_b: pd.Series) -> tuple[list[str], list[str]]:
    """Corta os dois textos de cada par para o número de palavras do mais curto, mantendo o início.

    É a regra da versão `size_normalized_texts` do Fake.br-Corpus.
    """
    palavras_a, palavras_b = textos_a.str.split(), textos_b.str.split()
    tamanhos = [min(len(a), len(b)) for a, b in zip(palavras_a, palavras_b)]
    return ([" ".join(p[:n]) for p, n in zip(palavras_a, tamanhos)],
            [" ".join(p[:n]) for p, n in zip(palavras_b, tamanhos)])


def carregar_fakebr(base: Path = FAKEBR, incluir_metricas: bool = False) -> pd.DataFrame:
    """Uma linha por texto de `full_texts` (fake e depois true, em ordem de id).

    `texto_normalizado` vem de `size_normalized_texts` e fica vazio nos ids que só existem em
    `full_texts` (`id_inconsistente`). Com `incluir_metricas`, traz também as métricas de estilo
    dos metadados e a data bruta.
    """
    linhas = []
    for classe in ["fake", "true"]:
        textos = ler_textos(base / "full_texts" / classe)
        normalizados = ler_textos(base / "size_normalized_texts" / classe)
        pasta_meta = base / "full_texts" / f"{classe}-meta-information"
        for id_, texto in textos.items():
            arquivo_meta = pasta_meta / f"{id_}-meta.txt"
            meta = ler_meta(arquivo_meta) if arquivo_meta.exists() else {}
            linha = {
                "dataset": "fakebr",
                "id_original": str(id_),
                "classe": classe,
                "grupo": f"fakebr-{id_}",
                "titulo": None,
                "texto_original": texto,
                "texto_normalizado": normalizados.get(id_),
                "link": meta.get("link"),
                "categoria": meta.get("categoria"),
                "data": parse_data(meta.get("data_bruta")),
                "id_inconsistente": id_ not in normalizados,
            }
            if incluir_metricas:
                linha |= {campo: meta.get(campo) for campo in ["autor", "data_bruta", *CAMPOS_NUMERICOS_FAKEBR]}
            linhas.append(linha)
    df = pd.DataFrame(linhas)
    if incluir_metricas:
        df[CAMPOS_NUMERICOS_FAKEBR] = df[CAMPOS_NUMERICOS_FAKEBR].apply(pd.to_numeric, errors="coerce")
    return df


def carregar_faketrue(csv: Path = FAKETRUE_CSV) -> pd.DataFrame:
    """Uma linha por texto (fakes e depois trues, na ordem do CSV).

    O grupo é o link do texto verdadeiro, que se repete em vários pares. `texto_normalizado`
    reconstrói a versão normalizada por tamanho com a mesma regra do Fake.br-Corpus.
    """
    bruto = pd.read_csv(csv)
    fake_truncado, true_truncado = truncar_pares(bruto["fake"], bruto["true"])
    comum = {"dataset": "faketrue", "categoria": None, "data": pd.NaT, "id_inconsistente": False}
    fake = pd.DataFrame({
        **comum, "id_original": bruto.index.astype(str), "classe": "fake", "grupo": bruto["link_t"],
        "titulo": bruto["title_fake"].str.strip(), "texto_original": bruto["fake"].str.strip(),
        "texto_normalizado": fake_truncado, "link": bruto["link_f"],
    })
    true = pd.DataFrame({
        **comum, "id_original": bruto.index.astype(str), "classe": "true", "grupo": bruto["link_t"],
        "titulo": None, "texto_original": bruto["true"].str.strip(),
        "texto_normalizado": true_truncado, "link": bruto["link_t"],
    })
    return pd.concat([fake, true], ignore_index=True)
