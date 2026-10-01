"""Prepara os datasets Fake.br-Corpus e FakeTrue.Br.

Gera uma camada genérica em data/interim (os dois datasets num esquema único, com sinalizações e
sem descartar nada) e visões por finalidade em data/processed:

- base_checagens: textos verdadeiros do FakeTrue.Br que formam a base de evidências do RAG,
  exceto as checagens mantidas fora para testar o caso "sem evidência" (IA-07);
- avaliacao_busca: cada boato do FakeTrue.Br com o link da checagem esperada (IA-03, IA-07);
- golden_set_candidatos: amostra para anotação do golden set v1;
- fewshot_candidatos: textos de desinformação do Fake.br disponíveis para exemplos de prompt.

Uso: python -m dados.preparar
"""

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from dados.caminhos import INTERIM, PROCESSED, REPORTS
from dados.carregar import carregar_fakebr, carregar_faketrue
from dados.texto import detectar_boilerplate, harmonizar, remover_boilerplate, tem_acento

SEED = 42
LIMIAR_BOILERPLATE = 20
LIMIAR_QUASE_DUPLICATA = 0.9
PROPORCAO_CHECAGENS_FORA_DA_BASE = 0.2
GOLDEN_MENSAGENS = 80
GOLDEN_ARTIGOS = 20
MIN_PALAVRAS = 10

MARCAS_CHECAGEM = r"\bcircula|fato ou fake|é fake|é \. ,|\bchecagem"


def _quase_duplicatas(df: pd.DataFrame) -> pd.Series:
    """Marca textos de um dataset com similaridade de cosseno >= limiar a algum texto do outro."""
    vetores = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(df["texto_harmonizado"])
    e_fakebr = (df["dataset"] == "fakebr").to_numpy()
    idx_fakebr, idx_faketrue = np.flatnonzero(e_fakebr), np.flatnonzero(~e_fakebr)
    vizinhos = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(vetores[idx_fakebr])
    distancia, indice = vizinhos.kneighbors(vetores[idx_faketrue])
    similares = (1 - distancia.ravel()) >= LIMIAR_QUASE_DUPLICATA

    marcado = np.zeros(len(df), dtype=bool)
    marcado[idx_faketrue[similares]] = True
    marcado[idx_fakebr[indice.ravel()[similares]]] = True
    return pd.Series(marcado, index=df.index)


def montar_camada_generica() -> tuple[pd.DataFrame, list[str]]:
    df = pd.concat([carregar_fakebr(), carregar_faketrue()], ignore_index=True)

    boilerplate = detectar_boilerplate(df["texto_original"], LIMIAR_BOILERPLATE)
    df["texto_limpo"] = df["texto_original"].map(lambda t: remover_boilerplate(t, boilerplate))
    df["texto_harmonizado"] = df["texto_limpo"].map(harmonizar)

    df["dominio"] = df["link"].str.extract(r"https?://(?:www\d?\.)?([^/]+)")[0]
    ano_url = df["link"].str.extract(r"/(20\d{2})/")[0].astype(float)
    df["ano"] = df["data"].dt.year.fillna(ano_url).astype("Int64")
    df["n_palavras"] = df["texto_limpo"].str.split().str.len()

    texto_minusculo = df["texto_original"].str.lower()
    e_checagem = (df["dataset"] == "faketrue") & (
        df["link"].str.contains("fato-ou-fake", na=False) | texto_minusculo.str.contains(MARCAS_CHECAGEM, regex=True)
    )
    df["natureza"] = np.select(
        [df["classe"] == "fake", (df["classe"] == "true") & e_checagem],
        ["desinformacao", "checagem"],
        default="noticia",
    )

    df["sem_acento"] = ~df["texto_original"].map(tem_acento)
    # O texto verdadeiro do FakeTrue.Br se repete em todos os pares da mesma checagem por construção;
    # nesse caso só conta como duplicata quando aparece em mais de uma checagem.
    chave = df.groupby(["dataset", "classe", "texto_harmonizado"])
    reutilizacao_esperada = (df["dataset"] == "faketrue") & (df["classe"] == "true")
    df["duplicado_interno"] = np.where(reutilizacao_esperada,
                                       chave["grupo"].transform("nunique") > 1,
                                       chave["grupo"].transform("size") > 1)
    df["quase_duplicata_outro_dataset"] = _quase_duplicatas(df)

    colunas = [
        "dataset", "id_original", "classe", "grupo", "natureza", "titulo",
        "texto_original", "texto_limpo", "texto_harmonizado",
        "link", "dominio", "categoria", "data", "ano", "n_palavras",
        "id_inconsistente", "sem_acento", "duplicado_interno", "quase_duplicata_outro_dataset",
    ]
    return df[colunas], boilerplate


def montar_visoes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    faketrue = df[df["dataset"] == "faketrue"]
    fakebr = df[df["dataset"] == "fakebr"]

    checagens = np.array(sorted(faketrue["grupo"].unique()))
    fora_da_base = set(rng.choice(checagens, size=round(len(checagens) * PROPORCAO_CHECAGENS_FORA_DA_BASE),
                                  replace=False))

    base = (faketrue[(faketrue["classe"] == "true") & ~faketrue["grupo"].isin(fora_da_base)]
            .drop_duplicates("grupo")
            [["grupo", "natureza", "texto_limpo", "link", "dominio", "ano"]]
            .rename(columns={"grupo": "id_checagem"}))

    avaliacao = (faketrue[faketrue["classe"] == "fake"]
                 [["id_original", "titulo", "texto_limpo", "link", "grupo"]]
                 .rename(columns={"link": "link_boato", "grupo": "link_checagem_esperada"}))
    avaliacao["checagem_na_base"] = ~avaliacao["link_checagem_esperada"].isin(fora_da_base)

    candidatas_mensagem = faketrue[
        (faketrue["classe"] == "fake")
        & ~faketrue["quase_duplicata_outro_dataset"]
        & (faketrue["n_palavras"] >= MIN_PALAVRAS)
    ].groupby("grupo").sample(1, random_state=SEED)
    mensagens = candidatas_mensagem.sample(GOLDEN_MENSAGENS, random_state=SEED).assign(
        tipo_entrada="mensagem", checagem_na_base=lambda d: ~d["grupo"].isin(fora_da_base))

    candidatos_artigo = fakebr[
        (fakebr["classe"] == "fake")
        & ~fakebr["quase_duplicata_outro_dataset"]
        & ~fakebr["id_inconsistente"]
        & (fakebr["n_palavras"] >= MIN_PALAVRAS)
    ]
    artigos = candidatos_artigo.sample(GOLDEN_ARTIGOS, random_state=SEED).assign(
        tipo_entrada="artigo", checagem_na_base=False)

    golden = pd.concat([mensagens, artigos], ignore_index=True)[
        ["dataset", "id_original", "grupo", "tipo_entrada", "texto_limpo", "link", "checagem_na_base"]]

    fewshot = candidatos_artigo[~candidatos_artigo["grupo"].isin(golden["grupo"])][
        ["id_original", "grupo", "categoria", "texto_limpo", "link", "ano"]]

    return {
        "base_checagens": base.reset_index(drop=True),
        "avaliacao_busca": avaliacao.reset_index(drop=True),
        "golden_set_candidatos": golden,
        "fewshot_candidatos": fewshot.reset_index(drop=True),
    }


def escrever_relatorio(df: pd.DataFrame, boilerplate: list[str], visoes: dict[str, pd.DataFrame]) -> str:
    contagem = df.groupby(["dataset", "classe"]).size().unstack()
    sinais = df.groupby("dataset")[
        ["id_inconsistente", "sem_acento", "duplicado_interno", "quase_duplicata_outro_dataset"]].sum()
    natureza = df.groupby(["dataset", "natureza"]).size().unstack(fill_value=0)
    golden = visoes["golden_set_candidatos"]
    avaliacao = visoes["avaliacao_busca"]
    linhas = [
        "# Preparação dos dados",
        "",
        "Gerado por `python -m dados.preparar` (`make data`). Os arquivos de dados ficam em `data/` e não são versionados.",
        "",
        "## Parâmetros",
        "",
        "| Parâmetro | Valor |",
        "|---|---|",
        f"| Semente aleatória | {SEED} |",
        f"| Limiar de boilerplate (textos com a frase) | {LIMIAR_BOILERPLATE} |",
        f"| Limiar de quase-duplicata (similaridade de cosseno) | {LIMIAR_QUASE_DUPLICATA} |",
        f"| Checagens mantidas fora da base | {PROPORCAO_CHECAGENS_FORA_DA_BASE:.0%} |",
        f"| Golden set v1: mensagens / artigos | {GOLDEN_MENSAGENS} / {GOLDEN_ARTIGOS} |",
        f"| Mínimo de palavras para golden set e few-shot | {MIN_PALAVRAS} |",
        "",
        "## Camada genérica (`data/interim/textos.parquet`)",
        "",
        "Textos por dataset e classe:",
        "",
        contagem.to_markdown(),
        "",
        "Natureza dos textos:",
        "",
        natureza.to_markdown(),
        "",
        "Sinalizações (textos marcados, nada é descartado nesta camada):",
        "",
        sinais.to_markdown(),
        "",
        f"Variações de boilerplate removidas em `texto_limpo`: {len(boilerplate)}, além dos padrões de rodapé.",
        "",
        "## Visões (`data/processed/`)",
        "",
        "| Arquivo | Linhas | Conteúdo |",
        "|---|---|---|",
        f"| `base_checagens.parquet` | {len(visoes['base_checagens'])} | Textos verdadeiros distintos do FakeTrue.Br que formam a base de evidências |",
        f"| `avaliacao_busca.parquet` | {len(avaliacao)} | Boatos do FakeTrue.Br com a checagem esperada; {int(avaliacao['checagem_na_base'].sum())} com a checagem na base e {int((~avaliacao['checagem_na_base']).sum())} sem |",
        f"| `golden_set_candidatos.parquet` | {len(golden)} | {int((golden['tipo_entrada'] == 'mensagem').sum())} mensagens (uma por checagem, {int(golden['checagem_na_base'].sum())} com checagem na base) e {int((golden['tipo_entrada'] == 'artigo').sum())} artigos, para anotação |",
        f"| `fewshot_candidatos.parquet` | {len(visoes['fewshot_candidatos'])} | Textos de desinformação do Fake.br fora do golden set, para exemplos de prompt |",
        "",
        "Regras aplicadas:",
        "",
        "- Textos de desinformação nunca entram na base de evidências (RD-01).",
        "- Golden set e few-shot não compartilham grupos, e excluem quase-duplicatas entre os datasets (RD-05).",
        "- Parte das checagens fica fora da base, para avaliar o caso \"sem evidência suficiente\" (IA-07).",
    ]
    return "\n".join(linhas) + "\n"


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    df, boilerplate = montar_camada_generica()
    df.to_parquet(INTERIM / "textos.parquet", index=False)
    (INTERIM / "boilerplate.json").write_text(json.dumps(boilerplate, ensure_ascii=False, indent=1), encoding="utf-8")

    visoes = montar_visoes(df)
    for nome, tabela in visoes.items():
        tabela.to_parquet(PROCESSED / f"{nome}.parquet", index=False)

    relatorio = escrever_relatorio(df, boilerplate, visoes)
    (REPORTS / "preparacao-dados.md").write_text(relatorio, encoding="utf-8")
    print(relatorio)


if __name__ == "__main__":
    main()
