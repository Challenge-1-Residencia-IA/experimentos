"""Funções estatísticas e de modelagem compartilhadas pelos notebooks de EDA."""

import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.pipeline import make_pipeline

from dados.texto import sem_acentos

SEED = 42


def carregar_stopwords(sem_acento: bool = False) -> list[str]:
    from nltk.corpus import stopwords

    try:
        palavras = stopwords.words("portuguese")
    except LookupError as erro:
        raise RuntimeError(
            "Stopwords do NLTK não encontradas. Rode `make requirements` "
            "(baixa o recurso para .venv/nltk_data)."
        ) from erro
    return sorted({sem_acentos(p) for p in palavras}) if sem_acento else palavras


def cliffs_delta(a, b) -> float:
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u / (len(a) * len(b)) - 1


def comparar(dados: pd.DataFrame, coluna: str) -> dict:
    """Medianas, delta de Cliff e p-valor de Mann-Whitney de uma coluna entre as classes fake e true."""
    fake = dados.loc[dados["classe"] == "fake", coluna].dropna()
    true = dados.loc[dados["classe"] == "true", coluna].dropna()
    _, p = mannwhitneyu(fake, true, alternative="two-sided")
    return {"mediana_fake": fake.median(), "mediana_true": true.median(),
            "cliffs_delta": cliffs_delta(fake, true), "p_valor": p}


def termos_discriminativos(textos: pd.Series, classes: pd.Series, top_n: int = 20,
                           ngram_range: tuple = (1, 2), stop_words: list | None = None) -> tuple:
    """Termos com maior chi² sobre presença/ausência, separados pela classe em que são mais frequentes."""
    vetorizador = CountVectorizer(binary=True, ngram_range=ngram_range, min_df=5, stop_words=stop_words)
    X = vetorizador.fit_transform(textos)
    e_fake = (classes == "fake").to_numpy()
    scores, _ = chi2(X, e_fake)
    tabela = pd.DataFrame({
        "chi2": scores,
        "prop_fake": np.asarray(X[e_fake].mean(axis=0)).ravel(),
        "prop_true": np.asarray(X[~e_fake].mean(axis=0)).ravel(),
    }, index=vetorizador.get_feature_names_out())
    top_fake = tabela[tabela.prop_fake > tabela.prop_true].nlargest(top_n, "chi2")
    top_true = tabela[tabela.prop_true > tabela.prop_fake].nlargest(top_n, "chi2")
    return top_fake, top_true


def type_token_ratio(texto: str) -> float:
    tokens = re.findall(r"\w+", texto.lower())
    return len(set(tokens)) / len(tokens) if tokens else np.nan


def comp_medio_palavra(texto: str) -> float:
    tokens = re.findall(r"\w+", texto)
    return np.mean([len(t) for t in tokens]) if tokens else np.nan


def comp_medio_sentenca(texto: str) -> float:
    sentencas = [s for s in re.split(r"[.!?]+", texto) if s.strip()]
    return len(texto.split()) / len(sentencas) if sentencas else np.nan


def modelo_texto():
    return make_pipeline(
        TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000),
    )


def avaliar(X, y, modelo, grupos=None) -> pd.Series:
    """Acurácia e F1 médios em validação cruzada de 5 folds.

    Com `grupos`, usa GroupKFold (itens do mesmo grupo nunca ficam no treino e no teste ao mesmo
    tempo); sem, KFold embaralhado.
    """
    if grupos is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    else:
        cv = GroupKFold(n_splits=5)
    resultado = cross_validate(modelo, X, y, groups=grupos, cv=cv, scoring=["accuracy", "f1"])
    return pd.Series({"acuracia": resultado["test_accuracy"].mean(),
                      "f1": resultado["test_f1"].mean()})
