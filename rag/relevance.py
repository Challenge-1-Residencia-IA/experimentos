"""Triagem conservadora de trechos recuperados para uma alegação."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "acerca", "afirma", "afirmacao", "ainda", "algum", "alguma", "aquela",
    "aquele", "aquilo", "como", "contra", "dessa", "desse", "deste", "disso",
    "entao", "estao", "existe", "falso", "falsa", "haver", "isso", "mesmo",
    "outra", "outro", "pode", "podem", "quando", "quais", "seria", "sobre",
    "verdade", "verdadeira", "verdadeiro", "voce",
}


def terms(value: str) -> set[str]:
    plain = unicodedata.normalize("NFKD", value.casefold())
    plain = "".join(char for char in plain if not unicodedata.combining(char))
    words = {word for word in re.findall(r"[a-z0-9]{4,}", plain) if word not in STOPWORDS}
    # Nomes como Rose/Roses não devem perder a correspondência por pluralização.
    return words | {word[:-1] for word in words if len(word) >= 5 and word.endswith("s")}


def relevance(claim: str, title: str, excerpt: str) -> tuple[str, float]:
    """Classifica correspondência lexical; não infere verdade nem polaridade."""
    query = terms(claim)
    if not query:
        return "relacionada", 0.0
    title_terms = terms(title)
    excerpt_terms = terms(excerpt[:1200])
    title_fraction = len(query & title_terms) / len(query)
    text_fraction = len(query & excerpt_terms) / len(query)
    direct_sentence = len(query) >= 2 and any(
        query <= terms(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", excerpt[:2000])
    )
    # O título precisa conter a maior parte dos elementos da alegação.
    # O corpo sozinho costuma trazer menções incidentais, insuficientes para prova.
    direct = len(query) >= 2 and (title_fraction == 1.0 or direct_sentence)
    if direct:
        level = "direta"
    elif title_fraction >= 0.4 or (title_fraction >= 0.25 and text_fraction >= 0.6):
        level = "relacionada"
    else:
        level = "nao_relacionada"
    return level, round(3 * title_fraction + text_fraction, 4)
