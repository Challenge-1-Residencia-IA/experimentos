import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd

MESES = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}

# Rodapés de veículos que aparecem em várias versões (telefone, pontuação), e por isso
# escapam da detecção de boilerplate por frases repetidas.
RODAPES = [
    r"ps\.?:?\s*esse artigo (?:e|foi) uma sugest[aã]o[^.]*\.",
    r"se voc[eê] qu(?:er|is)\w*\s+sugerir um tema?\s*(?:ao|para\s*o)\s*boatos\.org.*?(?:\d{4,5}-?\d{4}\.?(?: no whatsapp\.?)?|$)",
    r"clique nos links bit\.ly[^.]*\.?",
    r"na p[aá]gina, voc[eê] pode[^.]*\.",
    r"(?:v[ií]deo:? )?veja como identificar (?:se )?uma mensagem (?:[ée] )?falsa",
    r"enviar siga uol not[ií]cias enviar",
]


def sem_acentos(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def canonica(frase: str) -> str:
    """Forma usada para agrupar variações de uma mesma frase: sem acento, número nem pontuação."""
    return re.sub(r"\s+", " ", re.sub(r"[\d\W_]+", " ", sem_acentos(frase.lower()))).strip()


def harmonizar(texto: str) -> str:
    """Forma comparável entre datasets com pré-processamentos diferentes."""
    t = sem_acentos(texto.lower())
    t = re.sub(r"(\w)-(\w)", r"\1\2", t)
    t = re.sub(r"\d+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def frases(texto: str) -> list[str]:
    partes = re.split(r"[.!?]\s+|\s+,\s+,\s*|\n+", texto.lower())
    return [p.strip(" ,") for p in partes if len(p.strip(" ,")) > 20]


def detectar_boilerplate(textos: Iterable[str], limiar: int = 20) -> list[str]:
    """Variações de frases cuja forma canônica aparece em pelo menos `limiar` textos."""
    docs_por_grupo = Counter()
    variantes = defaultdict(set)
    for texto in textos:
        frases_do_texto = set(frases(texto))
        for grupo in {canonica(f) for f in frases_do_texto}:
            docs_por_grupo[grupo] += 1
        for frase in frases_do_texto:
            variantes[canonica(frase)].add(frase)
    selecionadas = {v for g, n in docs_por_grupo.items() if n >= limiar for v in variantes[g]}
    return sorted(selecionadas, key=len, reverse=True)


def remover_boilerplate(texto: str, boilerplate: list[str]) -> str:
    t = texto
    for frase in boilerplate:
        t = re.sub(re.escape(frase), " ", t, flags=re.IGNORECASE)
    for padrao in RODAPES:
        t = re.sub(padrao, " ", t, flags=re.IGNORECASE)
    return re.sub(r"[ \t]+", " ", t).strip()


def parse_data(bruto: str) -> pd.Timestamp:
    if not isinstance(bruto, str):
        return pd.NaT
    s = bruto.strip().lower()
    padroes = [
        (r"(\d{4})-(\d{1,2})-(\d{1,2})", lambda m: (m[1], m[2], m[3])),
        (r"(\d{1,2})/(\d{1,2})/(\d{2,4})\b", lambda m: (m[3], m[2], m[1])),
        (r"(\d{1,2}) de ([a-zç]+) de (\d{4})", lambda m: (m[3], MESES.get(m[2]), m[1])),
    ]
    for padrao, extrair in padroes:
        m = re.search(padrao, s)
        if m:
            ano, mes, dia = extrair(m)
            if len(ano) == 2:
                ano = "20" + ano
            if not 1990 <= int(ano) <= 2030:
                return pd.NaT
            try:
                return pd.Timestamp(int(ano), int(mes), int(dia))
            except (TypeError, ValueError):
                return pd.NaT
    return pd.NaT


def tem_acento(texto: str) -> bool:
    return bool(re.search(r"[áéíóúâêôãõçà]", texto.lower()))
