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


def normalizar_grafia(texto: str) -> str:
    """Desfaz as diferenças de pré-processamento entre os datasets: caixa, acentos, hífens e dígitos.

    Mantém as quebras de linha, que ainda servem para separar frases.
    """
    t = sem_acentos(texto.lower())
    t = re.sub(r"(\w)-(\w)", r"\1\2", t)
    return re.sub(r"\d+", " ", t)


def harmonizar(texto: str) -> str:
    """Forma comparável entre datasets com pré-processamentos diferentes."""
    return re.sub(r"\s+", " ", normalizar_grafia(texto)).strip()


def frases(texto: str) -> list[str]:
    partes = re.split(r"[.!?]\s+|\s+,\s+,\s*|\n+", texto.lower())
    return [p.strip(" ,") for p in partes if len(p.strip(" ,")) > 20]


def frases_repetidas(textos: Iterable[str], classes: Iterable[str] | None = None, limiar: int = 20) -> pd.DataFrame:
    """Grupos de frases (pela forma canônica) presentes em pelo menos `limiar` textos.

    Cada linha traz o número de textos, a contagem por classe (quando `classes` é informado),
    as variações de escrita encontradas e a mais frequente como exemplo.
    """
    textos = list(textos)
    classes = [None] * len(textos) if classes is None else list(classes)
    docs_por_grupo = Counter()
    classes_por_grupo = defaultdict(Counter)
    variantes = defaultdict(Counter)
    for texto, classe in zip(textos, classes):
        frases_do_texto = set(frases(texto))
        for grupo in {canonica(f) for f in frases_do_texto}:
            docs_por_grupo[grupo] += 1
            if classe is not None:
                classes_por_grupo[grupo][classe] += 1
        for frase in frases_do_texto:
            variantes[canonica(frase)][frase] += 1
    linhas = [
        {"grupo": g, "n_docs": n, "por_classe": dict(classes_por_grupo[g]),
         "variacoes": sorted(variantes[g]), "exemplo": variantes[g].most_common(1)[0][0]}
        for g, n in docs_por_grupo.items() if n >= limiar
    ]
    colunas = ["grupo", "n_docs", "por_classe", "variacoes", "exemplo"]
    return pd.DataFrame(linhas, columns=colunas).sort_values("n_docs", ascending=False, kind="stable", ignore_index=True)


def variacoes_boilerplate(tabela: pd.DataFrame) -> list[str]:
    """Todas as variações de uma tabela de `frases_repetidas`, da mais longa à mais curta (ordem de remoção)."""
    return sorted({v for variacoes in tabela["variacoes"] for v in variacoes}, key=lambda v: (-len(v), v))


def detectar_boilerplate(textos: Iterable[str], limiar: int = 20) -> list[str]:
    """Variações de frases cuja forma canônica aparece em pelo menos `limiar` textos."""
    return variacoes_boilerplate(frases_repetidas(textos, limiar=limiar))


def remover_boilerplate(texto: str, boilerplate: list[str]) -> str:
    t = texto
    for frase in boilerplate:
        t = re.sub(re.escape(frase), " ", t, flags=re.IGNORECASE)
    for padrao in RODAPES:
        t = re.sub(padrao, " ", t, flags=re.IGNORECASE)
    return re.sub(r"[ \t]+", " ", t).strip()


# Padrões aplicados depois de `sem_acentos`, por isso sem acento.
PADRAO_FONTES = (
    r"\b(g1|globo(\.com)?|rede globo|tv globo|uol|folha( de s\.? ?paulo)?|piaui|estadao|reuters|"
    r"agencia (brasil|estado)|diario do brasil|diariodobrasil|antagonista|afolhabrasil|thejornalbrasil|"
    r"jornal brasil|ceticismo politico|boatos(\.org)?|fato ou fake|afp|checamos|lupa|aos fatos|comprova|"
    r"e-?farsas)\b"
)
PADRAO_TEMPO = (
    r"\b(nest[ae]|ontem|hoje|segunda|terca|quarta|quinta|sexta|sabado|domingo|\w*feira|"
    r"janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b"
)
PADRAO_CHECAGEM = (
    r"\b(circulam?|fake|fals[oa]s?|checage[mn]s?|checad[oa]s?|montage[mn]s?|enganos[oa]s?|"
    r"boatos?|verifica\w*|desinforma\w*)\b|\be \. ,"
)


def mascarar(texto: str, boilerplate: list[str], remover_checagem: bool = False) -> str:
    """Apaga o que identifica veículo ou época: boilerplate, rodapés, nomes de veículos e checadores,
    marcadores de tempo e números. Opcionalmente, também a linguagem típica de checagem.

    Usado nos testes de atalho das EDAs: o que sobra é o sinal que não depende desses marcadores.
    """
    t = texto.lower()
    for frase in boilerplate:
        t = t.replace(frase, " ")
    t = sem_acentos(t)
    for padrao in RODAPES:
        t = re.sub(padrao, " ", t, flags=re.IGNORECASE)
    t = re.sub(PADRAO_FONTES, " ", t)
    t = re.sub(PADRAO_TEMPO, " ", t)
    t = re.sub(r"\d+", " ", t)
    if remover_checagem:
        t = re.sub(PADRAO_CHECAGEM, " ", t)
    return re.sub(r"\s+", " ", t).strip()


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
