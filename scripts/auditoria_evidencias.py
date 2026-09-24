"""Métricas auditáveis para respostas baseadas em alegações e evidências."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse


PALAVRAS_COMUNS = {
    "a", "as", "ao", "aos", "com", "como", "da", "das", "de", "do", "dos",
    "e", "em", "entre", "foi", "na", "nas", "no", "nos", "o", "os", "para",
    "por", "que", "se", "sem", "um", "uma", "uns", "umas",
}

DOMINIOS_PRIORITARIOS = (
    "gov.br", "jus.br", "leg.br", "edu.br", "who.int", "reuters.com", "apnews.com",
    "agenciabrasil.ebc.com.br", "fiocruz.br", "scielo.br", "aosfatos.org",
    "lupa.uol.com.br", "projetocomprova.com.br",
)


def dominio(url: str) -> str:
    host = urlparse(str(url or "")).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def fonte_prioritaria(url: str) -> bool:
    host = dominio(url)
    return any(host == item or host.endswith(f".{item}") for item in DOMINIOS_PRIORITARIOS)


def tokens_relevantes(texto: str) -> set[str]:
    normalizado = unicodedata.normalize("NFKD", str(texto or ""))
    normalizado = "".join(caractere for caractere in normalizado if not unicodedata.combining(caractere)).lower()
    return {
        token for token in re.findall(r"[a-z0-9]+", normalizado)
        if len(token) >= 3 and token not in PALAVRAS_COMUNS
    }


def cobertura_lexical(texto_referencia: str, texto_base: str) -> float:
    """Fração dos termos relevantes da referência presentes no texto-base."""
    referencia = tokens_relevantes(texto_referencia)
    if not referencia:
        return 0.0
    return len(referencia & tokens_relevantes(texto_base)) / len(referencia)


def extrair_fontes_do_prompt(texto: str) -> list[dict[str, str]]:
    """Converte blocos [N] do prompt em fontes utilizáveis pelas métricas."""
    blocos = re.split(r"(?m)(?=^\[\d+\])", str(texto or ""))
    fontes = []
    for bloco in blocos:
        indice = re.match(r"^\[(\d+)\]\s*(.*)", bloco, flags=re.DOTALL)
        if not indice:
            continue
        corpo = indice.group(2).strip()
        url = re.search(r"(?im)^URL:\s*(\S+)", corpo)
        trecho = re.search(r"(?is)(?:^|\n)(?:Trecho|Evidencia):\s*(.*)", corpo)
        primeira_linha = corpo.splitlines()[0].strip() if corpo else ""
        fontes.append({
            "titulo": primeira_linha,
            "url": url.group(1).strip() if url else "",
            "trecho": trecho.group(1).strip() if trecho else corpo,
        })
    return fontes


def calcular_metricas_auditoria(
    alegacao: str,
    verificacao: dict,
    fontes: list[dict[str, str]],
    minimo_fontes: int,
    metodo_extracao: str | None = None,
) -> dict[str, int | float]:
    usadas = [indice for indice in verificacao.get("fontes_usadas", []) if 1 <= indice <= len(fontes)]
    citadas = [fontes[indice - 1] for indice in usadas]
    texto_citado = " ".join(
        f"{fonte.get('titulo', '')} {fonte.get('trecho', '')}" for fonte in citadas
    )
    justificativa = str(verificacao.get("justificativa", "")).strip()
    referencias_justificativa = {
        int(valor) for valor in re.findall(r"\[(\d+)\]", justificativa)
        if 1 <= int(valor) <= len(fontes)
    }
    requisito_atendido = verificacao.get("veredito") == "INCONCLUSIVA" or len(usadas) >= minimo_fontes
    sobreposicao_justificativa = cobertura_lexical(justificativa, texto_citado) if citadas else 0.0
    componentes = [
        bool(verificacao.get("json_valido")),
        bool(verificacao.get("citacoes_validas")),
        bool(justificativa),
        requisito_atendido,
        verificacao.get("veredito") == "INCONCLUSIVA" or sobreposicao_justificativa >= 0.20,
    ]
    dominios = {dominio(fonte.get("url", "")) for fonte in fontes if dominio(fonte.get("url", ""))}
    prioritarias = sum(fonte_prioritaria(fonte.get("url", "")) for fonte in fontes)
    prioritarias_citadas = sum(fonte_prioritaria(fonte.get("url", "")) for fonte in citadas)
    return {
        "resposta_json_valida": int(bool(verificacao.get("json_valido"))),
        "citacoes_validas": int(bool(verificacao.get("citacoes_validas"))),
        "veredito_conclusivo": int(verificacao.get("veredito") != "INCONCLUSIVA"),
        "fontes_recuperadas": len(fontes),
        "dominios_unicos": len(dominios),
        "diversidade_de_dominios": len(dominios) / len(fontes) if fontes else 0.0,
        "fontes_citadas": len(citadas),
        "cobertura_fontes_recuperadas": len(citadas) / len(fontes) if fontes else 0.0,
        "requisito_minimo_fontes_atendido": int(requisito_atendido),
        "fontes_prioritarias_recuperadas": prioritarias,
        "fontes_prioritarias_citadas": prioritarias_citadas,
        "taxa_fontes_prioritarias_recuperadas": prioritarias / len(fontes) if fontes else 0.0,
        "taxa_fontes_prioritarias_citadas": prioritarias_citadas / len(citadas) if citadas else 0.0,
        "justificativa_presente": int(bool(justificativa)),
        "palavras_justificativa": len(justificativa.split()),
        "fontes_mencionadas_na_justificativa": len(referencias_justificativa & set(usadas)),
        "taxa_fontes_citadas_mencionadas_na_justificativa": (
            len(referencias_justificativa & set(usadas)) / len(usadas) if usadas else 0.0
        ),
        "sobreposicao_justificativa_evidencias": sobreposicao_justificativa,
        "sobreposicao_alegacao_evidencias": cobertura_lexical(alegacao, texto_citado) if citadas else 0.0,
        "veredito_alterado_pelo_validador": int(bool(verificacao.get("veredito_alterado"))),
        "indice_auditabilidade_heuristico": sum(componentes) / len(componentes),
        "extracao_html_direta": int(metodo_extracao == "html_direto") if metodo_extracao else 0,
    }


def montar_rastro_decisao(
    alegacao: str,
    verificacao: dict,
    fontes: list[dict[str, str]],
    minimo_fontes: int,
    metodo_extracao: str | None = None,
    auditoria_recuperacao: dict | None = None,
) -> dict:
    citadas = []
    for indice in verificacao.get("fontes_usadas", []):
        if 1 <= indice <= len(fontes):
            fonte = fontes[indice - 1]
            citadas.append({"indice": indice, **fonte})
    return {
        "observacao": (
            "Rastro baseado em entradas, saída declarada e validações automáticas. "
            "Não representa cadeia de pensamento privada nem prova correção factual."
        ),
        "entrada": {"alegacao": alegacao, "metodo_extracao": metodo_extracao},
        "recuperacao": auditoria_recuperacao or {
            "fontes_apresentadas_ao_modelo": len(fontes),
            "dominios": [dominio(fonte.get("url", "")) for fonte in fontes],
        },
        "regra_de_decisao": {
            "minimo_fontes_conclusivas": minimo_fontes,
            "veredito_conclusivo_exige_citacoes_validas": True,
        },
        "saida_declarada_pelo_modelo": {
            "veredito": verificacao.get("veredito_original"),
            "justificativa": verificacao.get("justificativa_original"),
            "fontes_usadas": verificacao.get("fontes_declaradas_original", verificacao.get("fontes_usadas", [])),
        },
        "validacao": {
            "json_valido": verificacao.get("json_valido"),
            "citacoes_validas": verificacao.get("citacoes_validas"),
            "ajustes_aplicados": verificacao.get("ajustes_aplicados", []),
            "veredito_final": verificacao.get("veredito"),
            "fontes_validas_finais": verificacao.get("fontes_usadas", []),
        },
        "fontes_citadas_com_trechos": citadas,
        "metricas": calcular_metricas_auditoria(
            alegacao, verificacao, fontes, minimo_fontes, metodo_extracao
        ),
    }
