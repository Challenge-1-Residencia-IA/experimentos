"""Guardrail de entrada: valida a mensagem do usuário antes de qualquer busca ou chamada ao modelo
(RNF-02, GR-03 e escopo do produto)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from guardrails.padroes import APRESENTACAO, FORA_DO_ESCOPO, INJECAO, normalizar

# Tamanho máximo aceito (caracteres). Valor provisório até a medição da linha de base (IA-01, "LB").
TAMANHO_MAXIMO = 6000
CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")


@dataclass
class DecisaoEntrada:
    permitida: bool
    mensagem: str
    """Mensagem limpa, que segue para a análise quando permitida."""
    motivo: str | None = None
    sinais: list[str] = field(default_factory=list)


def verificar_entrada(mensagem: str) -> DecisaoEntrada:
    """Recusa, sem chamar o modelo, mensagens vazias, saudações, pedidos fora do escopo e tentativas de
    alterar o comportamento do sistema; encurta as longas demais.

    Caracteres de controle e invisíveis (usados para esconder instruções) são removidos antes das
    demais verificações.
    """
    limpa = CONTROLE.sub("", mensagem or "").strip()
    sinais = ["caracteres_invisiveis"] if limpa != (mensagem or "").strip() else []
    if not limpa:
        return DecisaoEntrada(False, limpa, "vazia", sinais)
    if len(limpa) > TAMANHO_MAXIMO:
        # IA-01: acima do tamanho máximo, analisa-se o início da mensagem e o usuário é avisado.
        limpa = limpa[:TAMANHO_MAXIMO].rsplit(" ", 1)[0]
        sinais.append("truncada")
    normalizada = normalizar(limpa)
    if INJECAO.search(normalizada):
        return DecisaoEntrada(False, limpa, "tentativa_de_instrucao", [*sinais, "injecao"])
    if APRESENTACAO.search(normalizada):
        return DecisaoEntrada(False, limpa, "apresentacao", sinais)
    if FORA_DO_ESCOPO.search(normalizada):
        return DecisaoEntrada(False, limpa, "fora_do_escopo", sinais)
    return DecisaoEntrada(True, limpa, None, sinais)
