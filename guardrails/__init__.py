"""Guardrails da tutora: entrada, blindagem do conteúdo externo e saída.

Uso: `responder_com_guardrails(ask_online, mensagem, researcher, store=store)` no lugar de chamar
`ask_online` (ou `ask`) diretamente.
"""

from __future__ import annotations

from collections.abc import Callable

from guardrails.blindagem import blindar_trecho
from guardrails.entrada import TAMANHO_MAXIMO, DecisaoEntrada, verificar_entrada
from guardrails.saida import aplicar_saida

COMO_USAR = "Envie a mensagem, alegação ou link que você recebeu e quer verificar."
RESPOSTAS_RECUSA = {
    "vazia": (COMO_USAR, "Nenhuma mensagem foi recebida."),
    "apresentacao": (
        "Olá! Eu ajudo a avaliar informações que você recebeu: mostro fontes, checagens já publicadas e "
        "os sinais de manipulação da mensagem, para você mesmo tirar a conclusão. " + COMO_USAR,
        "Saudação ou pergunta sobre o assistente.",
    ),
    "fora_do_escopo": (
        "Isso está fora do que eu faço. Eu só ajudo a verificar informações, como notícias, mensagens "
        "encaminhadas, alegações e links; não crio códigos, textos ou trabalhos nem respondo a pedidos "
        "gerais. " + COMO_USAR,
        "Pedido fora do escopo de verificação de informações.",
    ),
    "tentativa_de_instrucao": (
        "Esta mensagem parece tentar alterar o funcionamento do assistente, e isso não é possível. " + COMO_USAR,
        "A mensagem contém instruções dirigidas ao assistente.",
    ),
}

AVISO_TRUNCADA = (
    f"A mensagem passou de {TAMANHO_MAXIMO} caracteres; a análise considerou só o início. "
    "Para analisar outra parte, envie o trecho ou o link original."
)

__all__ = ["blindar_trecho", "responder_com_guardrails", "verificar_entrada"]


def resposta_recusada(decisao: DecisaoEntrada) -> dict:
    """Resposta no mesmo formato da tutora, para a interface exibir sem tratamento especial."""
    texto, contexto = RESPOSTAS_RECUSA[decisao.motivo]
    return {
        "afirmacao": "", "alegacao_verificada": "",
        "evidencias_favoraveis": [], "evidencias_contrarias": [],
        "avaliacao_evidencias": "NAO_APLICAVEL", "veredito": texto,
        "contexto": f"{contexto} A mensagem não foi enviada para análise.",
        "pontos_de_atencao": [], "limitacoes": "Nenhuma busca foi feita para esta mensagem.",
        "fontes": [], "fontes_usadas": [], "fontes_encontradas": [], "fontes_descartadas": [],
        "fontes_consultadas": [], "fonte_enviada": None, "pesquisa_web": None,
        "limite_da_validacao": "Mensagem recusada pelo guardrail de entrada antes da análise.",
        "guardrails": {"entrada": {"recusada": True, "motivo": decisao.motivo, "sinais": decisao.sinais}},
    }


def responder_com_guardrails(responder: Callable[..., dict], mensagem: str, *args, **kwargs) -> dict:
    """Envolve a função de resposta da tutora com os guardrails de entrada e de saída."""
    decisao = verificar_entrada(mensagem)
    if not decisao.permitida:
        return resposta_recusada(decisao)
    resposta = responder(decisao.mensagem, *args, **kwargs)
    registro_saida = aplicar_saida(resposta, decisao.mensagem)
    if "truncada" in decisao.sinais:
        resposta["pontos_de_atencao"].append(AVISO_TRUNCADA)
    resposta["guardrails"] = {
        "entrada": {"recusada": False, "sinais": decisao.sinais},
        "saida": registro_saida,
    }
    return resposta
