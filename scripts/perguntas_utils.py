"""Contrato compartilhado entre treino e inferência de perguntas."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from evidencias_utils import extrair_json


DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
ARQUIVOS_PROMPTS = {
    "evidencias": ("perguntas_evidencias_sistema.txt", "perguntas_evidencias_usuario.txt"),
    "conversa": ("perguntas_conversa_sistema.txt", "perguntas_conversa_usuario.txt"),
}


def carregar_templates(modo: str, prompts_dir: Path = DEFAULT_PROMPTS_DIR) -> tuple[str, str]:
    """Lê os prompts a cada chamada para que alterações sejam aplicadas sem mudar código."""
    if modo not in ARQUIVOS_PROMPTS:
        raise ValueError(f"Modo de pergunta desconhecido: {modo}")
    sistema_nome, usuario_nome = ARQUIVOS_PROMPTS[modo]
    sistema = (prompts_dir / sistema_nome).read_text(encoding="utf-8").strip()
    usuario = (prompts_dir / usuario_nome).read_text(encoding="utf-8").strip()
    obrigatorios = ({"{minimo_fontes}"}, {"{pergunta}", "{evidencias}"}) if modo == "evidencias" else (set(), {"{pergunta}"})
    if (
        not sistema or not usuario
        or any(marcador not in sistema for marcador in obrigatorios[0])
        or any(marcador not in usuario for marcador in obrigatorios[1])
    ):
        raise ValueError(f"Templates vazios ou sem marcadores obrigatórios em {prompts_dir} para {modo}.")
    return sistema, usuario


def formatar_evidencias(fontes: list[dict[str, str]]) -> str:
    if not fontes:
        return "Nenhuma evidência recuperada."
    return "\n\n".join(
        f"[{indice}] {fonte.get('titulo', '')}\nURL: {fonte.get('url', '')}\nTrecho: {fonte.get('trecho', '')}"
        for indice, fonte in enumerate(fontes, 1)
    )


def montar_mensagens_pergunta(
    pergunta: str,
    modo: str,
    evidencias: str = "",
    minimo_fontes: int = 1,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
) -> list[dict[str, str]]:
    sistema, usuario = carregar_templates(modo, prompts_dir)
    if modo == "conversa":
        return [
            {"role": "system", "content": sistema},
            {"role": "user", "content": re.sub(r"\{pergunta\}", lambda _: pergunta.strip(), usuario)},
        ]
    valores = {"pergunta": pergunta.strip(), "evidencias": evidencias.strip()}
    return [
        {"role": "system", "content": sistema.replace("{minimo_fontes}", str(minimo_fontes))},
        {
            "role": "user",
            "content": re.sub(r"\{(pergunta|evidencias)\}", lambda achado: valores[achado.group(1)], usuario),
        },
    ]


def montar_prompt(tokenizer: Any, mensagens: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in mensagens) + "\nASSISTANT:"


def validar_resposta_factual(texto: str, total_fontes: int, minimo_fontes: int) -> dict:
    objeto = extrair_json(texto) or {}
    status = objeto.get("status")
    resposta = objeto.get("resposta")
    justificativa = objeto.get("justificativa")
    fontes_brutas = objeto.get("fontes_usadas")
    estrutura_valida = (
        status in {"RESPONDIDA", "INCONCLUSIVA"}
        and isinstance(resposta, str) and bool(resposta.strip())
        and isinstance(justificativa, str) and bool(justificativa.strip())
        and isinstance(fontes_brutas, list)
    )
    fontes = []
    if isinstance(fontes_brutas, list):
        for indice in fontes_brutas:
            if (
                isinstance(indice, int) and not isinstance(indice, bool)
                and 1 <= indice <= total_fontes and indice not in fontes
            ):
                fontes.append(indice)
    citacoes_validas = isinstance(fontes_brutas, list) and len(fontes) == len(fontes_brutas)
    indices_no_texto = {
        int(indice) for indice in re.findall(
            r"\[(\d+)\]", f"{resposta or ''} {justificativa or ''}"
        )
    }
    citacoes_validas = citacoes_validas and indices_no_texto.issubset(set(fontes))
    citacoes_no_texto = (
        isinstance(resposta, str) and isinstance(justificativa, str)
        and all(f"[{indice}]" in resposta and f"[{indice}]" in justificativa for indice in fontes)
    )
    conclusiva = (
        estrutura_valida and status == "RESPONDIDA" and citacoes_validas
        and len(fontes) >= minimo_fontes and citacoes_no_texto
    )
    diagnostico = {
        "fontes_exigidas": minimo_fontes,
        "fontes_declaradas": len(fontes),
        "indices_declarados": fontes,
        "indices_encontrados_no_texto": sorted(indices_no_texto),
        "campos_obrigatorios_validos": estrutura_valida,
    }
    if conclusiva:
        return {
            "status": "RESPONDIDA",
            "resposta": resposta.strip(),
            "justificativa": justificativa.strip(),
            "fontes_usadas": fontes,
            "json_valido": True,
            "citacoes_validas": True,
            "ajustes_aplicados": [],
            "diagnostico_validacao": diagnostico,
        }
    if estrutura_valida and status == "INCONCLUSIVA" and citacoes_validas and citacoes_no_texto:
        return {
            "status": "INCONCLUSIVA",
            "resposta": resposta.strip(),
            "justificativa": justificativa.strip(),
            "fontes_usadas": fontes,
            "json_valido": True,
            "citacoes_validas": True,
            "ajustes_aplicados": [],
            "diagnostico_validacao": diagnostico,
        }
    motivo = (
        "estrutura_invalida" if not estrutura_valida else
        "citacoes_invalidas" if not citacoes_validas or not citacoes_no_texto else
        "fontes_insuficientes"
    )
    return {
        "status": "INCONCLUSIVA",
        "resposta": "Não é possível responder com segurança usando as fontes disponíveis.",
        "justificativa": f"A resposta do modelo foi rebaixada porque não passou na validação: {motivo}.",
        "fontes_usadas": [],
        "json_valido": estrutura_valida,
        "citacoes_validas": citacoes_validas and bool(citacoes_no_texto),
        "ajustes_aplicados": [motivo],
        "diagnostico_validacao": diagnostico,
    }


def instrucao_correcao(validada: dict, minimo_fontes: int) -> str:
    """Solicita apenas a correção do contrato observável, sem pedir raciocínio interno."""
    motivo = ", ".join(validada.get("ajustes_aplicados", [])) or "formato_invalido"
    return (
        "A resposta anterior foi rejeitada pelo validador por: " + motivo + ". "
        f"Reescreva o JSON completo. Se usar RESPONDIDA, use pelo menos {minimo_fontes} fontes "
        "e inclua cada índice declarado em fontes_usadas tanto na resposta quanto na justificativa, "
        "no formato [1], [2]. Não acrescente texto fora do JSON. Se as evidências realmente não "
        "permitirem responder, retorne INCONCLUSIVA e explique objetivamente a insuficiência."
    )


def validar_resposta_conversa(texto: str) -> dict:
    objeto = extrair_json(texto) or {}
    resposta = objeto.get("resposta")
    justificativa = objeto.get("justificativa")
    estrutura_valida = (
        isinstance(resposta, str) and bool(resposta.strip())
        and isinstance(justificativa, str) and bool(justificativa.strip())
        and objeto.get("fontes_usadas") == []
    )
    if estrutura_valida:
        return {
            "resposta": resposta.strip(),
            "justificativa": justificativa.strip(),
            "fontes_usadas": [],
            "json_valido": True,
            "ajustes_aplicados": [],
        }
    return {
        "resposta": resposta.strip() if isinstance(resposta, str) and resposta.strip() else texto.strip(),
        "justificativa": "O modelo não forneceu uma justificativa estruturada para esta resposta.",
        "fontes_usadas": [],
        "json_valido": False,
        "ajustes_aplicados": ["estrutura_invalida"],
    }
