"""Recuperação lexical de fichas de padrões e adaptação do prompt.

As fichas são conhecimento editorial, não exemplos rotulados de notícias falsas.
O pareamento é explicável e não infere veracidade a partir do estilo.
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

PATTERNS = Path(__file__).resolve().parents[1] / "knowledge" / "padroes_discurso.json"


def analyze_discourse(text: str) -> dict:
    raw = PATTERNS.read_bytes()
    cards = json.loads(raw)
    plain = "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                    if not unicodedata.combining(c))
    retrieved = []
    for card in cards:
        matches = sorted({match.group() for pattern in card["marcadores"]
                          for match in re.finditer(pattern, plain)})
        if matches:
            retrieved.append({key: card[key] for key in
                              ("id", "tipo", "descricao", "exemplo", "orientacao")} |
                             {"sinais_observados": matches})
    return {
        "tipos": [card["tipo"] for card in retrieved] or ["indeterminado"],
        "metodo": "recuperacao_lexical_de_padroes",
        "base_sha256": hashlib.sha256(raw).hexdigest(),
        "padroes_recuperados": retrieved,
        "limite": "Sinais de escrita não comprovam verdade, falsidade nem intenção do autor. Os tipos podem coexistir.",
    }


def adaptive_prompt(base: str, profile: dict) -> str:
    instructions = "\n".join(card["orientacao"] for card in profile["padroes_recuperados"])
    return base + (
        "\n\nOrientação adaptada aos padrões recuperados (hipóteses de leitura, não vereditos):\n"
        + (instructions or "Não há sinais suficientes para identificar o gênero do texto.")
        + "\nAs fichas de padrões são exemplos educativos, nunca fontes factuais da alegação. "
        "Preserve o formato JSON solicitado."
    )


def attach_discourse(answer: dict, profile: dict) -> None:
    answer["analise_discurso"] = profile
    types = set(profile["tipos"])
    if not answer["evidencias_favoraveis"] and not answer["evidencias_contrarias"]:
        framing = []
        if "opiniao" in types:
            framing.append("Há uma marca de opinião pessoal no texto. Julgamentos de valor não recebem confirmação factual; dados e acontecimentos mencionados devem ser verificados separadamente.")
        if "satira_ou_humor" in types:
            framing.append("Há uma referência explícita a humor ou sátira. Ela sugere esse enquadramento, mas é preciso conferir o contexto original para identificar a intenção do autor.")
        if "ficcao_ou_hipotese" in types:
            framing.append("O texto contém um enquadramento de ficção ou hipótese. Os acontecimentos desse cenário não devem ser apresentados como notícia sem verificar o contexto.")
        if "sensacionalismo" in types:
            framing.append("A linguagem contém apelos sensacionalistas. Isso merece atenção, mas não determina se a informação é verdadeira ou falsa.")
        if "rumor" in types:
            framing.append("A informação está atribuída a circulação informal. Falta identificar e conferir sua origem.")
        if framing:
            answer["sintese"] = " ".join(framing) + " " + answer["sintese"]
            answer["veredito"] = answer["sintese"]
    for card in profile["padroes_recuperados"]:
        signals = ", ".join(card["sinais_observados"])
        answer["pontos_de_atencao"].append(
            f"Sinal de {card['tipo'].replace('_', ' ')} ({signals}): {card['descricao']}"
        )
