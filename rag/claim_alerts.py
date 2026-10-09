"""Alertas explicáveis de linguagem; não determinam a verdade de uma notícia."""

import re
import unicodedata


ALERT_VERSION = "claim-alerts-v1"


def claim_alerts(claim: str) -> dict:
    plain = "".join(c for c in unicodedata.normalize("NFKD", claim.casefold())
                    if not unicodedata.combining(c))
    # Hipóteses explícitas e narrativas fictícias não são acusações factuais.
    fictional = bool(re.search(r"\b(?:se|suponha|imagine|ficcao|ficticio|filme|romance|roteiro|jogo)\b", plain))
    negated = bool(re.search(r"\b(?:nao|nunca|jamais|desmente|desmentido)\b", plain))
    if fictional or negated:
        return {"versao": ALERT_VERSION, "nivel": "sem_alerta", "sinais": []}
    patterns = {
        "morte_ou_violencia": r"\b(?:mata|matou|assassina|assassinou|morreu|faleceu)\b",
        "roubo": r"\b(?:rouba|roubou|furta|furtou|sequestra|sequestrou)\b",
        "resgate_ou_fuga": r"\b(?:resgata|resgatou|liberta|libertou|foge|fugiu)\b",
    }
    signals = [name for name, pattern in patterns.items() if re.search(pattern, plain)]
    sensational = len(signals) == 3 and len(claim.split()) <= 45
    serious = "morte_ou_violencia" in signals or "roubo" in signals
    return {"versao": ALERT_VERSION,
            "nivel": "sequencia_sensacionalista" if sensational else "alegacao_grave" if serious else "sem_alerta",
            "sinais": signals if serious or sensational else []}


def apply_claim_alert(answer: dict) -> dict:
    """Acrescenta uma resposta automática calibrada, preservando evidências e links."""
    alert = claim_alerts(answer["afirmacao"])
    answer["alerta_alegacao"] = alert
    if alert["nivel"] == "sem_alerta":
        return answer
    caution = (
        "A mensagem combina roubo, morte ou violência e resgate ou fuga numa sequência curta. "
        "Esse formato é um sinal de sensacionalismo e pode aparecer em sátiras ou boatos; "
        "o sinal, sozinho, não comprova falsidade."
        if alert["nivel"] == "sequencia_sensacionalista" else
        "A mensagem contém uma alegação grave de morte, violência ou crime. "
        "Ela exige evidência específica e não deve ser tratada como fato pela repetição em páginas."
    )
    answer["pontos_de_atencao"].insert(0, caution)
    if alert["nivel"] == "sequencia_sensacionalista" and not (
        answer["evidencias_favoraveis"] or answer["evidencias_contrarias"]
    ):
        answer["sintese"] = (
            "Esta mensagem tem sinais de uma narrativa sensacionalista e merece forte desconfiança. "
            "Não encontramos trechos validados que sustentem essa sequência de acontecimentos. "
            "Páginas sobre pessoas ou assuntos parecidos não comprovam a história. "
            "Trate-a como alegação não verificada e evite compartilhá-la como notícia. "
            "Envie a publicação original, com data e autoria, para investigar cada acontecimento separadamente."
        )
        answer["veredito"] = answer["sintese"]
    return answer
