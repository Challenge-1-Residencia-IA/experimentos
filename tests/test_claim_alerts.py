from rag.claim_alerts import claim_alerts
from rag.tutor import safe_fallback


def test_ultimo_caso_gera_alerta_e_resposta_automatica():
    answer = safe_fallback(
        "Pablo Vittar rouba ônibus mata Neymar e resgata Lula na prisão, isso é verdade?", [])
    assert answer["alerta_alegacao"]["nivel"] == "sequencia_sensacionalista"
    assert "forte desconfiança" in answer["sintese"]
    assert "sátiras ou boatos" in answer["pontos_de_atencao"][0]
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    assert answer["aviso_ia"] == "Sou uma IA e posso cometer erros."


def test_noticia_grave_isolada_nao_e_rotulada_como_absurda():
    alert = claim_alerts("Um homem roubou um ônibus em Goiás.")
    assert alert["nivel"] == "alegacao_grave"


def test_ficcao_e_negacao_nao_acionam_alerta():
    for claim in ["No filme, a personagem rouba um ônibus, mata o vilão e resgata um preso.",
                  "O cantor não morreu.", "Imagine alguém que rouba, mata e resgata."]:
        assert claim_alerts(claim)["nivel"] == "sem_alerta"


def test_alerta_preserva_links_consultados():
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Um roubo sem relação",
            "origem": "Fonte", "grupo_origem": "fonte", "url": "https://example.org/noticia",
            "data": "", "qualificada": False, "trecho": "Um homem roubou um ônibus.",
            "relevancia": "nao_relacionada"}
    answer = safe_fallback("Alguém rouba ônibus, mata uma pessoa e resgata um preso?", [item])
    assert answer["fontes_descartadas"][0]["url"] == item["url"]
