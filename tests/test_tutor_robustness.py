import json

from rag import tutor


def hit(kind: str, url: str) -> dict:
    return {
        "metadata": {
            "kind": kind,
            "title": "Documento",
            "source": "Fonte",
            "origin_group": "grupo-fonte" if kind == "evidencia" else "nexoverdict",
            "url": url,
            "published_at": "2023",
            "conclusive_eligible": True,
        },
        "text": "Trecho recuperado.",
        "distance": 0.1,
    }


def response(citation: int, *, claim: str = "afirmação alterada") -> dict:
    return {
        "afirmacao": claim,
        "evidencias_favoraveis": [f"Fonte [{citation}] apresenta informação."],
        "evidencias_contrarias": [],
        "contexto": f"Contexto documentado [{citation}].",
        "pontos_de_atencao": [],
        "avaliacao_evidencias": "EVIDENCIAS_INSUFICIENTES",
        "limitacoes": "Há somente uma origem independente.",
        "fontes_usadas": [citation],
    }


def test_ask_separa_documentos_corrige_citacao_e_preserva_afirmacao(monkeypatch):
    generated = [response(2), response(1)]
    prompts = []
    tracked = {}
    diagnostics = {}

    def fake_ollama(_system, user, _schema, **_kwargs):
        prompts.append(json.loads(user))
        return generated.pop(0)

    def fake_track(question, items, payload, answer, context):
        tracked.update({"question": question, "items": items, "payload": payload,
                        "answer": answer, "context": context})

    monkeypatch.setattr(tutor, "ollama_json", fake_ollama)
    monkeypatch.setattr(tutor, "_track_answer", fake_track)
    question = "O gasto foi de R$ 52.361,34 em 2023."

    answer = tutor.ask(
        question,
        evidence=[hit("evidencia", "https://example.com/evidencia")],
        teaching=[hit("educativo", "file:///guia.md")],
        generation_diagnostics=diagnostics,
    )

    assert answer["afirmacao"] == question
    assert answer["fontes"][0]["tipo"] == "evidencia"
    assert len(prompts) == 2
    assert prompts[0]["evidencias_factuais"][0]["rotulo"] == "E1"
    assert prompts[0]["materiais_educativos"][0]["rotulo"] == "M2"
    assert "documentos" not in prompts[0]
    assert prompts[1]["erro_de_validacao"] == "evidencias_favoraveis cita material educativo como prova factual"
    metrics = tracked["context"]["metricas_execucao"]
    assert metrics["tentativas_geracao"] == 2.0
    assert metrics["primeira_resposta_valida"] == 0.0
    assert metrics["correcao_automatica_sucesso"] == 1.0
    assert metrics["fallback_seguro"] == 0.0
    assert metrics["afirmacao_original_preservada"] == 1.0
    assert len(tracked["payload"]["tentativas"]) == 2
    assert diagnostics["tentativas_geracao"] == 2
    assert diagnostics["correcao_automatica_sucesso"] is True


def test_ask_retorna_fallback_se_as_duas_respostas_forem_rejeitadas(monkeypatch):
    monkeypatch.setattr(tutor, "ollama_json", lambda *_args, **_kwargs: response(2))
    question = "A afirmação contém o número 2023."

    answer = tutor.ask(
        question,
        evidence=[hit("evidencia", "https://example.com/evidencia")],
        teaching=[hit("educativo", "file:///guia.md")],
        track=False,
    )

    assert answer["afirmacao"] == question
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    assert answer["evidencias_favoraveis"] == []
    assert answer["evidencias_contrarias"] == []
    assert answer["fontes"] == []
