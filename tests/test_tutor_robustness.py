import json
import time

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
            "relevancia": "direta",
        },
        "text": "Trecho recuperado.",
        "distance": 0.1,
    }


def response(citation: int, *, claim: str = "afirmação alterada") -> dict:
    return {
        "afirmacao": claim,
        "evidencias_favoraveis": [f"Fonte [{citation}]: “Trecho recuperado.”"],
        "evidencias_contrarias": [],
        "contexto": f"Contexto documentado [{citation}].",
        "pontos_de_atencao": [],
        "avaliacao_evidencias": "EVIDENCIAS_INSUFICIENTES",
        "limitacoes": "Há somente uma origem independente.",
        "fontes_usadas": [citation],
    }


def test_ask_separa_documentos_valida_uma_vez_e_preserva_afirmacao(monkeypatch):
    question = "O gasto foi de R$ 52.361,34 em 2023."
    generated = [response(1, claim=question)]
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

    answer = tutor.ask(
        question,
        evidence=[hit("evidencia", "https://example.com/evidencia")],
        teaching=[hit("educativo", "file:///guia.md")],
        generation_diagnostics=diagnostics,
    )

    assert answer["afirmacao"] == question
    assert answer["fontes"][0]["tipo"] == "evidencia"
    assert len(prompts) == 1
    assert prompts[0]["evidencias_factuais"][0]["rotulo"] == "E1"
    assert prompts[0]["materiais_educativos"][0]["rotulo"] == "M2"
    assert "documentos" not in prompts[0]
    metrics = tracked["context"]["metricas_execucao"]
    assert metrics["tentativas_geracao"] == 1.0
    assert metrics["primeira_resposta_valida"] == 1.0
    assert metrics["correcao_automatica_sucesso"] == 0.0
    assert metrics["fallback_seguro"] == 0.0
    assert metrics["afirmacao_original_preservada"] == 1.0
    assert len(tracked["payload"]["tentativas"]) == 1
    assert diagnostics["tentativas_geracao"] == 1
    assert diagnostics["correcao_automatica_sucesso"] is False


def test_ask_corrige_rotulo_isolado_localmente_sem_repetir_geracao(monkeypatch):
    calls = []

    def invalid_response(*_args, **_kwargs):
        calls.append(True)
        payload = response(1, claim="A afirmação contém o número 2023.")
        payload["evidencias_favoraveis"] = ["E1"]
        return payload

    monkeypatch.setattr(tutor, "ollama_json", invalid_response)
    question = "A afirmação contém o número 2023."
    diagnostics = {}

    answer = tutor.ask(
        question,
        evidence=[hit("evidencia", "https://example.com/evidencia")],
        teaching=[hit("educativo", "file:///guia.md")],
        track=False,
        generation_diagnostics=diagnostics,
    )

    assert answer["afirmacao"] == question
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    assert answer["evidencias_favoraveis"] == []
    assert answer["evidencias_contrarias"] == []
    assert answer["achados"] == [
        "Fonte [1] — página direta encontrada; conteúdo não validado como prova. Consulte o link abaixo."
    ]
    assert answer["fontes"] == []
    assert answer["veredito"].startswith("Não há fontes suficientes")
    assert [item["url"] for item in answer["fontes_encontradas"]] == [
        "https://example.com/evidencia"
    ]
    assert len(calls) == 1
    assert diagnostics["tentativas_geracao"] == 1
    assert diagnostics["correcao_automatica_sucesso"] is True
    assert diagnostics["referencias_corrigidas_localmente"] == 1
    assert diagnostics["fallback_seguro"] is False
    assert diagnostics["erros_validacao"] == []


def test_validate_ignora_fontes_declaradas_sem_citacao_no_texto():
    payload = response(1)
    payload["evidencias_favoraveis"] = []
    payload["contexto"] = "Contexto sem citação factual."

    answer = tutor.validate_answer(
        payload,
        [{
            "numero": 1, "tipo": "evidencia", "titulo": "Documento",
            "origem": "Fonte", "grupo_origem": "grupo-fonte",
            "url": "https://example.com/evidencia", "data": "2023",
            "qualificada": True, "trecho": "Trecho recuperado.",
            "distancia_vetorial": 0.1,
        }],
    )

    assert answer["fontes_usadas"] == []
    assert answer["fontes"] == []
    assert answer["fontes_encontradas"][0]["citada"] is False


def test_ask_retorna_fallback_extrativo_para_saida_irrecuperavel(monkeypatch):
    calls = []

    def invalid_response(*_args, **_kwargs):
        calls.append(True)
        payload = response(1, claim="A afirmação contém o número 2023.")
        payload["evidencias_favoraveis"] = ["afirmação sem referência"]
        return payload

    monkeypatch.setattr(tutor, "ollama_json", invalid_response)
    diagnostics = {}
    answer = tutor.ask(
        "A afirmação contém o número 2023.",
        evidence=[hit("evidencia", "https://example.com/evidencia")],
        teaching=[],
        track=False,
        generation_diagnostics=diagnostics,
    )

    assert len(calls) == 1
    assert diagnostics["fallback_seguro"] is True
    assert answer["evidencias_favoraveis"] == []
    assert answer["achados"] == [
        "Fonte [1] — página direta encontrada; conteúdo não validado como prova. Consulte o link abaixo."
    ]
    assert answer["fontes"] == []


def test_resposta_inconclusiva_mostra_fontes_encontradas_mesmo_sem_cita_las():
    items = [
        {
            "numero": 1, "tipo": "evidencia", "titulo": "Documento factual",
            "origem": "Fonte", "grupo_origem": "grupo-fonte",
            "url": "https://example.com/factual", "data": "2024",
            "qualificada": True, "trecho": "Trecho factual.",
            "distancia_vetorial": 0.2,
        },
        {
            "numero": 2, "tipo": "educativo", "titulo": "Guia",
            "origem": "NexoVerdict", "grupo_origem": "nexoverdict",
            "url": "file:///guia.md", "data": "", "qualificada": True,
            "trecho": "Como avaliar fontes.", "distancia_vetorial": 0.1,
        },
    ]
    payload = {
        "afirmacao": "Alegação sem confirmação suficiente.",
        "evidencias_favoraveis": [], "evidencias_contrarias": [],
        "contexto": "A busca encontrou uma fonte relacionada.",
        "pontos_de_atencao": [],
        "avaliacao_evidencias": "EVIDENCIAS_INSUFICIENTES",
        "limitacoes": "A fonte não basta para concluir.",
        "fontes_usadas": [],
    }

    answer = tutor.validate_answer(payload, items)

    assert answer["veredito"].startswith("Não há fontes suficientes")
    assert answer["fontes"] == []
    assert answer["fontes_encontradas"] == [{
        "numero": 1, "tipo": "evidencia", "titulo": "Documento factual",
        "origem": "Fonte", "grupo_origem": "grupo-fonte",
        "url": "https://example.com/factual", "data": "2024",
        "qualificada": True, "relevancia": "direta", "citada": False,
    }]


def test_modelo_que_muda_a_pergunta_e_totalmente_descartado(monkeypatch):
    question = "Flávio Bolsonaro vai acabar com o voto feminino?"
    direct = hit("evidencia", "https://example.com/checagem")
    direct["metadata"]["title"] = "É falso que Flávio Bolsonaro vai acabar com o voto feminino"
    direct["text"] = "A informação de que Flávio Bolsonaro acabaria com o voto feminino é falsa."
    related = hit("evidencia", "https://example.org/influenciadores")
    related["metadata"].update(title="Influenciadores atacam o voto feminino", relevancia="relacionada")
    unrelated = hit("evidencia", "https://example.net/auxilio")
    unrelated["metadata"].update(title="Vídeo falso sobre o Auxílio Brasil", relevancia="nao_relacionada")
    prompts = []

    def wrong_answer(_system, user, _schema, **_kwargs):
        prompts.append(json.loads(user))
        return {
            "afirmacao": "Influenciadores atacam o voto feminino",
            "evidencias_favoraveis": ["E2"],
            "evidencias_contrarias": [],
            "contexto": "Os influenciadores ameaçam a democracia.",
            "pontos_de_atencao": ["Investigue os influenciadores."],
            "limitacoes": "Faltam nomes dos influenciadores.",
        }

    monkeypatch.setattr(tutor, "ollama_json", wrong_answer)
    answer = tutor.ask(question, evidence=[related, unrelated, direct], teaching=[], track=False)

    assert answer["afirmacao"] == question
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    assert answer["fontes_usadas"] == []
    assert answer["fontes_encontradas"][0]["url"] == "https://example.com/checagem"
    assert len(answer["fontes_encontradas"]) == 2
    assert answer["fontes_descartadas"][0]["url"] == "https://example.net/auxilio"
    assert len(prompts[0]["evidencias_factuais"]) == 1
    assert prompts[0]["evidencias_factuais"][0]["url"] == "https://example.com/checagem"
    visible = " ".join([answer["contexto"], answer["limitacoes"], *answer["pontos_de_atencao"]])
    assert "influenciadores" not in visible.casefold()
    assert "conteúdo não validado como prova" in answer["achados"][0]


def test_veredito_conclusivo_depende_de_duas_origens_com_trechos_literais():
    question = "O projeto X foi aprovado?"
    evidence = []
    for number, group in enumerate(("origem-a", "origem-b"), 1):
        item = hit("evidencia", f"https://example{number}.com/projeto")
        item["metadata"].update(
            title="O projeto X foi aprovado?", origin_group=group,
            conclusive_eligible=True,
        )
        item["text"] = f"A fonte {number} registra que o projeto X não foi aprovado."
        evidence.append(item)
    items = []
    for number, item in enumerate(evidence, 1):
        items.append({
            "numero": number, "tipo": "evidencia", "titulo": item["metadata"]["title"],
            "origem": "Fonte", "grupo_origem": item["metadata"]["origin_group"],
            "url": item["metadata"]["url"], "data": "2023", "qualificada": True,
            "trecho": item["text"], "distancia_vetorial": 0.1,
            "relevancia": "direta",
        })
    payload = {
        "afirmacao": question,
        "evidencias_favoraveis": [],
        "evidencias_contrarias": [
            'Fonte [1]: “A fonte 1 registra que o projeto X não foi aprovado.”',
            'Fonte [2]: “A fonte 2 registra que o projeto X não foi aprovado.”',
        ],
    }

    answer = tutor.validate_answer(payload, items)
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_SUFICIENTES_CONTRARIAS"
    assert len(answer["fontes"]) == 2
    repeated = "A fonte 1 registra que o projeto X não foi aprovado."
    items[1]["trecho"] = repeated
    payload["evidencias_contrarias"][1] = f'Fonte [2]: “{repeated}”'
    answer = tutor.validate_answer(payload, items)
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    items[1]["trecho"] = "A fonte 2 registra que o projeto X não foi aprovado."
    payload["evidencias_contrarias"][1] = 'Fonte [2]: “A fonte 2 registra que o projeto X não foi aprovado.”'
    items[1]["grupo_origem"] = "origem-a"
    answer = tutor.validate_answer(payload, items)
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    items[0]["grupo_origem"] = "grupo_globo"
    items[1]["grupo_origem"] = "globo.com"
    answer = tutor.validate_answer(payload, items)
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    payload["evidencias_contrarias"][1] = 'Fonte [2]: “Algo inexistente neste documento.”'
    try:
        tutor.validate_answer(payload, items)
    except ValueError as error:
        assert "trecho literal" in str(error)
    else:
        raise AssertionError("Citação não literal foi aceita")


def test_sem_fonte_direta_nao_chama_modelo_mas_mostra_url(monkeypatch):
    related = hit("evidencia", "https://example.org/relacionada")
    related["metadata"].update(title="História do voto feminino", relevancia="relacionada")
    monkeypatch.setattr(tutor, "ollama_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("O modelo não deveria ser chamado")))
    diagnostics = {}

    answer = tutor.ask("Flávio Bolsonaro vai acabar com o voto feminino?",
                       evidence=[related], teaching=[], track=False,
                       generation_diagnostics=diagnostics)

    assert diagnostics["tentativas_geracao"] == 0
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    assert answer["fontes_encontradas"][0]["url"] == "https://example.org/relacionada"


def test_so_paginas_nao_relacionadas_explicam_limite_e_preservam_links(monkeypatch):
    from io import StringIO

    from rich.console import Console

    import main

    pages = []
    for index, title in enumerate(("Bandeira em evento", "Votos em campanha",
                                   "Vídeo atribuído a político", "Desmentido de vídeo"), 1):
        page = hit("evidencia", f"https://example.org/pagina-{index}")
        page["metadata"].update(title=title, relevancia="nao_relacionada")
        pages.append(page)
    monkeypatch.setattr(tutor, "ollama_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("O modelo não deveria ser chamado")))
    diagnostics = {}
    answer = tutor.ask("É verdade que um político ameaçou eleitores?", evidence=pages,
                       teaching=[], track=False, generation_diagnostics=diagnostics)

    assert diagnostics["tentativas_geracao"] == 0
    assert answer["veredito"].startswith("Não encontramos fontes que tratem desta alegação específica")
    assert "Consultamos 4 páginas" in answer["contexto"]
    assert "não prova que a alegação seja verdadeira ou falsa" in answer["limitacoes"]
    assert answer["fontes_encontradas"] == answer["fontes_usadas"] == []
    assert len(answer["fontes_descartadas"]) == 4

    output = StringIO()
    monkeypatch.setattr(main, "console", Console(file=output, force_terminal=False, width=120))
    main.render_answer(answer)
    displayed = output.getvalue()
    assert "não são prova a favor nem contra" in displayed
    for page in pages:
        assert page["metadata"]["url"] in displayed


def test_nenhuma_pagina_recuperada_nao_afirma_que_alegacao_e_falsa():
    answer = tutor.ask("Alegação sem resultado", evidence=[], teaching=[], track=False)

    assert "Não encontramos fontes" in answer["veredito"]
    assert "Não conseguimos recuperar páginas" in answer["contexto"]
    assert answer["fontes_descartadas"] == []


def test_latencia_total_inclui_pesquisa_web(monkeypatch):
    from rag import web

    item = hit("evidencia", "https://example.org/checagem")
    item["metadata"]["title"] = "O projeto X foi aprovado?"
    item["text"] = "O projeto X não foi aprovado."
    monkeypatch.setattr(web, "passages", lambda *_args: [item])
    monkeypatch.setattr(tutor, "_local_evidence", lambda *_args: [])
    monkeypatch.setattr(tutor, "ollama_json", lambda *_args, **_kwargs: {
        "afirmacao": "O projeto X foi aprovado?",
        "evidencias_favoraveis": [],
        "evidencias_contrarias": [],
    })
    captured = {}
    monkeypatch.setattr(tutor, "_track_answer",
                        lambda _question, _items, _trace, _answer, context: captured.update(context))

    class Researcher:
        def research(self, *_args, **_kwargs):
            time.sleep(0.02)
            return [], {"resultados": 1, "paginas_lidas": 1,
                        "fontes_qualificadas": 0, "evidencias_locais": 0}

    class Store:
        encoder = object()

        def search(self, *_args, **_kwargs):
            return []

    tutor.ask_online("O projeto X foi aprovado?", Researcher(), store=Store())
    timings = captured["metricas_execucao"]
    assert timings["latencia_pesquisa_web_s"] >= 0.02
    assert timings["latencia_total_s"] >= timings["latencia_pesquisa_web_s"]
    assert timings["latencia_total_s"] >= timings["latencia_geracao_s"]


def test_pergunta_identica_reutiliza_resposta_na_mesma_conversa(monkeypatch):
    from rag import web

    item = hit("evidencia", "https://example.org/checagem")
    item["metadata"]["title"] = "O projeto X foi aprovado?"
    item["text"] = "O projeto X não foi aprovado."
    monkeypatch.setattr(web, "passages", lambda *_args: [item])
    monkeypatch.setattr(tutor, "_local_evidence", lambda *_args: [])
    counts = {"search": 0, "model": 0}

    def generate(*_args, **_kwargs):
        counts["model"] += 1
        return {"afirmacao": "O projeto X foi aprovado?",
                "evidencias_favoraveis": [], "evidencias_contrarias": []}

    monkeypatch.setattr(tutor, "ollama_json", generate)
    contexts = []
    monkeypatch.setattr(tutor, "_track_answer",
                        lambda _q, _items, _trace, _answer, context: contexts.append(context))

    class Researcher:
        answer_cache = {}

        def research(self, *_args, **_kwargs):
            counts["search"] += 1
            return [], {"resultados": 1, "paginas_lidas": 1,
                        "fontes_qualificadas": 0, "evidencias_locais": 0}

    class Store:
        encoder = object()

        def search(self, *_args, **_kwargs):
            return []

    researcher = Researcher()
    first = tutor.ask_online("O projeto X foi aprovado?", researcher, store=Store())
    second = tutor.ask_online("O projeto X foi aprovado?", researcher, store=Store())
    assert counts == {"search": 1, "model": 1}
    assert len(contexts) == 2
    assert contexts[1]["metricas_execucao"]["cache_hit"] == 1.0
    assert first["veredito"] == second["veredito"]
    assert second["pesquisa_web"]["resposta_em_cache"] is True
