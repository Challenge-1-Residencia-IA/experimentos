from rag.claims import ambiguous_compound, split_claims
from rag import tutor, web


def test_separa_alegacoes_com_sujeito_compartilhado():
    parts = split_claims("É verdade que o vocalista do Guns N' Roses morreu e puxou o pé do Lula?")
    assert parts == [
        "o vocalista do Guns N' Roses morreu.",
        "o vocalista do Guns N' Roses puxou o pé do Lula.",
    ]
    assert split_claims("O projeto X foi aprovado?") == ["O projeto X foi aprovado?"]
    assert ambiguous_compound("Lula morreu e a eleição foi cancelada?")


def test_pergunta_composta_tem_analises_e_nunca_um_sim_global(monkeypatch):
    calls = []

    def single(claim, *_args, **_kwargs):
        calls.append(claim)
        return {"afirmacao": claim, "fontes": [], "fontes_encontradas": [],
                "avaliacao_evidencias": "EVIDENCIAS_SUFICIENTES_FAVORAVEIS"}

    monkeypatch.setattr(tutor, "_ask_online_single", single)
    answer = tutor.ask_online("O cantor famoso morreu e puxou o pé do político?", object(), store=object())
    assert len(calls) == len(answer["analises"]) == 2
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    assert "não há um veredito único" in answer["veredito"]


def test_google_fact_check_retorna_metadados_sem_transformar_em_prova(monkeypatch):
    researcher = web.WebResearcher()
    monkeypatch.setenv("FACT_CHECK_API_KEY", "chave-de-teste")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"claims": [{"text": "O cantor morreu", "claimReview": [{
                "url": "https://example.org/checagem", "title": "Boato sobre morte",
                "publisher": {"name": "Veículo"}, "textualRating": "Falso",
                "reviewDate": "2026-10-01",
            }]}]}

    requests_made = []

    def fake_get(url, *, params, timeout):
        requests_made.append((url, params, timeout))
        return Response()

    monkeypatch.setattr(web.requests, "get", fake_get)
    checks, status = researcher.fact_checks("O cantor morreu")
    assert status == "ok"
    assert checks[0]["avaliacao_do_veiculo"] == "Falso"
    assert checks[0]["alegacao_checada"] == "O cantor morreu"
    assert requests_made[0][1]["query"] == "O cantor morreu"
    researcher.fact_checks("O cantor morreu")
    assert len(requests_made) == 1


def test_fallback_nao_republica_boato_nem_o_titulo_no_resumo():
    item = {
        "numero": 1, "tipo": "evidencia", "titulo": "Morreu o cantor famoso",
        "origem": "blog", "grupo_origem": "blog", "url": "https://example.org/boato",
        "data": "", "qualificada": False, "trecho": "Morreu o cantor famoso.",
        "relevancia": "relacionada", "distancia_vetorial": 0.2,
    }
    answer = tutor.safe_fallback("O cantor famoso morreu?", [item])
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    assert "Morreu o cantor" not in answer["achados"][0]
    assert answer["fontes_encontradas"][0]["url"] == item["url"]


def test_desmentido_nao_pode_ser_usado_como_evidencia_favoravel():
    claim = "O projeto X foi aprovado?"
    item = {
        "numero": 1, "tipo": "evidencia", "titulo": claim, "origem": "Fonte",
        "grupo_origem": "origem", "url": "https://example.org/projeto", "data": "",
        "qualificada": True, "trecho": "O projeto X não foi aprovado.",
        "relevancia": "direta", "distancia_vetorial": 0.1,
    }
    payload = {"afirmacao": claim, "evidencias_favoraveis": [
        'Fonte [1]: “O projeto X não foi aprovado.”'], "evidencias_contrarias": []}
    try:
        tutor.validate_answer(payload, [item])
    except ValueError as error:
        assert "desmentido" in str(error)
    else:
        raise AssertionError("Desmentido aceito como favorável")


def test_conversa_inclui_checagem_sem_promover_rating_a_veredito(monkeypatch):
    check = {"alegacao_checada": "O projeto X foi aprovado", "titulo": "Checagem do projeto X",
             "url": "https://example.org/checagem", "veiculo": "Agência",
             "avaliacao_do_veiculo": "Falso", "data_revisao": "2026-10-01"}

    class Researcher:
        def fact_checks(self, _claim):
            return [check], "ok"

        def research(self, *_args, **_kwargs):
            return [], {"resultados": 0, "paginas_lidas": 0, "fontes_qualificadas": 0}

        def page(self, _url):
            return {"url": check["url"], "title": check["titulo"], "date": "2026-10-01",
                    "text": "O projeto X não foi aprovado.", "origin_group": "example.org"}

    class Store:
        encoder = object()

        def search(self, *_args, **_kwargs):
            return []

    seen = []
    monkeypatch.setattr(web, "passages", lambda _claim, pages, _encoder: seen.extend(pages) or [])
    monkeypatch.setattr(tutor, "_local_evidence", lambda *_args: [])
    answer = tutor.ask_online("O projeto X foi aprovado?", Researcher(), store=Store(), track=False)
    assert seen[0]["url"] == check["url"]
    assert answer["checagens_encontradas"] == [check]
    assert answer["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    assert answer["pesquisa_web"]["paginas_checagem_lidas"] == 1


def test_interface_mostra_desmentido_descartado_mesmo_se_ha_outras_fontes(monkeypatch):
    from io import StringIO
    from rich.console import Console
    import main

    output = StringIO()
    monkeypatch.setattr(main, "console", Console(file=output, force_terminal=False, width=120))
    answer = tutor.safe_fallback("O cantor morreu?", [
        {"numero": 1, "tipo": "evidencia", "titulo": "Rumor", "origem": "blog",
         "grupo_origem": "blog", "url": "https://example.org/rumor", "data": "",
         "qualificada": False, "trecho": "Rumor", "relevancia": "relacionada"},
        {"numero": 2, "tipo": "evidencia", "titulo": "Cantor não morreu",
         "origem": "jornal", "grupo_origem": "jornal", "url": "https://example.org/desmentido",
         "data": "", "qualificada": False, "trecho": "Desmentido",
         "relevancia": "nao_relacionada"},
    ])
    main.render_answer(answer)
    assert "https://example.org/desmentido" in output.getvalue()
