import pytest

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


@pytest.mark.parametrize("claim, quote, side, accepted", [
    ("O projeto X foi aprovado?", "O projeto X foi aprovado.", "evidencias_favoraveis", True),
    ("O projeto X foi aprovado?", "O projeto X não foi aprovado.", "evidencias_contrarias", True),
    ("O projeto X não foi aprovado?", "O projeto X não foi aprovado.", "evidencias_favoraveis", True),
    ("O projeto X não foi aprovado?", "O projeto X foi aprovado.", "evidencias_contrarias", True),
    ("O projeto X foi aprovado?", "O projeto X não foi aprovado.", "evidencias_favoraveis", False),
    ("O projeto X foi aprovado?", "O projeto X foi aprovado.", "evidencias_contrarias", False),
    ("O projeto X não foi aprovado?", "O projeto X foi aprovado.", "evidencias_favoraveis", False),
    ("O projeto X não foi aprovado?", "O projeto X não foi aprovado.", "evidencias_contrarias", False),
])
def test_polaridade_explicita_de_aprovacao_em_ambos_os_lados(claim, quote, side, accepted):
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Projeto X", "origem": "Fonte",
            "grupo_origem": "origem", "url": "https://example.org/projeto", "data": "",
            "qualificada": True, "trecho": quote, "relevancia": "direta"}
    payload = {"afirmacao": claim, "evidencias_favoraveis": [], "evidencias_contrarias": []}
    payload[side] = [f'Fonte [1]: “{quote}”']
    if accepted:
        assert tutor.validate_answer(payload, [item])[side] == [f'Fonte [1]: “{quote}”']
    else:
        with pytest.raises(ValueError, match="[Pp]olaridade|desmentido"):
            tutor.validate_answer(payload, [item])


@pytest.mark.parametrize("source_text, quote", [
    ("É falso que o projeto X foi aprovado.", "o projeto X foi aprovado"),
    ("O projeto X foi aprovado?", "O projeto X foi aprovado?"),
    ("O projeto X foi aprovado, mas o projeto X não foi aprovado.",
     "O projeto X foi aprovado, mas o projeto X não foi aprovado."),
    ("O projeto X pode ter sido aprovado.", "O projeto X pode ter sido aprovado."),
])
def test_aprovacao_ambigua_nao_vira_evidencia_contraria(source_text, quote):
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Projeto X", "origem": "Fonte",
            "grupo_origem": "origem", "url": "https://example.org/projeto", "data": "",
            "qualificada": True, "trecho": source_text, "relevancia": "direta"}
    payload = {"afirmacao": "O projeto X foi aprovado?", "evidencias_favoraveis": [],
               "evidencias_contrarias": [f'Fonte [1]: “{quote}”']}
    with pytest.raises(ValueError, match="Polaridade"):
        tutor.validate_answer(payload, [item])
    assert tutor.safe_fallback(payload["afirmacao"], [item])["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"


@pytest.mark.parametrize("claim, side", [
    ("O projeto Aurora foi aprovado?", "evidencias_contrarias"),
    ("O projeto Aurora foi aprovado?", "evidencias_favoraveis"),
    ("O projeto Aurora não foi aprovado?", "evidencias_contrarias"),
    ("O projeto Aurora não foi aprovado?", "evidencias_favoraveis"),
    ("O projeto Aurora foi aprovado em 2024.", "evidencias_contrarias"),
])
def test_ausencia_de_confirmacao_nao_define_polaridade(claim, side):
    quote = "Não há confirmação da aprovação do projeto Aurora em 2024."
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Projeto Aurora", "origem": "Fonte",
            "grupo_origem": "origem", "url": "https://example.org/aurora", "data": "",
            "qualificada": True, "trecho": quote, "relevancia": "direta"}
    payload = {"afirmacao": claim, "evidencias_favoraveis": [], "evidencias_contrarias": []}
    payload[side] = [f'Fonte [1]: “{quote}”']
    with pytest.raises(ValueError, match="Polaridade"):
        tutor.validate_answer(payload, [item])
    assert tutor.safe_fallback(claim, [item])["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"


@pytest.mark.parametrize("quote", [
    "O projeto Aurora será votado amanhã.",
    "Não encontramos registro de aprovação do projeto Aurora em 2024.",
])
@pytest.mark.parametrize("side", ["evidencias_favoraveis", "evidencias_contrarias"])
@pytest.mark.parametrize("claim", [
    "O projeto Aurora foi aprovado?",
    "O projeto Aurora foi aprovado em 2024.",
])
def test_sem_direcao_explicita_nao_e_evidencia_de_aprovacao(quote, side, claim):
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Projeto Aurora", "origem": "Fonte",
            "grupo_origem": "origem", "url": "https://example.org/aurora", "data": "",
            "qualificada": True, "trecho": quote, "relevancia": "direta"}
    payload = {"afirmacao": claim, "evidencias_favoraveis": [], "evidencias_contrarias": []}
    payload[side] = [f'Fonte [1]: “{quote}”']
    with pytest.raises(ValueError, match="Polaridade"):
        tutor.validate_answer(payload, [item])
    assert tutor.safe_fallback(claim, [item])["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"


@pytest.mark.parametrize("claim, source_text, quote", [
    ("A cantora Elisa morreu?",
     "Não há confirmação de que a cantora Elisa morreu.",
     "Não há confirmação de que a cantora Elisa morreu"),
    ("A cantora Elisa morreu?",
     "Até agora, não há confirmação de que a cantora Elisa morreu na terça-feira.",
     "a cantora Elisa morreu"),
    ("A atleta Elisa foi vacinada?",
     "Não encontramos registro de que a atleta Elisa foi vacinada.",
     "Não encontramos registro de que a atleta Elisa foi vacinada"),
    ("A atleta Elisa foi vacinada?",
     "Segundo a nota, não encontramos registro de que a atleta Elisa foi vacinada ontem.",
     "a atleta Elisa foi vacinada"),
    ("A vacina Luma foi aprovada?",
     "Não há evidência de que a vacina Luma foi aprovada.",
     "Não há evidência de que a vacina Luma foi aprovada"),
    ("A vacina Luma foi aprovada?",
     "Por enquanto, não há evidência de que a vacina Luma foi aprovada no país.",
     "a vacina Luma foi aprovada"),
    ("A cantora Elisa morreu?", "Não houve confirmação da morte da cantora Elisa.",
     "Não houve confirmação da morte da cantora Elisa."),
    ("A cantora Elisa morreu?", "Não existem registros da morte da cantora Elisa.",
     "Não existem registros da morte da cantora Elisa."),
    ("A cantora Elisa morreu?", "Não foram encontradas evidências da morte da cantora Elisa.",
     "Não foram encontradas evidências da morte da cantora Elisa."),
])
@pytest.mark.parametrize("side", ["evidencias_favoraveis", "evidencias_contrarias"])
def test_ausencia_de_prova_nao_sustenta_nenhum_lado_em_alegacoes_diversas(
    claim, source_text, quote, side,
):
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Nota de apuração",
            "origem": "Fonte", "grupo_origem": "origem",
            "url": "https://example.org/nota", "data": "", "qualificada": True,
            "trecho": source_text, "relevancia": "direta"}
    payload = {"afirmacao": claim, "evidencias_favoraveis": [],
               "evidencias_contrarias": []}
    payload[side] = [f'Fonte [1]: “{quote}”']

    with pytest.raises(ValueError):
        tutor.validate_answer(payload, [item])

    fallback = tutor.safe_fallback(claim, [item])
    assert fallback["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    assert fallback["evidencias_favoraveis"] == []
    assert fallback["evidencias_contrarias"] == []
    assert fallback["fontes_usadas"] == []


@pytest.mark.parametrize("claim, quote", [
    ("A cantora Elisa morreu?", "A cantora Elisa não morreu."),
    ("A atleta Elisa foi vacinada?", "A atleta Elisa não foi vacinada."),
    ("A vacina Luma foi aprovada?", "A vacina Luma não foi aprovada."),
])
def test_refutacao_factual_explicita_sustenta_lado_contrario(claim, quote):
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Nota de apuração",
            "origem": "Fonte", "grupo_origem": "origem",
            "url": "https://example.org/nota", "data": "", "qualificada": True,
            "trecho": quote, "relevancia": "direta"}
    payload = {"afirmacao": claim, "evidencias_favoraveis": [],
               "evidencias_contrarias": [f'Fonte [1]: “{quote}”']}

    answer = tutor.validate_answer(payload, [item])
    assert answer["evidencias_contrarias"] == [f'Fonte [1]: “{quote}”']
    assert answer["evidencias_favoraveis"] == []
    assert answer["fontes_usadas"] == [1]


def test_ressalva_sobre_outro_fato_nao_bloqueia_citacao_afirmativa():
    claim = "A cantora Elisa morreu?"
    quote = "a cantora Elisa morreu ontem, segundo o hospital."
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Nota do hospital",
            "origem": "Fonte", "grupo_origem": "origem",
            "url": "https://example.org/nota", "data": "", "qualificada": True,
            "trecho": "Não há confirmação oficial sobre os documentos, mas " + quote,
            "relevancia": "direta"}
    payload = {"afirmacao": claim, "evidencias_favoraveis": [f'Fonte [1]: “{quote}”'],
               "evidencias_contrarias": []}
    answer = tutor.validate_answer(payload, [item])
    assert answer["evidencias_favoraveis"] == [f'Fonte [1]: “{quote}”']


@pytest.mark.parametrize("quote, side, accepted", [
    ("O projeto Aurora foi aprovado em 2024.", "evidencias_favoraveis", True),
    ("O projeto Aurora foi aprovado em 15 de junho de 2024.", "evidencias_favoraveis", True),
    ("O projeto Aurora não foi aprovado em 15 de junho de 2024.", "evidencias_contrarias", False),
    ("O projeto Aurora foi aprovado em 2025.", "evidencias_favoraveis", False),
    ("O projeto Aurora foi aprovado em 2025.", "evidencias_contrarias", False),
    ("O projeto Aurora foi aprovado em 15 de junho de 2025.", "evidencias_favoraveis", False),
    ("O projeto Aurora não foi aprovado em 15 de junho de 2025.", "evidencias_contrarias", False),
    ("O projeto Aurora foi aprovado.", "evidencias_favoraveis", False),
])
def test_aprovacao_com_ano_exige_o_ano_da_alegacao(quote, side, accepted):
    claim = "O projeto Aurora foi aprovado em 2024."
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Projeto Aurora", "origem": "Fonte",
            "grupo_origem": "origem", "url": "https://example.org/aurora", "data": "",
            "qualificada": True, "trecho": quote, "relevancia": "direta"}
    payload = {"afirmacao": claim, "evidencias_favoraveis": [], "evidencias_contrarias": []}
    payload[side] = [f'Fonte [1]: “{quote}”']
    if accepted:
        assert tutor.validate_answer(payload, [item])[side]
    else:
        with pytest.raises(ValueError, match="Polaridade"):
            tutor.validate_answer(payload, [item])


def test_negacao_de_dia_nao_sustenta_negacao_do_ano():
    claim = "O projeto Aurora não foi aprovado em 2024."
    quote = "O projeto Aurora não foi aprovado em 15 de junho de 2024."
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Projeto Aurora", "origem": "Fonte",
            "grupo_origem": "origem", "url": "https://example.org/aurora", "data": "",
            "qualificada": True, "trecho": quote, "relevancia": "direta"}
    payload = {"afirmacao": claim, "evidencias_favoraveis": [f'Fonte [1]: “{quote}”'],
               "evidencias_contrarias": []}
    with pytest.raises(ValueError, match="Polaridade"):
        tutor.validate_answer(payload, [item])


def test_negacao_de_dia_nao_refuta_aprovacao_posterior_no_mesmo_ano():
    claim = "O projeto Aurora foi aprovado em 2024."
    quote = "O projeto Aurora não foi aprovado em 15 de junho de 2024."
    item = {"numero": 1, "tipo": "evidencia", "titulo": "Projeto Aurora", "origem": "Fonte",
            "grupo_origem": "origem", "url": "https://example.org/aurora", "data": "",
            "qualificada": True, "trecho": quote + " O projeto Aurora foi aprovado em 30 de junho de 2024.",
            "relevancia": "direta"}
    payload = {"afirmacao": claim, "evidencias_favoraveis": [],
               "evidencias_contrarias": [f'Fonte [1]: “{quote}”']}
    with pytest.raises(ValueError, match="Polaridade"):
        tutor.validate_answer(payload, [item])


def test_versao_de_resposta_reflete_novo_prompt():
    assert tutor.RESPONSE_VERSION == "analise-evidencias-v13"


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
