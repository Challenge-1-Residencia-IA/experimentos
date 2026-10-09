import pytest

from rag import cache, tutor, web
from rag.metrics import citation_metrics
from scripts.avaliar_rag_mlflow import summarize


def evidence(text="O projeto X foi aprovado."):
    return {"numero": 1, "tipo": "evidencia", "titulo": "O projeto X foi aprovado?",
            "origem": "Fonte", "grupo_origem": "example.org", "url": "https://example.org/x",
            "data": "", "qualificada": True, "trecho": text, "relevancia": "direta"}


def answer(text):
    return {"afirmacao": "O projeto X foi aprovado?", "evidencias_favoraveis": [text],
            "evidencias_contrarias": []}


def test_narrativa_inventada_ao_redor_da_citacao_nao_chega_ao_usuario():
    result = tutor.validate_answer(answer(
        'Está confirmado por todos os especialistas: [1] “O projeto X foi aprovado.” Sem dúvida.'
    ), [evidence()])
    assert result["evidencias_favoraveis"] == ['Fonte [1]: “O projeto X foi aprovado.”']


def test_segunda_citacao_inventada_na_mesma_evidencia_e_rejeitada():
    with pytest.raises(ValueError, match="único trecho"):
        tutor.validate_answer(answer(
            '[1] “O projeto X foi aprovado.” e “Todos os ministros confirmaram.”'
        ), [evidence()])


def test_recorte_que_omite_desmentido_e_rejeitado():
    with pytest.raises(ValueError, match="desmentido"):
        tutor.validate_answer(answer('[1] “O projeto X foi aprovado”'), [
            evidence("É falso que O projeto X foi aprovado.")])


def test_nao_constroi_citacao_cruzando_corpo_e_titulo():
    item = evidence("A votação ainda não aconteceu.")
    with pytest.raises(ValueError, match="trecho literal"):
        tutor.validate_answer(answer('[1] “aconteceu. O projeto X foi aprovado?”'), [item])


def test_cache_expira_limita_memoria_e_isola_mutacao(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(cache, "monotonic", lambda: clock[0])
    store = cache.ResearchCache(ttl=5, capacity=2)
    store.put("a", {"text": "original"})
    store.get("a")["text"] = "alterado"
    assert store.get("a")["text"] == "original"
    store.put("b", 2)
    store.put("c", 3)
    assert store.get("a") is None
    clock[0] = 5.0
    assert store.get("b") is None


def test_busca_repetida_usa_cache_e_depois_pesquisa_novamente(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(cache, "monotonic", lambda: clock[0])
    calls = []

    class Search:
        def __init__(self, **kwargs):
            pass

        def text(self, query, **kwargs):
            calls.append(query)
            return [{"title": str(len(calls))}]

    monkeypatch.setattr(web, "DDGS", Search)
    researcher = web.WebResearcher()
    assert researcher.search("notícia") == researcher.search("notícia")
    assert len(calls) == 1
    clock[0] = 301
    assert researcher.search("notícia")[0]["title"] == "2"


def test_cache_de_resposta_respeita_trava_inconclusiva(monkeypatch):
    researcher = web.WebResearcher()
    monkeypatch.setattr(researcher, "fact_checks", lambda _: ([], "ok"))
    monkeypatch.setattr(researcher, "research", lambda *a, **kw: ([], {}))
    monkeypatch.setattr(web, "passages", lambda *a, **kw: [])

    class Store:
        encoder = object()

        def search(self, *args, **kwargs):
            return []

    store = Store()
    tutor._ask_online_single("Alegação", researcher, store, track=False)
    result = tutor._ask_online_single("Alegação", researcher, store, track=False,
                                      inconclusive_reason="Alegação composta requer análise separada.")
    assert result["veredito"] == "Alegação composta requer análise separada."


def test_metricas_nao_inventam_100_por_cento_sem_citacao():
    assert "taxa_citacoes_validas" not in citation_metrics([], {"evidencias_favoraveis": []})
    result = citation_metrics([evidence()], answer("Fontes [1, 9]"))
    assert result["taxa_citacoes_validas"] == 0.5


def test_avaliacao_inclui_erros_e_abstencoes_no_f1():
    details = []
    for expected, predicted, error in [
        ("EVIDENCIAS_INSUFICIENTES", "EVIDENCIAS_INSUFICIENTES", None),
        ("EVIDENCIAS_SUFICIENTES_FAVORAVEIS", None, "modelo indisponível"),
    ]:
        details.append({"erro": error, "resposta": None, "avaliacao_revisada": expected,
                        "avaliacao_evidencias": predicted, "urls_esperadas": [], "latencia_s": 1.0})
    result = summarize(details)
    assert result["concordancia_total_revisada"] == 0.5
    assert result["f1_macro_avaliacao"] == 0.5


def test_afirmacao_vazia_e_rejeitada_antes_da_busca():
    with pytest.raises(ValueError, match="não vazia"):
        tutor.ask("   ", track=False)


def test_contexto_conclusivo_nao_descreve_todas_as_fontes_como_nao_confirmadas():
    first = evidence("O projeto X foi aprovado na sessão de ontem.")
    second = {**evidence("Na sessão de ontem, o projeto X recebeu aprovação."),
              "numero": 2, "grupo_origem": "outra.org"}
    payload = answer(f'Fonte [1]: “{first["trecho"]}”')
    payload["evidencias_favoraveis"].append(f'Fonte [2]: “{second["trecho"]}”')
    result = tutor.validate_answer(payload, [first, second])
    assert result["avaliacao_evidencias"] == "EVIDENCIAS_SUFICIENTES_FAVORAVEIS"
    assert "2 origens qualificadas" in result["contexto"]
    assert "não são uma confirmação" not in result["contexto"]


def test_erros_da_api_nao_ficam_em_cache(monkeypatch):
    import requests

    monkeypatch.setenv("FACT_CHECK_API_KEY", "teste")
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise requests.Timeout("temporário")

    monkeypatch.setattr(web.requests, "get", fail)
    researcher = web.WebResearcher()
    researcher.fact_checks("alegação")
    researcher.fact_checks("alegação")
    assert len(calls) == 2
