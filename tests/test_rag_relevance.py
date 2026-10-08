from rag.relevance import relevance
from rag.web import WebResearcher, decode_page, repair_mojibake


QUESTION = "É verdade que Flávio Bolsonaro vai acabar com o voto feminino?"


def test_checagem_exata_tem_prioridade_e_tema_generico_nao_e_direto():
    direct = relevance(
        QUESTION,
        "É falso que Flávio Bolsonaro vai acabar com o voto feminino",
        "A informação é falsa.",
    )
    related = relevance(
        QUESTION,
        "Flávio Bolsonaro lança plano para conquistar o voto feminino",
        "A campanha busca apoio das mulheres.",
    )
    unrelated = relevance(
        QUESTION,
        "Vídeo falso sobre Lula e o Auxílio Brasil",
        "Outro assunto.",
    )
    assert direct[0] == "direta"
    assert related[0] == "relacionada"
    assert unrelated[0] == "nao_relacionada"


def test_busca_prioriza_titulo_direto_antes_de_dominio_qualificado(monkeypatch):
    researcher = WebResearcher()
    seen = []
    results = [
        {"href": "https://bbc.com/generico", "title": "Voto feminino na eleição", "body": "Análise geral"},
        {"href": "https://example.org/checagem", "title": "É falso que Flávio Bolsonaro vai acabar com o voto feminino", "body": "Checagem"},
    ]
    monkeypatch.setattr(researcher, "search", lambda _query: results)

    def page(url):
        seen.append(url)
        return {"url": url, "title": next(item["title"] for item in results if item["href"] == url),
                "date": "", "text": "Texto recuperado.", "origin_group": url.split("/")[2]}

    monkeypatch.setattr(researcher, "page", page)
    pages, _diagnostics = researcher.research(QUESTION)
    assert pages[0]["url"] == "https://example.org/checagem"


def test_utf8_tem_prioridade_sobre_cabecalho_incorreto():
    assert decode_page("Flávio é citado".encode(), "iso-8859-1") == "Flávio é citado"
    assert repair_mojibake("FlÃ¡vio é citado") == "Flávio é citado"


def test_desmentido_de_morte_permanece_relacionado():
    level, _score = relevance(
        "O vocalista do Guns N' Roses morreu?",
        "Axl Rose não morreu",
        "O cantor e vocalista segue vivo.",
    )
    assert level in {"direta", "relacionada"}


def test_dominio_conhecido_generico_nao_encerra_busca_de_checagem(monkeypatch):
    researcher = WebResearcher()
    queries = []

    def search(query):
        queries.append(query)
        return [{"href": "https://bbc.com/tema", "title": "Tema geral", "body": "Notícia geral"}]

    monkeypatch.setattr(researcher, "search", search)
    monkeypatch.setattr(researcher, "page", lambda _url: {
        "url": "https://bbc.com/tema", "title": "Tema geral", "date": "",
        "text": "Esta página não analisa a afirmação enviada pelo usuário.",
        "origin_group": "bbc.com",
    })
    _pages, diagnostics = researcher.research("O projeto X foi aprovado?")
    assert len(queries) == 2
    assert diagnostics["fontes_qualificadas"] == 0
