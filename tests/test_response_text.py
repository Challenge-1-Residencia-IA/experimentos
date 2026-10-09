from rag.response_text import write_response


def answer():
    return {"afirmacao": "A obra foi aprovada?", "sintese": "Os trechos contradizem a afirmação.",
            "evidencias_favoraveis": [],
            "evidencias_contrarias": ['Fonte [1]: “A prefeitura rejeitou a proposta da obra.”'],
            "fontes_usadas": [1], "fontes": [{"numero": 1, "titulo": "Ata", "data": "2026"}],
            "contexto": "Há uma ata.", "limitacoes": "Há uma origem.", "fontes_descartadas": []}


def test_referencia_inventada_na_redacao_retorna_trecho_conferido():
    text, trace = write_response(answer(), lambda *a, **k: {"resposta":
        "A obra foi rejeitada, segundo uma fonte que não consta da análise [99]. Precisamos conferir outras fontes."})
    assert trace["fallback"]
    assert "[99]" not in text
    assert "A prefeitura rejeitou a proposta" in text


def test_redacao_sem_fontes_pode_explicar_opiniao():
    data = answer()
    data.update(evidencias_contrarias=[], fontes_usadas=[], fontes=[], afirmacao="Esse debate foi péssimo.")
    expected = "A palavra ‘péssimo’ expressa uma avaliação pessoal do debate. Para investigar uma informação factual, precisamos de uma afirmação específica sobre o que aconteceu."
    text, trace = write_response(data, lambda *a, **k: {"resposta": expected})
    assert text == expected
    assert not trace["fallback"]


def test_redacao_de_fatos_exige_citacoes():
    _, trace = write_response(answer(), lambda *a, **k: {"resposta":
        "A prefeitura rejeitou a proposta da obra. Isso contradiz a ideia de que ela já estaria aprovada. Ainda precisamos de uma confirmação independente."})
    assert trace["fallback"]


def test_opiniao_nao_termina_exigindo_confirmacao_factual():
    data = answer()
    data.update(evidencias_contrarias=[], fontes_usadas=[], fontes=[], afirmacao="Esse debate foi horrível.")
    text, trace = write_response(data, lambda *a, **k: {"natureza": "opiniao", "resposta":
        "A frase expressa uma opinião negativa sobre o debate. Sem fontes factuais, não podemos confirmar se essa avaliação é verdadeira."})
    assert trace["ajuste_enquadramento"] == "opiniao"
    assert "não podemos confirmar" not in text
    assert "quais aspectos" in text
