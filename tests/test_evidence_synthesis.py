import pytest

from rag import tutor


def source(number, text, qualified=False):
    return {"numero": number, "tipo": "evidencia", "titulo": "Projeto Aurora",
            "origem": f"Fonte {number}", "grupo_origem": f"origem-{number}",
            "url": f"https://fonte-{number}.example/aurora", "data": "",
            "qualificada": qualified, "trecho": text, "relevancia": "direta"}


@pytest.mark.parametrize("side, direction", [
    ("evidencias_favoraveis", "sustentam"),
    ("evidencias_contrarias", "contradizem"),
])
def test_analise_preserva_direcao_mesmo_sem_qualificacao(side, direction):
    text = ("O projeto Aurora foi aprovado pelo conselho." if direction == "sustentam"
            else "O projeto Aurora não foi aprovado pelo conselho.")
    payload = {"afirmacao": "O projeto Aurora foi aprovado?",
               "evidencias_favoraveis": [], "evidencias_contrarias": []}
    payload[side] = [f'Fonte [1]: “{text}”']
    result = tutor.validate_answer(payload, [source(1, text)])
    assert result["avaliacao_evidencias"] == "EVIDENCIAS_INSUFICIENTES"
    assert f"[1] {direction} a afirmação" in result["sintese"]
    assert "ainda falta corroborá-la" in result["sintese"]
    assert result["veredito"] == result["sintese"]
    assert result["aviso_ia"] == "Sou uma IA e posso cometer erros."
    assert len(result["fontes_encontradas"]) == 1


def test_duas_fontes_qualificadas_nao_apagam_uma_contradicao():
    texts = ["O projeto Aurora foi aprovado pelo conselho.",
             "O conselho aprovou o projeto Aurora na sessão.",
             "O projeto Aurora não foi aprovado pelo conselho."]
    items = [source(n, text, n < 3) for n, text in enumerate(texts, 1)]
    result = tutor.validate_answer({
        "afirmacao": "O projeto Aurora foi aprovado?",
        "evidencias_favoraveis": [f'Fonte [{n}]: “{texts[n-1]}”' for n in (1, 2)],
        "evidencias_contrarias": [f'Fonte [3]: “{texts[2]}”'],
    }, items)
    assert result["avaliacao_evidencias"] == "EVIDENCIAS_CONFLITANTES"
    assert "[1] [2]" in result["sintese"] and "[3]" in result["sintese"]
    assert "quantidade de páginas" in result["sintese"]


def test_sem_evidencias_avisa_sem_inventar_direcao():
    result = tutor.safe_fallback("O projeto Aurora foi aprovado?", [])
    assert result["sintese"].startswith("Não encontramos fontes")
    assert result["aviso_ia"] == tutor.AI_NOTICE


def test_interface_exibe_analise_e_aviso(capsys):
    import main
    main.render_answer(tutor.safe_fallback("O projeto Aurora foi aprovado?", []))
    rendered = capsys.readouterr().out
    assert "Análise" in rendered
    assert "Evidências favoráveis" not in rendered
    assert "Evidências contrárias" not in rendered
    assert "Sou uma IA e posso cometer erros." in rendered
    assert "Veredito" not in rendered
