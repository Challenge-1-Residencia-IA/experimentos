from rag.discourse import analyze_discourse, adaptive_prompt
from rag import tutor


def test_texto_misto_recupera_opiniao_fato_e_sensacionalismo():
    profile = analyze_discourse("Na minha opinião, é chocante: a prefeitura anunciou a obra!")
    assert set(profile["tipos"]) == {"opiniao", "alegacao_factual", "sensacionalismo"}
    prompt = adaptive_prompt("JSON", profile)
    assert "julgamento é subjetivo" in prompt
    assert "não" in prompt
    assert all(card["sinais_observados"] for card in profile["padroes_recuperados"])


def test_absurdo_sozinho_nao_e_satira():
    assert "satira_ou_humor" not in analyze_discourse("A Lua virou um queijo gigante") ["tipos"]
    assert "satira_ou_humor" in analyze_discourse("É uma paródia sobre a Lua") ["tipos"]


def test_sem_marcadores_fica_indeterminado():
    assert analyze_discourse("Bom dia") ["tipos"] == ["indeterminado"]


def test_prompt_adaptativo_recebe_fichas_sem_usar_como_evidencia(monkeypatch):
    captured = {}
    claim = "Na minha opinião, a prefeitura anunciou a obra."
    def generate(system, user, schema, **kwargs):
        captured.update(system=system, user=user)
        return {"afirmacao": claim, "evidencias_favoraveis": [], "evidencias_contrarias": []}
    monkeypatch.setattr(tutor, "ollama_json", generate)
    evidence = {"text": "A prefeitura anunciou a obra.", "metadata": {
        "kind": "evidencia", "title": claim, "source": "Fonte", "origin_group": "fonte",
        "url": "https://example.org/obra", "relevancia": "direta"}}
    result = tutor.ask(claim, evidence=[evidence], teaching=[], track=False)
    assert "julgamento é subjetivo" in captured["system"]
    assert "padroes_de_escrita" in captured["user"]
    assert "opiniao" in result["analise_discurso"]["tipos"]
    assert result["fontes_usadas"] == []
    assert "opinião pessoal" in result["sintese"]


def test_opiniao_sem_fontes_tem_resposta_adaptada_sem_chamar_modelo(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Não precisa chamar o modelo sem evidências")
    monkeypatch.setattr(tutor, "ollama_json", forbidden)
    result = tutor.ask("Na minha opinião, o debate foi péssimo.", evidence=[], teaching=[], track=False)
    assert "Julgamentos de valor" in result["sintese"]
    assert result["aviso_ia"] == tutor.AI_NOTICE
