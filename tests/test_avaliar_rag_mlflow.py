from scripts.avaliar_rag_mlflow import summarize


def test_summarize_registra_correcao_automatica_da_geracao():
    details = [{
        "erro": None,
        "resposta": {
            "fontes": [], "limitacoes": "Revisão necessária.",
            "evidencias_favoraveis": [], "evidencias_contrarias": [],
        },
        "avaliacao_revisada": None,
        "avaliacao_evidencias": "EVIDENCIAS_INSUFICIENTES",
        "urls_esperadas": [],
        "espera_sem_evidencia": True,
        "latencia_s": 1.0,
        "diagnostico_geracao": {
            "tentativas_geracao": 2,
            "primeira_resposta_valida": False,
            "correcao_automatica_sucesso": True,
            "fallback_seguro": False,
            "afirmacao_original_preservada": True,
        },
    }]

    metrics = summarize(details)

    assert metrics["tentativas_geracao_media"] == 2.0
    assert metrics["taxa_primeira_resposta_valida"] == 0.0
    assert metrics["taxa_correcao_automatica_sucesso"] == 1.0
    assert metrics["taxa_fallback_seguro"] == 0.0
    assert metrics["taxa_afirmacao_original_preservada"] == 1.0
