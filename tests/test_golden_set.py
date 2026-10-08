import pandas as pd

from dados.golden_set import chave, concordancia, leitura_kappa, para_item, sim_nao

LINHA_VALIDA = {
    "id": "GS001", "origem": "faketrue:432", "texto": "aviso importante: a OMS aprovou a vacina",
    "fonte_esperada": "https://www.boatos.org/x.html", "manter": "sim",
    "afirmacao_1": "A OMS aprovou a vacina", "natureza_1": "Fato verificável",
    "afirmacao_2": None, "natureza_2": None, "afirmacao_3": None, "natureza_3": None,
    "tema": "Saúde", "tec_urgencia": "sim", "tec_pedido_compartilhamento": "nao",
    "tec_autoridade_anonima": "sim", "tec_apelo_emocional": "não", "tec_outra": "-",
    "sensivel_ao_tempo": "sim", "precisa_contexto": "não", "tema_de_risco": "sim",
    "anotador": "Felipe Rodrigues", "observacoes": None,
}


def test_chave_unifica_grafias():
    assert chave("Opnião") == chave("opinião") == "opiniao"
    assert chave(" NÃO ") == chave("nao") == "nao"
    assert chave("-") is None and chave(float("nan")) is None


def test_sim_nao():
    assert sim_nao("Sim") is True
    assert sim_nao("nao") is False
    assert sim_nao("talvez") is None


def test_item_valido_sem_pendencias():
    item, pendencias = para_item(pd.Series(LINHA_VALIDA))
    assert pendencias == []
    assert item["afirmacoes"] == [{"texto": "A OMS aprovou a vacina", "natureza": "fato verificável"}]
    assert item["tema"] == "saúde"
    assert item["tecnicas"] == {"urgencia": True, "pedido_compartilhamento": False,
                                "autoridade_anonima": True, "apelo_emocional": False, "outra": None}
    assert item["origem"] == {"dataset": "faketrue", "id_original": "432"}


def test_pendencias_de_campos_vazios_e_valores_invalidos():
    linha = {**LINHA_VALIDA, "natureza_1": "outro", "tema": None, "tec_urgencia": None,
             "natureza_2": "opinião"}
    _, pendencias = para_item(pd.Series(linha))
    assert any("afirmacao_1 sem natureza" in p for p in pendencias)
    assert any("natureza_2 preenchida sem afirmação" in p for p in pendencias)
    assert any(p.startswith("tema inválido") for p in pendencias)
    assert "tec_urgencia vazio ou inválido" in pendencias


def test_concordancia_pareia_por_id_e_ignora_itens_de_um_so_anotador():
    a = pd.DataFrame([{**LINHA_VALIDA, "id": i, "tec_urgencia": v} for i, v in [("A", "sim"), ("B", "não"), ("C", "sim")]])
    b = pd.DataFrame([{**LINHA_VALIDA, "id": i, "tec_urgencia": v} for i, v in [("B", "nao"), ("A", "sim")]])
    tabela = concordancia(a, b)
    assert tabela.loc["tec_urgencia", "itens"] == 2
    assert tabela.loc["tec_urgencia", "kappa"] == 1.0


def test_leitura_kappa_segue_landis_e_koch():
    assert leitura_kappa(-0.1) == "pior que o acaso"
    assert leitura_kappa(0.6) == "moderada"
    assert leitura_kappa(0.8) == "substancial"
    assert leitura_kappa(0.81) == "quase perfeita"
