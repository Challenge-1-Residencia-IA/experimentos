from avaliacao.metricas import (
    agregar, aviso_tema_de_risco, corrigir, fonte_esperada_citada, fonte_esperada_encontrada,
    fonte_esperada_recuperada,
    fontes_completas, natureza_principal, sem_veredito, url_canonica,
)
from avaliacao.sistemas import Resposta

ITEM = {
    "id": "GS001",
    "fonte_esperada": "https://www.boatos.org/saude/vacina.html",
    "afirmacoes": [{"texto": "A OMS aprovou a vacina", "natureza": "fato verificável"}],
    "tecnicas": {"urgencia": True, "pedido_compartilhamento": False,
                 "autoridade_anonima": True, "apelo_emocional": False, "outra": None},
    "tema_de_risco": True,
}
FONTE = {"url": "https://boatos.org/saude/vacina.html/", "origem": "Boatos.org", "data": "2021",
         "citada": True, "descartada": False}


def resposta(**campos) -> Resposta:
    return Resposta(**{"texto": "", "trechos_citados": [], "fontes_recuperadas": [], **campos})


def test_url_canonica_ignora_www_barra_final_e_caixa():
    assert url_canonica("https://WWW.Boatos.org/a/b.html/") == url_canonica("http://boatos.org/a/b.html")


def test_sem_veredito_detecta_afirmacao_categorica_do_sistema():
    assert sem_veredito(ITEM, resposta(texto="Isso é falso."))  is False
    assert sem_veredito(ITEM, resposta(texto="A mensagem é verdadeira, segundo as fontes.")) is False
    assert sem_veredito(ITEM, resposta(texto="As fontes não sustentam a afirmação.")) is True


def test_sem_veredito_ignora_conclusao_citada_entre_aspas():
    assert sem_veredito(ITEM, resposta(texto='A checagem do G1 diz: “É falso que a OMS aprovou”.')) is True


def test_fonte_esperada_descartada_conta_como_encontrada_mas_nao_como_apresentada():
    descartada = resposta(fontes_recuperadas=[{**FONTE, "descartada": True, "citada": False}])
    assert fonte_esperada_encontrada(ITEM, descartada) is True
    assert fonte_esperada_recuperada(ITEM, descartada) is False


def test_fonte_esperada_recuperada_e_citada():
    assert fonte_esperada_recuperada(ITEM, resposta(fontes_recuperadas=[FONTE])) is True
    assert fonte_esperada_citada(ITEM, resposta(fontes_recuperadas=[{**FONTE, "citada": False}])) is False
    assert fonte_esperada_recuperada({**ITEM, "fonte_esperada": None}, resposta()) is None


def test_fontes_completas_exige_veiculo_data_e_link():
    assert fontes_completas(ITEM, resposta(fontes_recuperadas=[FONTE])) is True
    assert fontes_completas(ITEM, resposta(fontes_recuperadas=[{**FONTE, "data": ""}])) is False
    assert fontes_completas(ITEM, resposta()) is None


def test_aviso_tema_de_risco_so_se_aplica_a_itens_de_risco():
    assert aviso_tema_de_risco(ITEM, resposta(texto="Consulte fontes oficiais de saúde.")) is True
    assert aviso_tema_de_risco(ITEM, resposta(texto="Veja as fontes abaixo.")) is False
    assert aviso_tema_de_risco({**ITEM, "tema_de_risco": False}, resposta()) is None


def test_campos_nao_produzidos_ficam_como_nao_medidos():
    notas = corrigir(ITEM, resposta())
    assert notas["natureza_principal"] is None
    assert notas["tecnica_urgencia"] is None


def test_natureza_e_tecnicas_quando_o_sistema_produz():
    r = resposta(afirmacoes=[{"texto": "x", "natureza": "fato verificável"}],
                 tecnicas={"urgencia": True, "autoridade_anonima": False})
    notas = corrigir(ITEM, r)
    assert natureza_principal(ITEM, r) is True
    assert notas["tecnica_urgencia"] is True
    assert notas["tecnica_autoridade_anonima"] is False


def test_agregar_ignora_itens_com_erro_e_nao_medidos():
    resultados = [{"notas": {**corrigir(ITEM, resposta(texto="ok")), "sem_veredito": True}},
                  {"notas": {**corrigir(ITEM, resposta()), "sem_veredito": False}},
                  {"notas": None}]
    linha = {l["metrica"]: l for l in agregar(resultados)}
    assert linha["sem_veredito"]["itens"] == 2 and linha["sem_veredito"]["taxa"] == 0.5
    assert linha["natureza_principal"]["taxa"] is None
