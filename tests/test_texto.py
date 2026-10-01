import pandas as pd

from dados.texto import canonica, detectar_boilerplate, harmonizar, parse_data, remover_boilerplate, tem_acento


def test_parse_data_formatos():
    assert parse_data("2017-03-05") == pd.Timestamp(2017, 3, 5)
    assert parse_data("05/03/17") == pd.Timestamp(2017, 3, 5)
    assert parse_data("05/03/2017 às 10h") == pd.Timestamp(2017, 3, 5)
    assert parse_data("5 de março de 2017") == pd.Timestamp(2017, 3, 5)


def test_parse_data_invalida():
    assert pd.isna(parse_data(None))
    assert pd.isna(parse_data("sem data"))
    assert pd.isna(parse_data("05/03/1890"))
    assert pd.isna(parse_data("31/02/2017"))


def test_harmonizar_iguala_preprocessamentos():
    assert harmonizar("Ação em 2018, pré-candidato") == harmonizar("acao em 218, precandidato")


def test_canonica_agrupa_variacoes():
    assert canonica("Ligue (61) 99177-9164.") == canonica("ligue 61 993316821")


def test_tem_acento():
    assert tem_acento("Notícia")
    assert not tem_acento("noticia")


def test_detectar_e_remover_boilerplate():
    rodape = "assine a newsletter e receba as principais notícias"
    palavras = "abcdefghijklmnopqrstuvwxy"
    textos = [f"conteúdo específico sobre o tema {letra * 3}. {rodape}" for letra in palavras]
    boilerplate = detectar_boilerplate(textos, limiar=20)
    assert rodape in boilerplate
    assert not any("específico" in frase for frase in boilerplate)
    assert remover_boilerplate(textos[0], boilerplate) == "conteúdo específico sobre o tema aaa."


def test_remover_rodape_por_padrao():
    texto = "O boato é falso. Se você quiser sugerir um tema ao Boatos.org, ligue (61) 99331-6821."
    assert remover_boilerplate(texto, []) == "O boato é falso."


def test_regex_do_pandas_reconhece_letras_acentuadas():
    textos = pd.Series(["nesta terça-feira", "ação"])
    assert textos.str.contains(r"\w+-feira").tolist() == [True, False]
    assert textos.str.contains(r"ção\b").tolist() == [False, True]
