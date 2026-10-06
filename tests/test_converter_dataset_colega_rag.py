import json

import pandas as pd

from rag.corpus import read_corpus
from scripts.converter_dataset_colega_rag import canonical_url, convert


def test_conversao_preserva_educativo_deduplica_e_impede_vazamento(tmp_path):
    evidence_path = tmp_path / "base.parquet"
    evaluation_path = tmp_path / "avaliacao.parquet"
    interim_path = tmp_path / "interim.parquet"
    corpus_path = tmp_path / "documents.jsonl"
    cases_path = tmp_path / "casos.jsonl"
    manifest_path = tmp_path / "manifest.json"

    pd.DataFrame([
        {"id_checagem": "a", "natureza": "checagem", "texto_limpo": "mesmo texto factual",
         "link": "https://g1.globo.com/a/", "dominio": "g1.globo.com", "ano": 2023},
        {"id_checagem": "b", "natureza": "noticia", "texto_limpo": "mesmo texto factual",
         "link": "https://noticias.uol.com.br/b", "dominio": "noticias.uol.com.br", "ano": 2023},
        {"id_checagem": "c", "natureza": "noticia", "texto_limpo": "outro texto factual",
         "link": "https://g1.globo.com/c", "dominio": "g1.globo.com", "ano": 2022},
    ]).to_parquet(evidence_path, index=False)
    pd.DataFrame([
        {"id_original": "1", "titulo": "Alegação duplicada", "texto_limpo": "entrada 1",
         "link_boato": "https://boato/1", "link_checagem_esperada": "https://noticias.uol.com.br/b",
         "checagem_na_base": True},
        {"id_original": "2", "titulo": "Alegação conhecida", "texto_limpo": "entrada 2",
         "link_boato": "https://boato/2", "link_checagem_esperada": "https://g1.globo.com/c",
         "checagem_na_base": True},
        {"id_original": "3", "titulo": "Alegação reservada", "texto_limpo": "entrada 3",
         "link_boato": "https://boato/3", "link_checagem_esperada": "https://g1.globo.com/fora",
         "checagem_na_base": False},
    ]).to_parquet(evaluation_path, index=False)
    pd.DataFrame([
        {"link": "https://g1.globo.com/a", "titulo": "Título da checagem"},
        {"link": "https://g1.globo.com/c", "titulo": "Título da notícia"},
    ]).to_parquet(interim_path, index=False)
    corpus_path.write_text(json.dumps({
        "id": "educativo-1", "kind": "educativo", "source": "NexoVerdict",
        "origin_group": "nexoverdict", "url": "file:///guia", "title": "Guia",
        "published_at": "", "license": "conteúdo próprio", "content_hash": "x",
        "raw_path": "guia.md", "text": "Como verificar uma notícia.",
    }) + "\n", encoding="utf-8")

    manifest = convert(
        evidence_path, evaluation_path, interim_path, corpus_path, cases_path, manifest_path
    )

    corpus = read_corpus(corpus_path)
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    assert len(corpus) == 3
    assert sum(item["kind"] == "evidencia" for item in corpus.values()) == 2
    assert sum(item["kind"] == "educativo" for item in corpus.values()) == 1
    assert manifest["corpus"]["discarded_exact_duplicates"] == 1
    assert cases[0]["urls_esperadas"] == ["https://g1.globo.com/a"]
    assert cases[2]["urls_esperadas"] == []
    assert cases[2]["espera_sem_evidencia"] is True
    assert "https://g1.globo.com/fora" not in {item["url"] for item in corpus.values()}
    assert manifest["validation"]["reserved_url_leaks"] == 0
    assert manifest["conversion"]["publication_allowed"] is False


def test_canonical_url_remove_fragmento_e_rastreadores():
    assert canonical_url("HTTPS://WWW.Exemplo.com/a/?utm_source=x&b=2#parte") == "https://exemplo.com/a?b=2"
