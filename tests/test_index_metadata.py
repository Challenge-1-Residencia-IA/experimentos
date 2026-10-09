import numpy as np
import pytest

from rag import store
from rag.index_metadata import index_metadata


def test_ingestao_preserva_qualificacao_e_permite_revogar(tmp_path):
    import json
    from rag.corpus import ingest, read_corpus

    (tmp_path / "fonte.txt").write_text("Conteúdo revisado da fonte. " * 10, encoding="utf-8")
    entry = {"path": "fonte.txt", "kind": "evidencia", "source": "Fonte",
             "origin_group": "fonte", "conclusive_eligible": True}
    config = tmp_path / "sources.json"
    corpus = tmp_path / "documents.jsonl"
    config.write_text(json.dumps({"documents": [entry]}), encoding="utf-8")
    assert ingest(config, corpus)["new"] == 1
    assert next(iter(read_corpus(corpus).values()))["conclusive_eligible"] is True
    entry["conclusive_eligible"] = False
    config.write_text(json.dumps({"documents": [entry]}), encoding="utf-8")
    assert ingest(config, corpus)["updated"] == 1
    assert next(iter(read_corpus(corpus).values()))["conclusive_eligible"] is False


def test_mudanca_do_divisor_invalida_atualizacao_periodica(tmp_path, monkeypatch):
    from rag import auto_refresh

    config = tmp_path / "sources.json"
    config.write_text('{"documents": []}', encoding="utf-8")
    before = auto_refresh.config_signature(config)
    monkeypatch.setattr(auto_refresh, "CHUNKING_VERSION", "outra-versao")
    assert auto_refresh.config_signature(config) != before


@pytest.mark.parametrize("value,expected", [(True, True), (False, False), ("false", False), ("true", False), (None, False)])
def test_qualificacao_exige_booleano_explicito(value, expected):
    assert index_metadata({"conclusive_eligible": value})["conclusive_eligible"] is expected


def test_chroma_reindexa_versao_antiga_preserva_qualificacao_e_depois_reutiliza(monkeypatch):
    doc = {"id": "d", "kind": "evidencia", "source": "Fonte", "origin_group": "fonte",
           "title": "Título", "url": "https://example.org", "text": "Texto completo.",
           "content_hash": "hash", "conclusive_eligible": True}

    class Encoder:
        calls = 0

        def encode(self, texts, **kwargs):
            self.calls += 1
            return np.ones((len(texts), 2))

    class Collection:
        ids = ["old"]
        metadata = [{**index_metadata(doc), "document_id": "d", "content_hash": "hash",
                     "chunking_version": "antiga"}]

        def get(self, **kwargs):
            return {"ids": self.ids, "metadatas": self.metadata}

        def upsert(self, ids, metadatas, **kwargs):
            self.ids, self.metadata = ids, metadatas

        def delete(self, **kwargs):
            pass

    instance = store.VectorStore.__new__(store.VectorStore)
    instance.encoder = Encoder()
    instance.collections = {"evidencia": Collection()}
    monkeypatch.setattr(store, "read_corpus", lambda _: {"d": doc})
    assert instance.index()["indexed"] == 1
    assert instance.collections["evidencia"].metadata[0]["conclusive_eligible"] is True
    assert instance.index()["unchanged"] == 1
    assert instance.encoder.calls == 1
    doc["conclusive_eligible"] = False
    assert instance.index()["indexed"] == 1
    assert instance.collections["evidencia"].metadata[0]["conclusive_eligible"] is False
