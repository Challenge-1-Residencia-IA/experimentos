"""Contrato do transporte HTTPS sem modificar o banco remoto."""

import json

import numpy as np
import pytest
from neon_serverless import NeonDbError

from rag import neon_http_store, store


class FakeEncoder:
    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **_kwargs):
        return np.array([[0.1] * 384 for _ in texts])


class FakeClient:
    def __init__(self, *, existing=None, search_rows=None):
        self.existing = existing or []
        self.search_rows = search_rows or []
        self.queries = []
        self.batches = []

    def __call__(self, query, params=None):
        self.queries.append((query, params))
        if "SELECT DISTINCT embedding_model" in query:
            return []
        if "SELECT COUNT(*)" in query:
            return [{"total": 2}]
        if "SELECT id, content_hash, metadata" in query:
            return self.existing
        if "SELECT id, passage, metadata" in query:
            return self.search_rows
        return []

    def transaction(self, statements):
        self.batches.append(statements)
        return [[] for _ in statements]

    def close(self):
        pass


def test_factory_seleciona_https_sem_conexao(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("VECTOR_BACKEND", "neon")
    monkeypatch.setenv("NEON_TRANSPORT", "https")
    monkeypatch.setattr(neon_http_store, "NeonHttpVectorStore", lambda: sentinel)
    assert store.create_vector_store() is sentinel


def test_https_busca_mesma_tabela_com_parametros_e_deduplicacao():
    metadata = {"document_id": "a", "origin_group": "grupo"}
    client = FakeClient(search_rows=[
        {"id": "a-1", "passage": "primeiro", "metadata": metadata, "distance": 0.1},
        {"id": "a-2", "passage": "segundo", "metadata": metadata, "distance": 0.2},
        {"id": "b-1", "passage": "terceiro", "metadata": {**metadata, "document_id": "b"},
         "distance": 0.3},
    ])
    remote = neon_http_store.NeonHttpVectorStore(
        "postgresql://user:password@example.org/db", encoder=FakeEncoder(), client=client
    )
    found = remote.search("consulta", "evidencia", limit=3, max_per_source=2)
    assert [item["id"] for item in found] == ["a-1", "b-1"]
    query, params = next((query, params) for query, params in client.queries
                         if "ORDER BY embedding <=>" in query)
    assert "nexoverdict_rag_chunks" in query
    assert "embedding <=> $1::vector" in query
    assert params[1:3] == ["nexoverdict-rag", "evidencia"]
    assert len(json.loads(params[0])) == 384


def test_https_indexa_documento_em_lote_atomico(monkeypatch):
    client = FakeClient()
    remote = neon_http_store.NeonHttpVectorStore(
        "postgresql://user:password@example.org/db", encoder=FakeEncoder(), client=client
    )
    monkeypatch.setattr(neon_http_store, "read_corpus", lambda _path: {
        "doc": {"id": "doc", "kind": "evidencia", "content_hash": "hash",
                "text": "Trecho para indexar.", "source": "Fonte", "origin_group": "grupo",
                "title": "Título", "url": "https://exemplo.org", "published_at": "",
                "conclusive_eligible": True}
    })
    counts = remote.index()
    assert counts["indexed"] == 1
    assert len(client.batches) == 1
    batch = client.batches[0]
    assert "DELETE FROM" in batch[0][0]
    assert batch[0][1] == ["nexoverdict-rag", "evidencia", "doc"]
    assert "$8::jsonb" in batch[1][0]
    assert "$10::vector" in batch[1][0]
    assert json.loads(batch[1][1][7])["document_id"] == "doc"
    assert json.loads(batch[1][1][7])["conclusive_eligible"] is True
    assert json.loads(batch[1][1][7])["chunking_version"] == neon_http_store.index_metadata({})["chunking_version"]
    assert len(json.loads(batch[1][1][9])) == 384


def test_https_indice_ausente_informa_preparacao():
    class MissingTable(FakeClient):
        def __call__(self, query, params=None):
            raise NeonDbError("relation does not exist", code="42P01")

    remote = neon_http_store.NeonHttpVectorStore(
        "postgresql://user:password@example.org/db", encoder=FakeEncoder(), client=MissingTable()
    )
    with pytest.raises(ValueError, match="opção 2"):
        remote.count("evidencia")
