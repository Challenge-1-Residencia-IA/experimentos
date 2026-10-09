"""Comportamento do backend Neon sem conexão de rede."""

import numpy as np
import pytest

from rag import neon_store, store


class FakeEncoder:
    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **_kwargs):
        return np.array([[0.1] * 384 for _ in texts])


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, *, existing=None, search_rows=None):
        self.existing = existing or []
        self.search_rows = search_rows or []
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def commit(self):
        self.statements.append(("COMMIT", None))

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if "SELECT DISTINCT embedding_model" in sql:
            return FakeResult([])
        if "SELECT id, content_hash, metadata" in sql:
            return FakeResult(self.existing)
        if "SELECT id, passage, metadata" in sql:
            return FakeResult(self.search_rows)
        return FakeResult([])


def test_factory_seleciona_neon_sem_conectar(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("VECTOR_BACKEND", "neon")
    monkeypatch.setenv("NEON_TRANSPORT", "tcp")
    monkeypatch.setattr(neon_store, "NeonVectorStore", lambda: sentinel)

    assert store.create_vector_store() is sentinel


def test_neon_exige_url_e_nao_tenta_conectar(monkeypatch):
    monkeypatch.setenv("NEON_DATABASE_URL", "")
    with pytest.raises(ValueError, match="NEON_DATABASE_URL"):
        neon_store.NeonVectorStore(encoder=FakeEncoder())


def test_busca_neon_preserva_deduplicacao_por_documento(monkeypatch):
    metadata = {"document_id": "a", "origin_group": "grupo", "url": "https://a"}
    connection = FakeConnection(search_rows=[
        ("a-1", "primeiro", metadata, 0.1),
        ("a-2", "segundo", metadata, 0.2),
        ("b-1", "terceiro", {**metadata, "document_id": "b"}, 0.3),
    ])
    remote = neon_store.NeonVectorStore("postgresql://dummy/dummy", encoder=FakeEncoder())
    monkeypatch.setattr(remote, "_connect", lambda **_kwargs: connection)

    found = remote.search("consulta", "evidencia", limit=3, max_per_source=2)

    assert [item["id"] for item in found] == ["a-1", "b-1"]
    sql, params = next((sql, params) for sql, params in connection.statements
                       if "ORDER BY embedding <=>" in sql)
    assert "namespace = %s AND kind = %s" in sql
    assert params[1:3] == ("nexoverdict-rag", "evidencia")


def test_indexacao_neon_altera_so_documento_do_namespace(monkeypatch):
    connection = FakeConnection()
    remote = neon_store.NeonVectorStore("postgresql://dummy/dummy", encoder=FakeEncoder())
    monkeypatch.setattr(remote, "_connect", lambda **_kwargs: connection)
    monkeypatch.setattr(neon_store, "read_corpus", lambda _path: {
        "doc": {
            "id": "doc", "kind": "evidencia", "content_hash": "hash",
            "text": "Trecho para indexar.", "source": "Fonte", "origin_group": "grupo",
            "title": "Título", "url": "https://exemplo.org", "published_at": "",
            "conclusive_eligible": True,
        }
    })

    result = remote.index()

    assert result["indexed"] == 1
    deletes = [(sql, params) for sql, params in connection.statements if "DELETE FROM" in sql]
    assert len(deletes) == 1
    assert "namespace = %s AND kind = %s AND document_id = %s" in deletes[0][0]
    assert deletes[0][1] == ("nexoverdict-rag", "evidencia", "doc")
    assert any("INSERT INTO" in sql for sql, _params in connection.statements)
    params = next(params for sql, params in connection.statements if "INSERT INTO" in sql)
    assert params[7].obj["conclusive_eligible"] is True
    assert params[7].obj["chunking_version"] == neon_store.index_metadata({})["chunking_version"]
