import numpy as np

from rag.store import VectorStore


class FakeEncoder:
    def encode(self, _texts, normalize_embeddings=True):
        return np.array([[0.1, 0.2]])


class FakeCollection:
    def count(self):
        return 3

    def query(self, **_kwargs):
        common = {
            "kind": "evidencia", "source": "Fonte", "origin_group": "grupo",
            "title": "Título", "url": "https://example.com/a", "published_at": "",
            "content_hash": "hash",
        }
        return {
            "ids": [["a-1", "a-2", "b-1"]],
            "documents": [["trecho 1", "trecho 2", "trecho 3"]],
            "metadatas": [[
                {**common, "document_id": "a", "chunk_index": 0},
                {**common, "document_id": "a", "chunk_index": 1},
                {**common, "document_id": "b", "chunk_index": 0, "url": "https://example.com/b"},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }


def test_search_retorna_no_maximo_um_chunk_por_documento():
    store = VectorStore.__new__(VectorStore)
    store.encoder = FakeEncoder()
    store.collections = {"evidencia": FakeCollection()}

    results = store.search("consulta", "evidencia", limit=3, max_per_source=3)

    assert [item["id"] for item in results] == ["a-1", "b-1"]
