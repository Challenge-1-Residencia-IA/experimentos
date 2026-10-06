"""Indexação incremental e recuperação no Chroma local."""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag.corpus import CORPUS, ROOT, chunks, read_corpus, stable_id

STORE = ROOT / "vector_store" / "chroma"
MODEL = "intfloat/multilingual-e5-small"
COLLECTIONS = {"evidencia": "evidencias", "educativo": "materiais_educativos"}


class VectorStore:
    def __init__(self, path: Path = STORE, model_name: str = MODEL,
                 model_revision: str | None = None):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name, revision=model_revision)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collections = {
            kind: self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine", "embedding_model": model_name},
            )
            for kind, name in COLLECTIONS.items()
        }
        for collection in self.collections.values():
            if collection.metadata and collection.metadata.get("embedding_model") != model_name:
                raise ValueError("O índice usa outro modelo de embeddings. Crie um novo diretório de índice.")

    def index(self, corpus_path: Path = CORPUS) -> dict[str, int]:
        documents = read_corpus(corpus_path)
        counts = {"indexed": 0, "unchanged": 0, "removed_chunks": 0}
        for document in documents.values():
            kind = document["kind"]
            collection = self.collections[kind]
            document_id = document["id"]
            old = collection.get(where={"document_id": document_id}, include=["metadatas"])
            hashes = {item.get("content_hash") for item in old.get("metadatas", [])}
            old_metadata = old.get("metadatas") or []
            current_metadata = old_metadata[0] if old_metadata else {}
            metadata_unchanged = all(
                current_metadata.get(field) == document.get(field, "")
                for field in ("kind", "source", "origin_group", "title", "url", "published_at")
            )
            if hashes == {document["content_hash"]} and old["ids"] and metadata_unchanged:
                counts["unchanged"] += 1
                continue
            parts = chunks(document["text"])
            if not parts:
                continue
            vectors = self.encoder.encode(
                [f"passage: {part}" for part in parts],
                normalize_embeddings=True,
                batch_size=32,
            ).tolist()
            ids = [stable_id(f"{document_id}:{index}:{document['content_hash']}") for index in range(len(parts))]
            metadata = [
                {
                    "document_id": document_id,
                    "kind": kind,
                    "source": document["source"],
                    "origin_group": document["origin_group"],
                    "title": document["title"],
                    "url": document["url"],
                    "published_at": document.get("published_at", ""),
                    "content_hash": document["content_hash"],
                    "chunk_index": index,
                }
                for index in range(len(parts))
            ]
            collection.upsert(ids=ids, embeddings=vectors, documents=parts, metadatas=metadata)
            stale = set(old["ids"]) - set(ids)
            if stale:
                collection.delete(ids=list(stale))
            counts["indexed"] += 1
        for kind, collection in self.collections.items():
            existing = collection.get(include=["metadatas"])
            obsolete = [
                identifier
                for identifier, metadata in zip(existing["ids"], existing.get("metadatas") or [])
                if metadata["document_id"] not in documents
                or documents[metadata["document_id"]]["kind"] != kind
            ]
            if obsolete:
                collection.delete(ids=obsolete)
                counts["removed_chunks"] += len(obsolete)
        return counts

    def search(self, query: str, kind: str, limit: int = 5, max_per_source: int = 2) -> list[dict]:
        if kind not in self.collections:
            raise ValueError(f"Tipo de documento inválido: {kind}")
        collection = self.collections[kind]
        if collection.count() == 0:
            return []
        vector = self.encoder.encode([f"query: {query}"], normalize_embeddings=True).tolist()
        result = collection.query(
            query_embeddings=vector,
            n_results=min(collection.count(), max(limit * 4, limit)),
            include=["documents", "metadatas", "distances"],
        )
        selected = []
        seen = {}
        for identifier, text, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            group = metadata["origin_group"]
            if seen.get(group, 0) >= max_per_source:
                continue
            selected.append({"id": identifier, "text": text, "metadata": metadata, "distance": distance})
            seen[group] = seen.get(group, 0) + 1
            if len(selected) == limit:
                break
        return selected
