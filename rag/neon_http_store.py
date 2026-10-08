"""Acesso ao índice pgvector do Neon por SQL-over-HTTPS."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rag.corpus import CORPUS, chunks, read_corpus, stable_id
from rag.neon_store import EMBEDDING_DIMENSION, METADATA_FIELDS, TABLE, NeonVectorStore
from rag.store import COLLECTIONS


class NeonHttpVectorStore(NeonVectorStore):
    """Mesma tabela/namespace do transporte TCP, com transações HTTP por documento."""

    def __init__(self, database_url: str | None = None, *, client=None, **kwargs):
        super().__init__(database_url, **kwargs)
        self._http_client = client
        self._model_checked = False

    def _sql(self):
        if self._http_client is None:
            from neon_serverless import neon

            self._http_client = neon(self.database_url, timeout=20.0)
        return self._http_client

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
        self._model_checked = False

    def initialize(self) -> None:
        """Cria somente os objetos deste RAG, sem alterar outras tabelas."""
        sql = self._sql()
        sql.transaction([
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                namespace text NOT NULL,
                id text NOT NULL,
                kind text NOT NULL CHECK (kind IN ('evidencia', 'educativo')),
                document_id text NOT NULL,
                content_hash text NOT NULL,
                chunk_index integer NOT NULL,
                passage text NOT NULL,
                metadata jsonb NOT NULL,
                embedding_model text NOT NULL,
                embedding vector({EMBEDDING_DIMENSION}) NOT NULL,
                PRIMARY KEY (namespace, id)
            )""",
            f"""CREATE INDEX IF NOT EXISTS nexoverdict_rag_lookup_idx
                ON {TABLE} (namespace, kind, document_id)""",
        ])
        self._model_checked = False

    def _check_model_http(self) -> None:
        from neon_serverless import NeonDbError

        if self._model_checked:
            return
        try:
            rows = self._sql()(
                f"SELECT DISTINCT embedding_model FROM {TABLE} WHERE namespace = $1 LIMIT 2",
                [self.namespace],
            )
        except NeonDbError as error:
            if error.code == "42P01":
                raise ValueError(
                    "Índice Neon ainda não preparado. Execute a opção 2 do menu no "
                    "computador que possui o corpus."
                ) from error
            raise
        models = {row["embedding_model"] for row in rows}
        if models and models != {self.model_name}:
            raise ValueError(
                "Este namespace do Neon usa outro modelo de embeddings. "
                "Escolha outro NEON_VECTOR_NAMESPACE ou recrie os dados de forma controlada."
            )
        self._model_checked = True

    def count(self, kind: str) -> int:
        if kind not in COLLECTIONS:
            raise ValueError(f"Tipo de documento inválido: {kind}")
        self._check_model_http()
        rows = self._sql()(
            f"SELECT COUNT(*) AS total FROM {TABLE} WHERE namespace = $1 AND kind = $2",
            [self.namespace, kind],
        )
        return int(rows[0]["total"])

    def index(self, corpus_path: Path = CORPUS) -> dict[str, int]:
        """Atualiza cada documento atomicamente; não apaga outros documentos remotos."""
        documents = read_corpus(corpus_path)
        self._check_model_http()
        sql = self._sql()
        counts = {"indexed": 0, "unchanged": 0, "removed_chunks": 0}
        for document in documents.values():
            kind = document["kind"]
            if kind not in COLLECTIONS:
                raise ValueError(f"Tipo de documento inválido: {kind}")
            document_id = document["id"]
            old = sql(
                f"""SELECT id, content_hash, metadata FROM {TABLE}
                    WHERE namespace = $1 AND kind = $2 AND document_id = $3""",
                [self.namespace, kind, document_id],
            )
            metadata = {
                "document_id": document_id,
                "chunk_index": 0,
                "content_hash": document["content_hash"],
                **{field: document.get(field, "") for field in METADATA_FIELDS},
            }
            if old and {row["content_hash"] for row in old} == {document["content_hash"]} and all(
                all(row["metadata"].get(field) == metadata[field] for field in METADATA_FIELDS)
                for row in old
            ):
                counts["unchanged"] += 1
                continue
            parts = chunks(document["text"])
            if not parts:
                continue
            vectors = self.encoder.encode(
                [f"passage: {part}" for part in parts],
                normalize_embeddings=True, batch_size=32,
            ).tolist()
            identifiers = [
                stable_id(f"{document_id}:{index}:{document['content_hash']}")
                for index in range(len(parts))
            ]
            operations = [(
                f"DELETE FROM {TABLE} WHERE namespace = $1 AND kind = $2 AND document_id = $3",
                [self.namespace, kind, document_id],
            )]
            for index, (identifier, part, vector) in enumerate(zip(identifiers, parts, vectors)):
                operations.append((
                    f"""INSERT INTO {TABLE}
                        (namespace, id, kind, document_id, content_hash, chunk_index,
                         passage, metadata, embedding_model, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10::vector)
                        ON CONFLICT (namespace, id) DO UPDATE SET
                            kind = EXCLUDED.kind, document_id = EXCLUDED.document_id,
                            content_hash = EXCLUDED.content_hash,
                            chunk_index = EXCLUDED.chunk_index, passage = EXCLUDED.passage,
                            metadata = EXCLUDED.metadata,
                            embedding_model = EXCLUDED.embedding_model,
                            embedding = EXCLUDED.embedding""",
                    [self.namespace, identifier, kind, document_id,
                     document["content_hash"], index, part,
                     json.dumps({**metadata, "chunk_index": index}, ensure_ascii=False),
                     self.model_name, json.dumps(vector, separators=(",", ":"))],
                ))
            # O driver envia o lote como uma única transação; falha não apaga o documento anterior.
            sql.transaction(operations)
            counts["indexed"] += 1
            counts["removed_chunks"] += len({row["id"] for row in old} - set(identifiers))
        return counts

    def search(self, query: str, kind: str, limit: int = 5,
               max_per_source: int = 2) -> list[dict]:
        if kind not in COLLECTIONS:
            raise ValueError(f"Tipo de documento inválido: {kind}")
        if limit < 1:
            return []
        vector = self.encoder.encode(
            [f"query: {query}"], normalize_embeddings=True,
        )[0].tolist()
        self._check_model_http()
        rows = self._sql()(
            f"""SELECT id, passage, metadata, embedding <=> $1::vector AS distance
                FROM {TABLE}
                WHERE namespace = $2 AND kind = $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4""",
            [json.dumps(vector, separators=(",", ":")), self.namespace,
             kind, max(limit * 4, limit)],
        )
        selected = []
        seen_groups = Counter()
        seen_documents = set()
        for row in rows:
            metadata = row["metadata"]
            document_id = metadata["document_id"]
            group = metadata["origin_group"]
            if document_id in seen_documents or seen_groups[group] >= max_per_source:
                continue
            selected.append({"id": row["id"], "text": row["passage"],
                             "metadata": metadata, "distance": row["distance"]})
            seen_documents.add(document_id)
            seen_groups[group] += 1
            if len(selected) == limit:
                break
        return selected
