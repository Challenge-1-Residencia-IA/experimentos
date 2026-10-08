"""Índice RAG compartilhado no Neon/PostgreSQL com pgvector."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rag.corpus import CORPUS, ROOT, chunks, read_corpus, stable_id
from rag.store import COLLECTIONS, MODEL

EMBEDDING_DIMENSION = 384
TABLE = "nexoverdict_rag_chunks"
METADATA_FIELDS = (
    "kind", "source", "origin_group", "title", "url", "published_at",
    "natureza", "license", "dataset_name",
)


class _CollectionCount:
    """Compatibilidade com os pontos que consultam collection.count()."""

    def __init__(self, store: NeonVectorStore, kind: str):
        self.store = store
        self.kind = kind

    def count(self) -> int:
        return self.store.count(self.kind)


class NeonVectorStore:
    """Usa os mesmos trechos, IDs e embeddings do índice Chroma local."""

    def __init__(self, database_url: str | None = None, *,
                 namespace: str | None = None, model_name: str = MODEL,
                 model_revision: str | None = None, encoder=None):
        load_dotenv(ROOT / ".env")
        self.database_url = (database_url or os.getenv("NEON_DATABASE_URL", "")).strip()
        if not self.database_url:
            raise ValueError("Defina NEON_DATABASE_URL no .env para usar VECTOR_BACKEND=neon")
        self.namespace = (namespace or os.getenv("NEON_VECTOR_NAMESPACE", "nexoverdict-rag")).strip()
        if not self.namespace:
            raise ValueError("NEON_VECTOR_NAMESPACE não pode ser vazio")
        self.model_name = model_name
        self.encoder = encoder if encoder is not None else SentenceTransformer(
            model_name, revision=model_revision
        )
        dimension = self.encoder.get_sentence_embedding_dimension()
        if dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"O índice Neon exige embeddings de {EMBEDDING_DIMENSION} dimensões; "
                f"o modelo atual gera {dimension}"
            )
        self.collections = {kind: _CollectionCount(self, kind) for kind in COLLECTIONS}

    def _connect(self, *, register: bool = True):
        import psycopg

        connection = psycopg.connect(
            self.database_url, sslmode="require", connect_timeout=10,
        )
        if register:
            from pgvector.psycopg import register_vector

            try:
                register_vector(connection)
            except psycopg.ProgrammingError as error:
                connection.close()
                raise ValueError(
                    "pgvector não está habilitado no Neon. Execute a opção 2 do menu "
                    "no computador que possui o corpus."
                ) from error
        return connection

    def initialize(self) -> None:
        """Cria apenas os objetos deste projeto; não altera outras tabelas."""
        with self._connect(register=False) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
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
                )
            """)
            connection.execute(f"""
                CREATE INDEX IF NOT EXISTS nexoverdict_rag_lookup_idx
                ON {TABLE} (namespace, kind, document_id)
            """)

    def _check_model(self, connection) -> None:
        from psycopg.errors import UndefinedTable

        try:
            rows = connection.execute(
                f"SELECT DISTINCT embedding_model FROM {TABLE} WHERE namespace = %s LIMIT 2",
                (self.namespace,),
            ).fetchall()
        except UndefinedTable as error:
            raise ValueError(
                "Índice Neon ainda não preparado. Execute a opção 2 do menu no "
                "computador que possui o corpus."
            ) from error
        models = {row[0] for row in rows}
        if models and models != {self.model_name}:
            raise ValueError(
                "Este namespace do Neon usa outro modelo de embeddings. "
                "Escolha outro NEON_VECTOR_NAMESPACE ou recrie os dados de forma controlada."
            )

    def count(self, kind: str) -> int:
        if kind not in COLLECTIONS:
            raise ValueError(f"Tipo de documento inválido: {kind}")
        with self._connect() as connection:
            self._check_model(connection)
            row = connection.execute(
                f"SELECT COUNT(*) FROM {TABLE} WHERE namespace = %s AND kind = %s",
                (self.namespace, kind),
            ).fetchone()
        return row[0]

    def index(self, corpus_path: Path = CORPUS) -> dict[str, int]:
        """Atualiza documentos presentes no corpus; nunca apaga outros documentos remotos."""
        from pgvector import Vector
        from psycopg.types.json import Jsonb

        documents = read_corpus(corpus_path)
        counts = {"indexed": 0, "unchanged": 0, "removed_chunks": 0}
        with self._connect() as connection:
            self._check_model(connection)
            connection.commit()
            for document in documents.values():
                kind = document["kind"]
                if kind not in COLLECTIONS:
                    raise ValueError(f"Tipo de documento inválido: {kind}")
                document_id = document["id"]
                old = connection.execute(
                    f"""SELECT id, content_hash, metadata FROM {TABLE}
                        WHERE namespace = %s AND kind = %s AND document_id = %s""",
                    (self.namespace, kind, document_id),
                ).fetchall()
                metadata = {
                    "document_id": document_id,
                    "chunk_index": 0,
                    "content_hash": document["content_hash"],
                    **{field: document.get(field, "") for field in METADATA_FIELDS},
                }
                if old and {row[1] for row in old} == {document["content_hash"]} and all(
                    all(row[2].get(field) == metadata[field] for field in METADATA_FIELDS)
                    for row in old
                ):
                    counts["unchanged"] += 1
                    connection.commit()
                    continue
                parts = chunks(document["text"])
                if not parts:
                    connection.commit()
                    continue
                vectors = self.encoder.encode(
                    [f"passage: {part}" for part in parts],
                    normalize_embeddings=True, batch_size=32,
                ).tolist()
                identifiers = [
                    stable_id(f"{document_id}:{index}:{document['content_hash']}")
                    for index in range(len(parts))
                ]
                connection.execute(
                    f"DELETE FROM {TABLE} WHERE namespace = %s AND kind = %s AND document_id = %s",
                    (self.namespace, kind, document_id),
                )
                for index, (identifier, part, vector) in enumerate(zip(identifiers, parts, vectors)):
                    connection.execute(
                        f"""INSERT INTO {TABLE}
                            (namespace, id, kind, document_id, content_hash, chunk_index,
                             passage, metadata, embedding_model, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (namespace, id) DO UPDATE SET
                                kind = EXCLUDED.kind, document_id = EXCLUDED.document_id,
                                content_hash = EXCLUDED.content_hash,
                                chunk_index = EXCLUDED.chunk_index, passage = EXCLUDED.passage,
                                metadata = EXCLUDED.metadata,
                                embedding_model = EXCLUDED.embedding_model,
                                embedding = EXCLUDED.embedding""",
                        (self.namespace, identifier, kind, document_id,
                         document["content_hash"], index, part,
                         Jsonb({**metadata, "chunk_index": index}), self.model_name, Vector(vector)),
                    )
                counts["indexed"] += 1
                counts["removed_chunks"] += len({row[0] for row in old} - set(identifiers))
                # Cada documento é atômico e uma nova execução retoma os demais.
                connection.commit()
        return counts

    def search(self, query: str, kind: str, limit: int = 5,
               max_per_source: int = 2) -> list[dict]:
        from pgvector import Vector

        if kind not in COLLECTIONS:
            raise ValueError(f"Tipo de documento inválido: {kind}")
        if limit < 1:
            return []
        vector = Vector(self.encoder.encode(
            [f"query: {query}"], normalize_embeddings=True,
        )[0].tolist())
        with self._connect() as connection:
            self._check_model(connection)
            rows = connection.execute(
                f"""SELECT id, passage, metadata, embedding <=> %s AS distance
                    FROM {TABLE}
                    WHERE namespace = %s AND kind = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s""",
                (vector, self.namespace, kind, vector, max(limit * 4, limit)),
            ).fetchall()
        selected = []
        seen_groups = Counter()
        seen_documents = set()
        for identifier, passage, metadata, distance in rows:
            document_id = metadata["document_id"]
            group = metadata["origin_group"]
            if document_id in seen_documents or seen_groups[group] >= max_per_source:
                continue
            selected.append({"id": identifier, "text": passage,
                             "metadata": metadata, "distance": distance})
            seen_documents.add(document_id)
            seen_groups[group] += 1
            if len(selected) == limit:
                break
        return selected
