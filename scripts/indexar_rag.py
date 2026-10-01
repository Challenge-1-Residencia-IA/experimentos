"""Atualiza o índice vetorial com o corpus local."""

from rag.store import VectorStore


if __name__ == "__main__":
    print(VectorStore().index())
