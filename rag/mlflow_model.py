"""Modelo RAG registrável no MLflow; os pesos do Qwen são obtidos pelo Ollama."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow.pyfunc
import requests


def ollama_digest(model_name: str, chat_url: str | None = None) -> str:
    """Obtém o digest completo do modelo instalado no servidor Ollama."""
    chat_url = chat_url or os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    if "/api/" not in chat_url:
        raise ValueError("OLLAMA_URL deve apontar para /api/chat do Ollama")
    tags_url = chat_url.rsplit("/api/", 1)[0] + "/api/tags"
    response = requests.get(tags_url, timeout=15)
    response.raise_for_status()
    for model in response.json().get("models", []):
        if model.get("name") == model_name or model.get("model") == model_name:
            digest = model.get("digest", "")
            if len(digest) == 64:
                return digest
    raise ValueError(f"O modelo {model_name} não está instalado no Ollama em {tags_url}")


class RAGModel(mlflow.pyfunc.PythonModel):
    """Executa o RAG empacotado usando a versão exata do Qwen no Ollama."""

    def __init__(self, ollama_model: str, ollama_sha256: str,
                 embedding_model: str, embedding_revision: str):
        self.ollama_model = ollama_model
        self.ollama_sha256 = ollama_sha256
        self.embedding_model = embedding_model
        self.embedding_revision = embedding_revision
        self.store = None

    def load_context(self, context) -> None:
        from rag.store import VectorStore
        from rag import web

        web.POLICY_PATH = Path(context.artifacts["web_sources"])
        web.source_policy.cache_clear()

        self.store = VectorStore(
            path=Path(context.artifacts["index"]),
            model_name=self.embedding_model,
            model_revision=self.embedding_revision,
        )

    def predict(self, context, model_input, params=None):
        from rag.tutor import ask, ask_online
        from rag.web import WebResearcher

        if self.store is None:
            raise RuntimeError("O índice RAG ainda não foi carregado")
        installed = ollama_digest(self.ollama_model)
        if installed != self.ollama_sha256:
            raise ValueError(
                f"Digest do Ollama divergente para {self.ollama_model}: esperado {self.ollama_sha256}, encontrado {installed}"
            )
        if not hasattr(model_input, "columns") or "pergunta" not in model_input.columns:
            raise ValueError("A entrada deve ser uma tabela com coluna 'pergunta'")
        questions = model_input["pergunta"].tolist()
        if any(not isinstance(question, str) or not question.strip() for question in questions):
            raise ValueError("Todas as perguntas devem ser textos não vazios")
        if "modo" in model_input.columns:
            modes = model_input["modo"].tolist()
        else:
            modes = ["web"] * len(questions)
        if any(mode not in {"web", "local"} for mode in modes):
            raise ValueError("modo deve ser 'web' ou 'local'")
        researcher = WebResearcher() if "web" in modes else None
        answers = []
        for question, mode in zip(questions, modes):
            if mode == "web":
                answer = ask_online(question, researcher, store=self.store, track=False,
                                    model_name=self.ollama_model)
            else:
                answer = ask(question, store=self.store, track=False,
                             model_name=self.ollama_model)
            answers.append(json.dumps(answer, ensure_ascii=False))
        return answers
