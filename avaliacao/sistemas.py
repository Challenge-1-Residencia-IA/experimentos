"""Adaptadores dos sistemas avaliados.

Cada adaptador recebe a mensagem do usuário e devolve uma `Resposta` padronizada. As métricas
leem só a `Resposta`, então trocar o sistema avaliado (a tutora RAG hoje, a API do bot-core
depois) não muda o restante da avaliação. Campos que o sistema não produz ficam como `None` e
aparecem como "não medido" no boletim.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field


@dataclass
class Resposta:
    texto: str
    """Texto apresentado ao usuário (sem os trechos citados das fontes)."""
    trechos_citados: list[str]
    fontes_recuperadas: list[dict]
    """Fontes mostradas ao usuário, cada uma com `url`, `origem`, `data`, `citada` e `descartada`
    (True quando o sistema a mostra só por transparência, como não relacionada à alegação)."""
    classificacao: str | None = None
    afirmacoes: list[dict] | None = None
    """Afirmações extraídas, cada uma com `texto` e `natureza`; None se o sistema não extrai."""
    tecnicas: dict[str, bool] | None = None
    """Técnicas de manipulação apontadas; None se o sistema não aponta."""
    diagnostico: dict = field(default_factory=dict)


def sha256_texto(valor: str) -> str:
    return hashlib.sha256(valor.encode("utf-8")).hexdigest()[:12]


def resposta_da_tutora(resposta: dict, diagnostico: dict | None = None) -> Resposta:
    """Converte a resposta da tutora (dicionário) para o formato padronizado da avaliação."""
    textos = [resposta.get("veredito", ""), resposta.get("contexto", ""),
              *resposta.get("pontos_de_atencao", []), resposta.get("limitacoes", "")]
    trechos = [*resposta.get("evidencias_favoraveis", []), *resposta.get("evidencias_contrarias", []),
               *resposta.get("achados", [])]
    return Resposta(
        texto="\n".join(t for t in textos if t),
        trechos_citados=trechos,
        fontes_recuperadas=[{**f, "descartada": False} for f in resposta.get("fontes_encontradas", [])]
        + [{**f, "descartada": True} for f in resposta.get("fontes_descartadas", [])],
        classificacao=resposta.get("avaliacao_evidencias"),
        diagnostico={**(diagnostico or {}), "guardrails": resposta.get("guardrails")},
    )


class TutoraRAG:
    """Protótipo de tutora RAG do pacote `rag/` (ChromaDB + Qwen via Ollama)."""

    nome = "tutora-rag"

    def __init__(self) -> None:
        from rag.store import VectorStore

        self.store = VectorStore()

    def versao(self) -> dict[str, str]:
        from rag.corpus import CORPUS
        from rag.store import MODEL
        from rag.tutor import RESPONSE_VERSION, SYSTEM_PROMPT

        return {
            "sistema": self.nome,
            "modelo_llm": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            "modelo_embeddings": MODEL,
            "versao_resposta": RESPONSE_VERSION,
            "prompt_sha256": sha256_texto(SYSTEM_PROMPT),
            "corpus_sha256": sha256_texto(CORPUS.read_text(encoding="utf-8")) if CORPUS.exists() else "ausente",
        }

    def urls_na_base(self) -> set[str]:
        """URLs indexadas como evidência, para separar "não recuperou" de "não está na base"."""
        dados = self.store.collections["evidencia"].get(include=["metadatas"])
        return {m["url"] for m in dados["metadatas"]}

    def _perguntar(self, mensagem: str, diagnostico: dict) -> dict:
        from rag.tutor import ask

        return ask(mensagem, self.store, track=False, generation_diagnostics=diagnostico)

    def responder(self, mensagem: str) -> Resposta:
        diagnostico: dict = {}
        return resposta_da_tutora(self._perguntar(mensagem, diagnostico), diagnostico)


class TutoraRAGWeb(TutoraRAG):
    """Modo usado pelo menu da tutora: pesquisa na web e complementa com a base local."""

    nome = "tutora-rag-web"

    def __init__(self) -> None:
        from rag.web import WebResearcher

        super().__init__()
        self.pesquisador = WebResearcher()

    def _perguntar(self, mensagem: str, diagnostico: dict) -> dict:
        from rag.tutor import ask_online

        return ask_online(mensagem, self.pesquisador, store=self.store, track=False)


class ModeloMLflow:
    """Modelo MLflow (pyfunc) salvo por `scripts.registrar_modelo_rag_mlflow --salvar-em`.

    Avalia o artefato entregue: a tutora no modo com busca na internet, com os guardrails, exatamente
    como empacotada (prompt, índice, corpus e código).
    """

    nome = "modelo-mlflow"
    CAMINHO_PADRAO = "models/tutora-rag-v1"

    def __init__(self, caminho: str | None = None, modo: str = "web", modelo=None) -> None:
        """`modelo` permite reaproveitar um modelo já carregado (evita uma segunda cópia na GPU)."""
        import json
        from pathlib import Path

        import mlflow

        self.caminho = Path(caminho or os.getenv("AVALIACAO_MODELO", self.CAMINHO_PADRAO)).resolve()
        self.modo = modo
        self.modelo = modelo if modelo is not None else mlflow.pyfunc.load_model(str(self.caminho))
        self.manifesto = json.loads((self.caminho / "manifest.json").read_text(encoding="utf-8"))

    def versao(self) -> dict[str, str]:
        m = self.manifesto
        return {
            "sistema": f"{self.nome} ({self.caminho.name}, modo {self.modo})",
            "modelo_llm": m["ollama_model"],
            "ollama_sha256": m["ollama_sha256"][:12],
            "modelo_embeddings": m["embedding_model"],
            "versao_resposta": m["response_version"],
            "prompt_sha256": m["contract"]["prompt_sha256"][:12],
            "corpus_sha256": m["corpus_sha256"][:12],
            "indice_sha256": m["index_sha256"][:12],
        }

    def urls_na_base(self) -> set[str]:
        import json

        corpus = self.caminho / "artifacts" / "documents.jsonl"
        linhas = corpus.read_text(encoding="utf-8").splitlines()
        return {d["url"] for d in map(json.loads, filter(None, linhas)) if d.get("kind") == "evidencia"}

    def responder(self, mensagem: str) -> Resposta:
        import json

        import pandas as pd

        saida = self.modelo.predict(pd.DataFrame([{"pergunta": mensagem, "modo": self.modo}]))
        return resposta_da_tutora(json.loads(saida[0]))


SISTEMAS = {TutoraRAG.nome: TutoraRAG, TutoraRAGWeb.nome: TutoraRAGWeb, ModeloMLflow.nome: ModeloMLflow}

