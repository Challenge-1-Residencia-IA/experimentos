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
        resposta = self._perguntar(mensagem, diagnostico)
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
            diagnostico=diagnostico,
        )


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


SISTEMAS = {TutoraRAG.nome: TutoraRAG, TutoraRAGWeb.nome: TutoraRAGWeb}

