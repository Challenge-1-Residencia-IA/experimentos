"""Fragmentação em intervalos literais com limite e sobreposição explícitos."""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

CHUNKING_VERSION = "literal-windows-v2-180w-30overlap"

# Expressões regulares comuns de ruído em notícias e páginas brasileiras
NOISE_PATTERNS = [
    re.compile(r"Compartilhe (?:no|pelo|via) (?:WhatsApp|Facebook|Twitter|X|Telegram|LinkedIn|e-mail)[^\n.]*", re.IGNORECASE),
    re.compile(r"Siga (?:o|a|no) [^\n.]+ nas redes sociais[^\n.]*", re.IGNORECASE),
    re.compile(r"Clique aqui para (?:se inscrever|assistir|entrar)[^\n.]*", re.IGNORECASE),
    re.compile(r"Todos os direitos reservados[^\n.]*", re.IGNORECASE),
    re.compile(r"©\s*\d{4}[^\n.]*", re.IGNORECASE),
    re.compile(r"Foto:\s*[^/\n]+/[^\n]+", re.IGNORECASE),
    re.compile(r"<!--.*?-->", re.DOTALL),
]


def clean_document_text(text: str) -> str:
    """Normaliza texto, remove caracteres nulos e ruídos comuns de internet."""
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    # Normalização unicode mantendo acentuação
    text = unicodedata.normalize("NFC", text)
    for pattern in NOISE_PATTERNS:
        text = pattern.sub("", text)
    # Colapso de múltiplos espaços e quebras excessivas
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class RecursiveTextSplitter:
    """Preserva intervalos do texto e prefere fins de frase/parágrafo.

    O nome público é mantido por compatibilidade. Frases maiores que o limite
    precisam ser divididas, mas sua pontuação e suas palavras são preservadas.
    """

    chunk_size: int = 180  # em palavras por padrão (compatível com modelos de embedding como E5)
    chunk_overlap: int = 35
    min_chunk_size: int = 25  # compatibilidade; não permite ultrapassar o limite
    unit: str = "words"  # "words" ou "characters"

    def __post_init__(self):
        if self.unit not in {"words", "characters"}:
            raise ValueError("unit deve ser words ou characters")
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Exige chunk_size > 0 e 0 <= chunk_overlap < chunk_size")

    def _length(self, text: str) -> int:
        if self.unit == "characters":
            return len(text)
        return len(text.split())

    def split_text(
        self,
        text: str,
        title: str = "",
        source: str = "",
        inject_context: bool = False,
    ) -> list[str]:
        """Divide o texto em chunks de qualidade com opção de injeção de contexto."""
        cleaned = clean_document_text(text)
        if not cleaned:
            return []

        results = []
        context_prefix = ""
        if inject_context and (title or source):
            parts = []
            if title:
                parts.append(f"Documento: {title.strip()}")
            if source:
                parts.append(f"Fonte: {source.strip()}")
            context_prefix = f"[{' | '.join(parts)}]\n"
        budget = self.chunk_size - self._length(context_prefix)
        if budget <= self.chunk_overlap:
            raise ValueError("O contexto ocupa o espaço reservado ao conteúdo do trecho")
        for piece in self._literal_windows(cleaned, budget):
            if piece:
                chunk_str = f"{context_prefix}{piece}" if context_prefix else piece
                results.append(chunk_str)

        return results if results else [cleaned]

    def _literal_windows(self, text: str, budget: int) -> list[str]:
        """Recorta intervalos contíguos sem apagar pontuação ou duplicar junções."""
        spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        if self.unit == "characters":
            spans = [(i, i + 1) for i in range(len(text))]
        result = []
        start = 0
        while start < len(spans):
            end = min(start + budget, len(spans))
            if end < len(spans) and self.unit == "words":
                floor = start + max(self.chunk_overlap + 1, budget // 2)
                for boundary in range(end, floor - 1, -1):
                    token = text[spans[boundary - 1][0]:spans[boundary - 1][1]]
                    gap = text[spans[boundary - 1][1]:spans[boundary][0]]
                    if re.search(r'[.!?][”\"\')]*$', token) or "\n\n" in gap:
                        end = boundary
                        break
            piece = text[spans[start][0]:spans[end - 1][1]].strip()
            if piece:
                result.append(piece)
            if end == len(spans):
                break
            start = end - self.chunk_overlap
        return result

def split_large_text(
    text: str,
    max_words: int = 180,
    overlap: int = 35,
    title: str = "",
    source: str = "",
    inject_context: bool = False,
) -> list[str]:
    """Função conveniente para fragmentar textos com parâmetros recomendados."""
    splitter = RecursiveTextSplitter(
        chunk_size=max_words,
        chunk_overlap=overlap,
        unit="words",
    )
    return splitter.split_text(text, title=title, source=source, inject_context=inject_context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Testa fragmentação de textos longos para RAG.")
    parser.add_argument("file", type=Path, help="Arquivo de texto a fragmentar (.txt, .md)")
    parser.add_argument("--max-words", type=int, default=180, help="Máximo de palavras por chunk")
    parser.add_argument("--overlap", type=int, default=35, help="Sobreposição de palavras")
    parser.add_argument("--inject-context", action="store_true", help="Injetar título e fonte no chunk")
    parser.add_argument("--title", type=str, default="", help="Título do documento")
    parser.add_argument("--source", type=str, default="", help="Fonte do documento")
    args = parser.parse_args()

    content = args.file.read_text(encoding="utf-8")
    title = args.title or args.file.stem
    chunks = split_large_text(
        content,
        max_words=args.max_words,
        overlap=args.overlap,
        title=title,
        source=args.source,
        inject_context=args.inject_context,
    )

    print(f"Texto original: {len(content)} caracteres, {len(content.split())} palavras")
    print(f"Chunks gerados: {len(chunks)}\n" + "=" * 60)
    for i, chunk in enumerate(chunks, 1):
        words = len(chunk.split())
        print(f"\n--- [Chunk {i}/{len(chunks)} - {words} palavras] ---")
        print(chunk)


if __name__ == "__main__":
    main()
