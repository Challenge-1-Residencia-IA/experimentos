"""Divisor recursivo inteligente de texto para RAG de alta qualidade."""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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
    """Divide textos longos respeitando a hierarquia semântica e sintática.
    
    Tenta quebrar em:
    1. Seções de Markdown / Títulos (##, ###)
    2. Parágrafos duplos (\\n\\n)
    3. Quebras de linha (\\n)
    4. Sentenças completas (. ! ?)
    5. Orações / pontuação intermediária (; :)
    6. Palavras (espaço)
    
    Isso impede que orações, números estatísticos ou citações fiquem partidos ao meio.
    """

    chunk_size: int = 180  # em palavras por padrão (compatível com modelos de embedding como E5)
    chunk_overlap: int = 35
    min_chunk_size: int = 25  # mínimo de palavras para evitar chunks sem conteúdo
    unit: str = "words"  # "words" ou "characters"
    separators: list[str] = field(
        default_factory=lambda: [
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
        ]
    )

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

        raw_chunks = self._recursive_split(cleaned, self.separators)
        merged = self._merge_splits(raw_chunks)

        results = []
        context_prefix = ""
        if inject_context and (title or source):
            parts = []
            if title:
                parts.append(f"Documento: {title.strip()}")
            if source:
                parts.append(f"Fonte: {source.strip()}")
            context_prefix = f"[{' | '.join(parts)}]\n"

        for piece in merged:
            piece = piece.strip()
            if self._length(piece) < self.min_chunk_size and results:
                # Se for muito pequeno, acopla ao chunk anterior se possível
                results[-1] = (results[-1] + " " + piece).strip()
                continue
            if piece:
                chunk_str = f"{context_prefix}{piece}" if context_prefix else piece
                results.append(chunk_str)

        return results if results else [cleaned]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits = []
        for s in splits:
            if not s:
                continue
            if self._length(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator=separator))
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    sub_chunks = self._recursive_split(s, new_separators)
                    final_chunks.extend(sub_chunks)

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator=separator))

        return final_chunks

    def _merge_splits(self, splits: Sequence[str], separator: str = " ") -> list[str]:
        docs = []
        current_doc: list[str] = []
        total = 0

        for s in splits:
            s_len = self._length(s)
            if total + s_len > self.chunk_size:
                if total > 0:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)
                    # Cria overlap preservando os últimos elementos
                    while total > self.chunk_overlap and current_doc:
                        removed = current_doc.pop(0)
                        total -= self._length(removed)
                current_doc = [s]
                total = s_len
            else:
                current_doc.append(s)
                total += s_len

        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs


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
