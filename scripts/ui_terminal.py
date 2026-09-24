"""Componentes Rich reutilizáveis para execuções visuais no terminal."""

from __future__ import annotations

from typing import Iterable, Mapping

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console()
CORES_VEREDITO = {
    "FALSA": "bold red",
    "VERDADEIRA": "bold green",
    "FORA_DE_CONTEXTO": "bold yellow",
    "INCONCLUSIVA": "bold cyan",
    "INDEFINIDO": "bold dim",
}


def cabecalho(titulo: str, subtitulo: str = "") -> None:
    texto = Text(titulo, style="bold white")
    if subtitulo:
        texto.append(f"\n{subtitulo}", style="dim")
    console.print(Panel(texto, border_style="bright_blue", expand=False))


def painel_veredito(veredito: str, justificativa: str = "", titulo: str = "Resultado") -> None:
    estilo = CORES_VEREDITO.get(veredito, "bold white")
    texto = Text(veredito, style=estilo, justify="center")
    if justificativa:
        texto.append(f"\n\n{justificativa}", style="white")
    console.print(Panel(texto, title=titulo, border_style=estilo, expand=False))


def tabela_metricas(metricas: Mapping[str, float], titulo: str = "Métricas") -> None:
    tabela = Table(title=titulo, border_style="bright_blue")
    tabela.add_column("Métrica", style="bold")
    tabela.add_column("Valor", justify="right")
    for nome, valor in metricas.items():
        exibicao = f"{valor * 100:.1f}%" if isinstance(valor, float) and 0 <= valor <= 1 else str(valor)
        tabela.add_row(nome.replace("_", " "), exibicao)
    console.print(tabela)


def tabela_fontes(fontes: Iterable[dict[str, str]], usadas: Iterable[int] = ()) -> None:
    usadas_set = set(usadas)
    tabela = Table(title="Fontes recuperadas", border_style="bright_blue", show_lines=True)
    tabela.add_column("#", justify="right", width=4)
    tabela.add_column("Usada", justify="center", width=7)
    tabela.add_column("Título", max_width=52)
    tabela.add_column("Domínio", max_width=32)
    for indice, fonte in enumerate(fontes, 1):
        url = fonte.get("url", "")
        dominio = url.split("/")[2] if "://" in url else url
        tabela.add_row(
            str(indice),
            "✓" if indice in usadas_set else "",
            fonte.get("titulo", "") or "Sem título",
            dominio,
        )
    console.print(tabela)
