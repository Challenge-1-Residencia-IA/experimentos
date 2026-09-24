#!/usr/bin/env python3
"""Classifica uma notícia com um adaptador LoRA já treinado."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from mlx_lm import generate, load

from preparar_fakebr import PROMPT_SISTEMA
from ui_terminal import cabecalho, console, painel_veredito


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_ADAPTERS = ROOT_DIR / "models/adapters"


def montar_prompt(tokenizer, texto: str) -> str:
    mensagens = [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": f"Notícia:\n{texto}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    return f"{PROMPT_SISTEMA}\n\nNotícia:\n{texto}\nResposta:"


def interpretar_resposta(resposta: str) -> str:
    texto = resposta.upper()
    if "VERDADEIRA" in texto or "TRUE" in texto:
        return "VERDADEIRA"
    if "FALSA" in texto or "FAKE" in texto:
        return "FALSA"
    return "INDEFINIDO"


def main(args: argparse.Namespace) -> None:
    if not (args.adapters_dir / "adapters.safetensors").exists():
        raise FileNotFoundError(f"Adaptador não encontrado em '{args.adapters_dir}'.")

    texto = args.texto or args.arquivo.read_text(encoding="utf-8")
    if args.visual:
        cabecalho("Classificação de notícia", f"Adaptador: {args.adapters_dir.name}")
        console.print(f"[dim]Analisando {len(texto):,} caracteres...[/dim]")
    modelo, tokenizer = load(args.model, adapter_path=str(args.adapters_dir))
    resposta = generate(modelo, tokenizer, prompt=montar_prompt(tokenizer, texto), max_tokens=8, verbose=False)
    veredito = interpretar_resposta(resposta)
    if args.visual:
        painel_veredito(veredito, titulo="Classificação do modelo")
    else:
        print(veredito)


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Classifica uma notícia sem executar novo fine-tuning.")
    entrada = parser.add_mutually_exclusive_group(required=True)
    entrada.add_argument("--texto", help="Texto da notícia a classificar.")
    entrada.add_argument("--arquivo", type=Path, help="Arquivo UTF-8 contendo a notícia.")
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_REPO") or DEFAULT_MODEL)
    parser.add_argument("--adapters-dir", type=Path, default=DEFAULT_ADAPTERS)
    parser.add_argument("--visual", action="store_true", help="Mostra um painel visual no terminal.")
    main(parser.parse_args())
