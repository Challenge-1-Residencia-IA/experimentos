#!/usr/bin/env python3
"""Verifica uma notícia por URL usando busca web e evidências citadas."""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import requests
from bs4 import BeautifulSoup
from mlx_lm import generate, load

from amostrar_e_verificar import (
    buscar_evidencias,
    formatar_evidencias,
)
from classificar_noticia import interpretar_resposta
from preparar_fakebr import PROMPT_SISTEMA


ROOT_DIR = Path(__file__).resolve().parents[1]
USER_AGENT = "SiftFactCheck/1.0 (+local research tool)"
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_ADAPTERS = ROOT_DIR / "models/adapters"


def extrair_texto(url: str, timeout: int, max_chars: int) -> tuple[str, str]:
    """Baixa HTML público e remove elementos que não pertencem à notícia."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("A URL deve começar com http:// ou https://.")
    resposta = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resposta.raise_for_status()
    if "html" not in resposta.headers.get("Content-Type", "").lower():
        raise ValueError("A URL não retornou uma página HTML.")

    soup = BeautifulSoup(resposta.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    titulo = soup.title.get_text(" ", strip=True) if soup.title else ""
    conteudo = soup.find("article") or soup.find("main") or soup.body
    texto = conteudo.get_text(" ", strip=True) if conteudo else ""
    texto = " ".join(texto.split())
    if len(texto) < 100:
        raise ValueError("Não foi possível extrair texto suficiente da notícia.")
    return titulo, texto[:max_chars]


def montar_prompt(tokenizer, texto: str, evidencias: str) -> str:
    mensagens = [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": f"Notícia:\n{texto}\n\nEvidências encontradas:\n{evidencias}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    return f"{PROMPT_SISTEMA}\n\nNotícia:\n{texto}\n\nEvidências:\n{evidencias}\nResposta:"


def arquivo_uri(path: Path) -> str:
    return f"sqlite:///{path.resolve()}"


def configurar_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        if tracking_uri.startswith(("http://", "https://", "databricks")):
            client.create_experiment(experiment_name)
        else:
            artifacts = (ROOT_DIR / "mlartifacts" / experiment_name).resolve().as_uri()
            client.create_experiment(experiment_name, artifact_location=artifacts)
    mlflow.set_experiment(experiment_name)


def main(args: argparse.Namespace) -> None:
    print("Baixando e extraindo a notícia...", flush=True)
    titulo, texto = extrair_texto(args.url, args.timeout, args.max_chars)
    print("Pesquisando evidências na web...", flush=True)
    consulta = titulo or texto[:300]
    fontes = buscar_evidencias(consulta, args.max_resultados)
    if not (args.adapters_dir / "adapters.safetensors").exists():
        raise FileNotFoundError(f"Adaptador não encontrado em '{args.adapters_dir}'.")
    print("Carregando o modelo fine-tunado e gerando veredito...", flush=True)
    modelo, tokenizer = load(args.model, adapter_path=str(args.adapters_dir))
    resposta = generate(
        modelo,
        tokenizer,
        prompt=montar_prompt(tokenizer, texto, formatar_evidencias(fontes)),
        max_tokens=8,
        verbose=False,
    )
    resultado = {
        "url_analisada": args.url,
        "titulo": titulo,
        "consulta_de_evidencias": consulta,
        "veredito_modelo_fine_tunado": interpretar_resposta(resposta),
        "resposta_bruta_modelo": resposta,
        "fontes_recuperadas": fontes,
        "aviso": "O veredito foi produzido pelo adaptador LoRA. Ele foi treinado apenas com rótulos Fake.br, não com pares de evidências; use as fontes recuperadas para revisão humana.",
    }
    configurar_mlflow(args.tracking_uri, args.experiment_name)
    with mlflow.start_run(run_name="verificacao-link"):
        mlflow.set_tags({
            "stage": "link_verification",
            "veredito": resultado["veredito_modelo_fine_tunado"],
        })
        mlflow.log_params({
            "url": args.url,
            "model": args.model,
            "adapters_dir": str(args.adapters_dir.resolve()),
        })
        mlflow.log_metrics({
            "fontes_recuperadas": len(fontes),
            "caracteres_noticia": len(texto),
            "veredito": len(resultado)
        })
        mlflow.log_dict(resultado, "verification/resultado.json")
    texto_saida = json.dumps(resultado, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(texto_saida, encoding="utf-8")
        print(f"Resultado salvo em: {args.output}")
    else:
        print(texto_saida)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai uma notícia por URL e a verifica com evidências web.")
    parser.add_argument("url", help="URL pública da notícia.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapters-dir", type=Path, default=DEFAULT_ADAPTERS)
    parser.add_argument("--max-resultados", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, help="Salva o relatório JSON neste caminho.")
    parser.add_argument("--experiment-name", default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or "fakebr-verificacao")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI") or arquivo_uri(ROOT_DIR / "mlflow.db"))
    args = parser.parse_args()
    if args.max_resultados <= 0 or args.max_chars <= 0 or args.timeout <= 0:
        parser.error("--max-resultados, --max-chars e --timeout devem ser positivos.")
    main(args)
