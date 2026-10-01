#!/usr/bin/env python3
"""Responde perguntas com o modelo local, opcionalmente apoiado por busca web."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
import mlflow

from amostrar_e_verificar import buscar_evidencias
from perguntas_utils import (
    formatar_evidencias, montar_mensagens_pergunta, montar_prompt,
    instrucao_correcao, validar_resposta_conversa, validar_resposta_factual,
)
from ui_terminal import cabecalho, console, tabela_fontes


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DOMINIOS_SOCIAIS = {
    "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com", "youtube.com",
}

def host_normalizado(url: str) -> str:
    host = (urlparse(str(url or "")).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def filtrar_fontes(fontes: list[dict[str, str]]) -> list[dict[str, str]]:
    """Mantém uma URL por domínio e remove resultados vazios e redes sociais."""
    urls_vistas: set[str] = set()
    dominios_vistos: set[str] = set()
    saida = []
    for fonte in fontes:
        url = str(fonte.get("url", "")).strip()
        host = host_normalizado(url)
        if not url or not host:
            continue
        if any(host == social or host.endswith(f".{social}") for social in DOMINIOS_SOCIAIS):
            continue
        url_normalizada = url.rstrip("/")
        if url_normalizada in urls_vistas or host in dominios_vistos:
            continue
        urls_vistas.add(url_normalizada)
        dominios_vistos.add(host)
        saida.append(fonte)
    return saida


def carregar_modelo(modelo_id: str, adapters_dir: Path | None):
    from mlx_lm import load

    if adapters_dir:
        if not (adapters_dir / "adapters.safetensors").exists():
            raise FileNotFoundError(f"Adapter não encontrado em '{adapters_dir}'.")
        return load(modelo_id, adapter_path=str(adapters_dir))
    return load(modelo_id)


def validar_adapter_perguntas(adapters_dir: Path | None, permitir_incompativel: bool = False) -> None:
    """Bloqueia adapters conhecidos de outras tarefas no fluxo de perguntas."""
    if not adapters_dir or permitir_incompativel:
        return
    nome = str(adapters_dir).lower()
    tarefas_incompativeis = ("evidencias", "fakebr", "classificacao", "classificação")
    if any(tarefa in nome for tarefa in tarefas_incompativeis):
        raise ValueError(
            f"O adapter '{adapters_dir}' parece ter sido treinado para classificação/verificação, "
            "não para perguntas. Use o modelo-base, um caminho contendo 'perguntas', ou passe "
            "--permitir-adapter-incompativel conscientemente."
        )


def responder(args: argparse.Namespace) -> dict:
    from mlx_lm import generate

    validar_adapter_perguntas(args.adapters_dir, args.permitir_adapter_incompativel)

    if args.modo == "conversa":
        modelo, tokenizer = carregar_modelo(args.model, args.adapters_dir)
        mensagens = montar_mensagens_pergunta(args.pergunta, "conversa", prompts_dir=args.prompts_dir)
        prompt_modelo = montar_prompt(tokenizer, mensagens)
        texto = generate(
            modelo,
            tokenizer,
            prompt=prompt_modelo,
            max_tokens=args.max_tokens,
            verbose=False,
        ).strip()
        validada = validar_resposta_conversa(texto)
        return {
            "modo": "conversa",
            "pergunta": args.pergunta,
            **validada,
            "prompt_modelo": prompt_modelo,
            "resposta_bruta_modelo": texto,
            "aviso": "Resposta gerada sem pesquisa ou validação de fontes.",
        }

    fontes_brutas = buscar_evidencias(args.pergunta, args.max_resultados * 3)
    fontes = filtrar_fontes(fontes_brutas)[:args.max_resultados]
    if len(fontes) < args.min_fontes:
        validada = {
            "status": "INCONCLUSIVA",
            "resposta": "Não foram recuperadas fontes independentes suficientes para responder.",
            "justificativa": f"Foram recuperadas {len(fontes)} fonte(s) de domínios distintos; o mínimo exigido é {args.min_fontes}.",
            "fontes_usadas": [],
            "json_valido": True,
            "citacoes_validas": True,
            "ajustes_aplicados": ["fontes_insuficientes"],
        }
        resposta_bruta = ""
        prompt_modelo = ""
    else:
        modelo, tokenizer = carregar_modelo(args.model, args.adapters_dir)
        mensagens = montar_mensagens_pergunta(
            args.pergunta, "evidencias", formatar_evidencias(fontes), args.min_fontes, args.prompts_dir
        )
        prompt_modelo = montar_prompt(tokenizer, mensagens)
        tentativas = []
        resposta_bruta = ""
        for numero in range(1, args.max_tentativas + 1):
            prompt_modelo = montar_prompt(tokenizer, mensagens)
            resposta_bruta = generate(
                modelo, tokenizer, prompt=prompt_modelo,
                max_tokens=args.max_tokens, verbose=False,
            )
            validada = validar_resposta_factual(resposta_bruta, len(fontes), args.min_fontes)
            tentativas.append({
                "numero": numero,
                "prompt_modelo": prompt_modelo,
                "resposta_bruta_modelo": resposta_bruta,
                "status_validado": validada["status"],
                "ajustes_aplicados": validada.get("ajustes_aplicados", []),
                "diagnostico_validacao": validada.get("diagnostico_validacao", {}),
            })
            if not validada.get("ajustes_aplicados"):
                break
            mensagens.extend([
                {"role": "assistant", "content": resposta_bruta},
                {"role": "user", "content": instrucao_correcao(validada, args.min_fontes)},
            ])
    return {
        "modo": "evidencias",
        "pergunta": args.pergunta,
        **validada,
        "fontes_recuperadas": fontes,
        "prompt_modelo": prompt_modelo,
        "resposta_bruta_modelo": resposta_bruta,
        "tentativas": tentativas if len(fontes) >= args.min_fontes else [],
        "aviso": "A resposta usa snippets de busca; confira os links antes de decisões importantes.",
    }


def configurar_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        if tracking_uri.startswith(("http://", "https://", "databricks")):
            client.create_experiment(experiment_name)
        else:
            client.create_experiment(
                experiment_name,
                artifact_location=(ROOT_DIR / "mlartifacts" / experiment_name).resolve().as_uri(),
            )
    mlflow.set_experiment(experiment_name)


def registrar_pergunta_mlflow(resultado: dict, args: argparse.Namespace) -> str:
    configurar_mlflow(args.tracking_uri, args.experiment_name)
    with mlflow.start_run(run_name=f"pergunta-{args.modo}") as run:
        mlflow.log_params({
            "modo": args.modo,
            "modelo": args.model,
            "adaptador": str(args.adapters_dir or ""),
            "diretorio_prompts": str(args.prompts_dir.resolve()),
            "max_tokens": args.max_tokens,
            "min_fontes": args.min_fontes,
            "max_tentativas": getattr(args, "max_tentativas", 1),
        })
        mlflow.set_tags({"tarefa": "pergunta_resposta", "biblioteca": "mlx-lm"})
        mlflow.log_metrics({
            "fontes_recuperadas": len(resultado.get("fontes_recuperadas", [])),
            "fontes_citadas": len(resultado.get("fontes_usadas", [])),
            "resposta_conclusiva": int(resultado.get("status", "RESPONDIDA") == "RESPONDIDA"),
            "json_valido": int(resultado.get("json_valido", False)),
            "citacoes_validas": int(resultado.get("citacoes_validas", True)),
            "tentativas": len(resultado.get("tentativas", [])),
        })
        mlflow.log_dict(resultado, "pergunta/resultado.json")
        mlflow.log_text(resultado["justificativa"], "pergunta/justificativa.txt")
        return run.info.run_id


def main(args: argparse.Namespace) -> None:
    if args.visual:
        cabecalho("Perguntar ao modelo local", f"Modo: {args.modo}")
        console.print(f"[bold]Pergunta:[/bold] {args.pergunta}")
    resultado = responder(args)
    resultado["mlflow_run_id"] = registrar_pergunta_mlflow(resultado, args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.visual:
        status = resultado.get("status", "RESPOSTA")
        cor = "green" if status in {"RESPONDIDA", "RESPOSTA"} else "yellow"
        console.print(f"\n[bold {cor}]{status}[/bold {cor}]\n{resultado['resposta']}")
        console.print(f"\n[bold]Justificativa:[/bold] {resultado['justificativa']}")
        if resultado.get("fontes_recuperadas"):
            tabela_fontes(resultado["fontes_recuperadas"], resultado.get("fontes_usadas", []))
        if args.output:
            console.print(f"[dim]Relatório salvo em: {args.output}[/dim]")
        console.print(f"[dim]MLflow run: {resultado['mlflow_run_id']}[/dim]")
    else:
        print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Responde perguntas com o modelo local, com ou sem evidências web.")
    parser.add_argument("pergunta", help="Pergunta enviada pelo usuário.")
    parser.add_argument("--modo", choices=("evidencias", "conversa"), default="evidencias")
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_REPO") or DEFAULT_MODEL)
    parser.add_argument("--adapters-dir", type=Path, help="Adapter opcional; sem ele, usa somente o modelo-base.")
    parser.add_argument("--max-resultados", type=int, default=5)
    parser.add_argument("--min-fontes", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--max-tentativas", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--prompts-dir", type=Path, default=ROOT_DIR / "prompts", help="Pasta com templates de prompts de perguntas.")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(ROOT_DIR / 'mlflow.db').resolve()}")
    parser.add_argument("--experiment-name", default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or "nexoverdict-perguntas")
    parser.add_argument("--permitir-adapter-incompativel", action="store_true")
    args = parser.parse_args()
    if not args.pergunta.strip():
        parser.error("A pergunta não pode ser vazia.")
    if any(valor <= 0 for valor in (args.max_resultados, args.min_fontes, args.max_tokens, args.max_tentativas)):
        parser.error("Os limites numéricos devem ser positivos.")
    if args.min_fontes > args.max_resultados:
        parser.error("--min-fontes não pode ser maior que --max-resultados.")
    try:
        validar_adapter_perguntas(args.adapters_dir, args.permitir_adapter_incompativel)
    except ValueError as erro:
        parser.error(str(erro))
    main(args)
