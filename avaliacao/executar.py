"""Roda um sistema de IA sobre o golden set e gera o boletim da avaliação.

Saídas de cada execução (identificada por data e hora):

- `reports/avaliacoes/<id>.md`: boletim agregado por requisito (versionado no git);
- `results/avaliacoes/<id>.json`: resposta e notas de cada item (contém textos do dataset; fora do git);
- uma run no MLflow (experimento `avaliacao-golden-set`), com versão do sistema, métricas e artefatos.

Uso: python -m avaliacao.executar [--sistema tutora-rag] [--limite N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from avaliacao.metricas import agregar, corrigir, url_canonica
from avaliacao.sistemas import SISTEMAS, sha256_texto
from dados import golden_set
from dados.caminhos import RAIZ, REPORTS

RESULTADOS = RAIZ / "results" / "avaliacoes"
BOLETINS = REPORTS / "avaliacoes"
EXPERIMENTO_MLFLOW = "avaliacao-golden-set"


def revisao_git() -> str:
    saida = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True, text=True)
    return saida.stdout.strip() or "desconhecida"


def avaliar(sistema, itens: list[dict]) -> list[dict]:
    resultados = []
    for numero, item in enumerate(itens, 1):
        inicio = time.perf_counter()
        resultado = {"id": item["id"], "erro": None, "resposta": None, "notas": None}
        try:
            resposta = sistema.responder(item["texto"])
            resultado["resposta"] = asdict(resposta)
            resultado["notas"] = corrigir(item, resposta)
        except Exception as erro:  # um item com erro não interrompe a avaliação
            resultado["erro"] = f"{type(erro).__name__}: {erro}"
        resultado["latencia_s"] = round(time.perf_counter() - inicio, 2)
        resultados.append(resultado)
        estado = resultado["erro"] or (resultado["resposta"] or {}).get("classificacao")
        print(f"[{numero}/{len(itens)}] {item['id']} {resultado['latencia_s']:.1f}s {estado}", flush=True)
    return resultados


def boletim(id_execucao: str, versao: dict, itens: list[dict], resultados: list[dict],
            tabela: list[dict], cobertura_base: dict) -> str:
    latencias = pd.Series([r["latencia_s"] for r in resultados])
    erros = [r for r in resultados if r["erro"]]
    classificacoes = pd.Series([(r["resposta"] or {}).get("classificacao") for r in resultados]).value_counts()
    medidas = [l for l in tabela if l["taxa"] is not None]
    nao_medidas = [l for l in tabela if l["taxa"] is None]
    linhas = [
        f"# Avaliação {id_execucao}",
        "",
        "## Versão avaliada",
        "",
        "| Item | Valor |",
        "|---|---|",
        *[f"| {k} | `{v}` |" for k, v in versao.items()],
        "",
        "## Resultado por requisito",
        "",
        "| Requisito | Métrica | Itens | Acertos | Taxa |",
        "|---|---|---|---|---|",
        *[f"| {l['requisito']} | {l['descricao']} | {l['itens']} | {l['acertos']} | {l['taxa']:.0%} |" for l in medidas],
        "",
        "**Não medido** (o sistema não produz o campo necessário): "
        + (", ".join(f"{l['requisito']} {l['descricao'].lower()}" for l in nao_medidas) or "nenhum") + ".",
        "",
        "## Operação",
        "",
        f"- Itens avaliados: {len(resultados)}; com erro: {len(erros)}.",
        f"- Latência por item: mediana {latencias.median():.1f} s, p95 {latencias.quantile(0.95):.1f} s (RNF-05).",
        f"- Classificação das evidências: {', '.join(f'{k} {v}' for k, v in classificacoes.items()) or '—'}.",
        f"- Fonte esperada presente na base indexada: {cobertura_base['na_base']} de {cobertura_base['total']} itens.",
    ]
    if erros:
        linhas += ["", "## Erros", "", *[f"- {r['id']}: {r['erro']}" for r in erros]]
    return "\n".join(linhas) + "\n"


def registrar_mlflow(id_execucao: str, versao: dict, tabela: list[dict], resultados: list[dict],
                     caminho_boletim: Path, caminho_detalhes: Path) -> str:
    import mlflow

    from rag.dagshub_tracking import tracking_uri

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
    with mlflow.start_run(run_name=id_execucao) as run:
        mlflow.log_params(versao)
        mlflow.log_metrics({f"{l['metrica']}": l["taxa"] for l in tabela if l["taxa"] is not None})
        latencias = pd.Series([r["latencia_s"] for r in resultados])
        mlflow.log_metrics({"latencia_mediana_s": latencias.median(), "latencia_p95_s": latencias.quantile(0.95),
                            "taxa_erros": sum(bool(r["erro"]) for r in resultados) / len(resultados)})
        mlflow.log_artifact(str(caminho_boletim))
        mlflow.log_artifact(str(caminho_detalhes))
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sistema", choices=sorted(SISTEMAS), default="tutora-rag")
    parser.add_argument("--limite", type=int, help="Avaliar só os N primeiros itens (teste rápido)")
    parser.add_argument("--sem-mlflow", action="store_true", help="Não registrar a execução no MLflow")
    args = parser.parse_args()

    load_dotenv(RAIZ / ".env")
    caminho_gs = golden_set.PROCESSED / f"golden_set_{golden_set.VERSAO}.jsonl"
    itens = golden_set.carregar(caminho_gs)[: args.limite]
    sistema = SISTEMAS[args.sistema]()
    versao = {
        **sistema.versao(),
        "golden_set": f"{golden_set.VERSAO} ({len(itens)} itens)",
        "golden_set_sha256": sha256_texto(caminho_gs.read_text(encoding="utf-8")),
        "git": revisao_git(),
    }
    id_execucao = datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_{args.sistema}"

    resultados = avaliar(sistema, itens)
    tabela = agregar(resultados)
    base = {url_canonica(u) for u in sistema.urls_na_base()}
    cobertura = {"total": len(itens), "na_base": sum(url_canonica(i.get("fonte_esperada")) in base for i in itens)}

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    BOLETINS.mkdir(parents=True, exist_ok=True)
    caminho_detalhes = RESULTADOS / f"{id_execucao}.json"
    caminho_detalhes.write_text(json.dumps({"versao": versao, "metricas": tabela, "itens": resultados},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
    texto = boletim(id_execucao, versao, itens, resultados, tabela, cobertura)
    caminho_boletim = BOLETINS / f"{id_execucao}.md"
    caminho_boletim.write_text(texto, encoding="utf-8")
    print("\n" + texto)
    if not args.sem_mlflow:
        run_id = registrar_mlflow(id_execucao, versao, tabela, resultados, caminho_boletim, caminho_detalhes)
        print(f"MLflow: experimento {EXPERIMENTO_MLFLOW}, run {run_id}")
    print(f"Boletim: {caminho_boletim}\nDetalhes: {caminho_detalhes}")


if __name__ == "__main__":
    main()
