#!/usr/bin/env python3
"""Avalia o adaptador de verificacao no conjunto de teste nunca usado no treino."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from auditoria_evidencias import calcular_metricas_auditoria, extrair_fontes_do_prompt, montar_rastro_decisao
from evidencias_utils import ROTULOS_EVIDENCIAS, extrair_json, normalizar_veredito, validar_resposta
from ui_terminal import cabecalho, console, tabela_metricas


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
CRITERIOS_PROTO = {
    "f1_macro": 0.60,
    "taxa_json_valido": 0.95,
    "taxa_citacoes_validas": 0.95,
    "taxa_abstencao_correta": 0.60,
    "f1_minimo_por_classe": 0.50,
}


def arquivo_uri(path: Path) -> str:
    return f"sqlite:///{path.resolve()}"


def configurar_mlflow(endereco: str, nome_experimento: str) -> None:
    """Seleciona o banco do MLflow e cria o experimento quando necessário."""
    mlflow.set_tracking_uri(endereco)
    cliente = mlflow.tracking.MlflowClient()
    if cliente.get_experiment_by_name(nome_experimento) is None:
        if endereco.startswith(("http://", "https://", "databricks")):
            cliente.create_experiment(nome_experimento)
        else:
            artefatos = (ROOT_DIR / "mlartifacts" / nome_experimento).resolve().as_uri()
            cliente.create_experiment(nome_experimento, artifact_location=artefatos)
    mlflow.set_experiment(nome_experimento)


def metricas_mlflow(metricas: dict, relatorio_classes: dict, presentes: list[str]) -> dict[str, float]:
    """Traduz métricas da avaliação para nomes claros no painel do MLflow."""
    saida = {
        "acuracia": metricas["accuracy"],
        "precisao_macro": metricas["precision_macro"],
        "revocacao_macro": metricas["recall_macro"],
        "f1_macro": metricas["f1_macro"],
        "taxa_json_valido": metricas["taxa_json_valido"],
        "taxa_citacoes_validas": metricas["taxa_citacoes_validas"],
        "taxa_abstencao_correta": metricas["taxa_abstencao_correta"],
    }
    for rotulo in presentes:
        nome = rotulo.lower()
        saida[f"f1_{nome}"] = float(relatorio_classes[rotulo]["f1-score"])
        saida[f"precisao_{nome}"] = float(relatorio_classes[rotulo]["precision"])
        saida[f"revocacao_{nome}"] = float(relatorio_classes[rotulo]["recall"])
        saida[f"suporte_{nome}"] = float(relatorio_classes[rotulo]["support"])
    for nome in (
        "media_fontes_citadas",
        "media_cobertura_fontes",
        "taxa_requisito_minimo_fontes_atendido",
        "taxa_justificativa_presente",
        "media_sobreposicao_justificativa_evidencias",
        "media_sobreposicao_alegacao_evidencias",
        "media_indice_auditabilidade_heuristico",
        "taxa_veredito_alterado_pelo_validador",
    ):
        saida[nome] = float(metricas[nome])
    return saida


def avaliar_criterios(metricas_pt: dict[str, float], presentes: list[str]) -> dict:
    verificacoes = {
        "f1_macro": metricas_pt["f1_macro"] >= CRITERIOS_PROTO["f1_macro"],
        "json_valido": metricas_pt["taxa_json_valido"] >= CRITERIOS_PROTO["taxa_json_valido"],
        "citacoes_validas": metricas_pt["taxa_citacoes_validas"] >= CRITERIOS_PROTO["taxa_citacoes_validas"],
        "abstencao_correta": metricas_pt["taxa_abstencao_correta"] >= CRITERIOS_PROTO["taxa_abstencao_correta"],
    }
    for rotulo in presentes:
        verificacoes[f"f1_{rotulo.lower()}"] = (
            metricas_pt[f"f1_{rotulo.lower()}"] >= CRITERIOS_PROTO["f1_minimo_por_classe"]
        )
    return {
        "limites_prototipo": CRITERIOS_PROTO,
        "criterios_atendidos": verificacoes,
        "quantidade_atendida": sum(verificacoes.values()),
        "quantidade_total": len(verificacoes),
        "aprovado_para_prototipo_com_revisao_humana": all(verificacoes.values()),
    }


def carregar_jsonl(caminho: Path) -> list[dict]:
    registros = []
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, 1):
            try:
                item = json.loads(linha)
            except json.JSONDecodeError as erro:
                raise ValueError(f"JSON invalido em {caminho}:{numero}: {erro}") from erro
            mensagens = item.get("messages", [])
            if len(mensagens) < 3 or mensagens[-1].get("role") != "assistant":
                raise ValueError(f"Exemplo sem resposta assistant em {caminho}:{numero}")
            registros.append(item)
    return registros


def total_fontes(mensagens: list[dict]) -> int:
    texto_usuario = next((m.get("content", "") for m in mensagens if m.get("role") == "user"), "")
    indices = [int(valor) for valor in re.findall(r"(?m)^\[(\d+)\]", texto_usuario)]
    return max(indices, default=0)


def entrada_de_evidencias(mensagens: list[dict]) -> tuple[str, list[dict[str, str]]]:
    texto_usuario = next((m.get("content", "") for m in mensagens if m.get("role") == "user"), "")
    alegacao = re.search(r"(?is)ALEGACAO:\s*(.*?)\s*EVIDENCIAS:", texto_usuario)
    return (alegacao.group(1).strip() if alegacao else "", extrair_fontes_do_prompt(texto_usuario))


def main(args: argparse.Namespace) -> None:
    if not args.test_jsonl.exists():
        raise FileNotFoundError(f"Teste nao encontrado: '{args.test_jsonl}'.")
    if not (args.adapters_dir / "adapters.safetensors").exists():
        raise FileNotFoundError(f"Adaptador nao encontrado em '{args.adapters_dir}'.")

    from mlx_lm import generate, load

    dados = carregar_jsonl(args.test_jsonl)
    if args.limit:
        dados = dados[:args.limit]
    if args.visual:
        cabecalho("Avaliação do modelo de evidências", f"Amostras: {len(dados)} | Adaptador: {args.adapters_dir.name}")
        console.print("[dim]Carregando modelo local...[/dim]")
    modelo, tokenizer = load(args.model, adapter_path=str(args.adapters_dir))
    linhas = []
    progresso = Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console, disable=not args.visual,
    )
    with progresso:
        tarefa = progresso.add_task("Avaliando exemplos", total=len(dados))
        for indice, item in enumerate(dados, 1):
            mensagens = item["messages"]
            alvo = extrair_json(mensagens[-1]["content"]) or {}
            esperado = normalizar_veredito(alvo.get("veredito"))
            prompt = tokenizer.apply_chat_template(mensagens[:-1], tokenize=False, add_generation_prompt=True)
            resposta = generate(modelo, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
            qtd_fontes = total_fontes(mensagens)
            predito = validar_resposta(resposta, qtd_fontes)
            alegacao, fontes = entrada_de_evidencias(mensagens)
            indicadores = calcular_metricas_auditoria(alegacao, predito, fontes, minimo_fontes=1)
            rastro = montar_rastro_decisao(alegacao, predito, fontes, minimo_fontes=1)
            fontes_esperadas = alvo.get("fontes_usadas", []) if isinstance(alvo.get("fontes_usadas", []), list) else []
            abstencao_esperada = esperado == "INCONCLUSIVA"
            linhas.append({
            "id": indice,
            "esperado": esperado,
            "predito": predito["veredito"],
            "acertou": esperado == predito["veredito"],
            "json_valido": predito["json_valido"],
            "citacoes_validas": predito["citacoes_validas"],
            "fontes_esperadas": json.dumps(fontes_esperadas),
            "fontes_preditas": json.dumps(predito["fontes_usadas"]),
            "veredito_original_modelo": predito["veredito_original"],
            "veredito_alterado_validador": predito["veredito_alterado"],
            "ajustes_validador": json.dumps(predito["ajustes_aplicados"], ensure_ascii=False),
            "justificativa_modelo": predito["justificativa_original"],
            "rastro_decisao": json.dumps(rastro, ensure_ascii=False),
            "abstencao_esperada": abstencao_esperada,
            "abstencao_correta": (not abstencao_esperada) or predito["veredito"] == "INCONCLUSIVA",
            "resposta_bruta": resposta,
            **indicadores,
            })
            progresso.advance(tarefa)
            if not args.visual:
                print(f"[{indice}/{len(dados)}] esperado={esperado} predito={predito['veredito']}", flush=True)

    quadro = pd.DataFrame(linhas)
    y_true, y_pred = quadro["esperado"], quadro["predito"]
    presentes = [rotulo for rotulo in ROTULOS_EVIDENCIAS if rotulo in set(y_true)]
    relatorio_classes = classification_report(
        y_true, y_pred, labels=presentes, output_dict=True, zero_division=0
    )
    matriz = confusion_matrix(y_true, y_pred, labels=list(ROTULOS_EVIDENCIAS)).tolist()
    inconclusivos = quadro[quadro["abstencao_esperada"]]
    metricas = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(relatorio_classes["macro avg"]["precision"]),
        "recall_macro": float(relatorio_classes["macro avg"]["recall"]),
        "f1_macro": float(relatorio_classes["macro avg"]["f1-score"]),
        "taxa_json_valido": float(quadro["json_valido"].mean()),
        "taxa_citacoes_validas": float(quadro["citacoes_validas"].mean()),
        "taxa_abstencao_correta": float(inconclusivos["abstencao_correta"].mean()) if len(inconclusivos) else 0.0,
        "media_fontes_citadas": float(quadro["fontes_citadas"].mean()),
        "media_cobertura_fontes": float(quadro["cobertura_fontes_recuperadas"].mean()),
        "taxa_requisito_minimo_fontes_atendido": float(quadro["requisito_minimo_fontes_atendido"].mean()),
        "taxa_justificativa_presente": float(quadro["justificativa_presente"].mean()),
        "media_sobreposicao_justificativa_evidencias": float(quadro["sobreposicao_justificativa_evidencias"].mean()),
        "media_sobreposicao_alegacao_evidencias": float(quadro["sobreposicao_alegacao_evidencias"].mean()),
        "media_indice_auditabilidade_heuristico": float(quadro["indice_auditabilidade_heuristico"].mean()),
        "taxa_veredito_alterado_pelo_validador": float(quadro["veredito_alterado_pelo_validador"].mean()),
    }
    metricas_pt = metricas_mlflow(metricas, relatorio_classes, presentes)
    criterios = avaliar_criterios(metricas_pt, presentes)
    metricas_pt["proporcao_criterios_atendidos"] = criterios["quantidade_atendida"] / criterios["quantidade_total"]
    metricas_pt["aprovado_prototipo"] = float(criterios["aprovado_para_prototipo_com_revisao_humana"])
    rodada = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / f"predicoes_evidencias_{rodada}.csv"
    json_path = args.results_dir / f"metricas_evidencias_{rodada}.json"
    quadro.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "metadados": {
            "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "adapters_dir": str(args.adapters_dir.resolve()),
            "test_jsonl": str(args.test_jsonl.resolve()),
            "amostras": len(quadro),
        },
        "metricas": metricas,
        "rotulos_matriz_confusao": list(ROTULOS_EVIDENCIAS),
        "matriz_confusao": matriz,
        "por_classe": relatorio_classes,
        "criterios_qualidade": criterios,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    configurar_mlflow(args.tracking_uri, args.experiment_name)
    with mlflow.start_run(run_name=f"avaliacao-evidencias-{args.adapters_dir.name}-{rodada}"):
        mlflow.set_tags({
            "etapa": "avaliacao_evidencias",
            "adaptador": args.adapters_dir.name,
            "aprovado_prototipo": str(criterios["aprovado_para_prototipo_com_revisao_humana"]).lower(),
        })
        mlflow.log_params({
            "modelo": args.model,
            "diretorio_adaptadores": str(args.adapters_dir.resolve()),
            "conjunto_teste": str(args.test_jsonl.resolve()),
            "quantidade_amostras": len(quadro),
            "maximo_tokens": args.max_tokens,
            **{f"limite_{nome}": valor for nome, valor in CRITERIOS_PROTO.items()},
        })
        mlflow.log_metrics(metricas_pt)
        mlflow.log_artifact(str(csv_path), artifact_path="avaliacao")
        mlflow.log_artifact(str(json_path), artifact_path="avaliacao")
    if args.visual:
        tabela_metricas(metricas, "Métricas de avaliação")
        tabela = Table(title="F1 por classe", border_style="bright_blue")
        tabela.add_column("Classe", style="bold")
        tabela.add_column("Suporte", justify="right")
        tabela.add_column("F1", justify="right")
        for rotulo in presentes:
            dados_classe = relatorio_classes[rotulo]
            tabela.add_row(rotulo, str(int(dados_classe["support"])), f"{dados_classe['f1-score'] * 100:.1f}%")
        console.print(tabela)
        console.print(f"[green]✓[/green] Resultados: [link=file://{json_path}]{json_path}[/link] e {csv_path}")
    else:
        print(json.dumps(metricas, ensure_ascii=False, indent=2))
        print(f"Resultados: {json_path} e {csv_path}")


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Avalia o modelo de alegacao + evidencias.")
    parser.add_argument("--test-jsonl", type=Path, default=ROOT_DIR / "data/evidencias-por-url-v1/test.jsonl")
    parser.add_argument("--adapters-dir", type=Path, default=ROOT_DIR / "models/adapters-evidencias-por-url-v1")
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_REPO") or DEFAULT_MODEL)
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--visual", action="store_true", help="Mostra progresso e tabelas no terminal.")
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or "sift-lora-fine-tuning",
        help="Nome do experimento no MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI") or arquivo_uri(ROOT_DIR / "mlflow.db"),
        help="Endereço do servidor ou banco do MLflow.",
    )
    argumentos = parser.parse_args()
    if argumentos.max_tokens <= 0 or (argumentos.limit is not None and argumentos.limit <= 0):
        parser.error("--max-tokens e --limit devem ser positivos.")
    main(argumentos)
