#!/usr/bin/env python3
"""Verifica uma notícia por URL usando busca web e evidências citadas."""

import argparse
import ipaddress
import json
import os
import re
import socket
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import mlflow
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from amostrar_e_verificar import (
    buscar_evidencias,
    formatar_evidencias,
)
from auditoria_evidencias import calcular_metricas_auditoria, montar_rastro_decisao
from evidencias_utils import montar_prompt_evidencias, validar_resposta
from ui_terminal import cabecalho, console, painel_veredito, tabela_fontes, tabela_metricas


ROOT_DIR = Path(__file__).resolve().parents[1]
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_ADAPTERS = ROOT_DIR / "models/adapters-evidencias"
DOMINIOS_SOCIAIS = {"facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com", "youtube.com"}
DOMINIOS_PRIORITARIOS = (
    ".gov.br", ".jus.br", ".leg.br", ".edu.br", "who.int", "reuters.com",
    "apnews.com", "agenciabrasil.ebc.com.br", "fiocruz.br", "scielo.br",
    "aosfatos.org", "lupa.uol.com.br", "projetocomprova.com.br",
)


def extrair_texto_da_busca(url: str, max_chars: int) -> tuple[str, str, str]:
    """Recupera título e trecho indexados quando o site bloqueia acesso direto."""
    parsed = urlparse(url)
    termos_slug = re.sub(r"[-_/]+", " ", unquote(parsed.path)).strip()
    consulta = f'site:{parsed.netloc} "{termos_slug[-240:]}"'
    resultados = buscar_evidencias(consulta, 10)
    alvo = url_normalizada(url)
    candidatos = sorted(
        (item for item in resultados if host_normalizado(item.get("url", "")) == host_normalizado(url)),
        key=lambda item: url_normalizada(item.get("url", "")) != alvo,
    )
    for item in candidatos:
        titulo = str(item.get("titulo", "")).strip()
        trecho = str(item.get("trecho", "")).strip()
        texto = " ".join(parte for parte in (titulo, trecho) if parte)
        if titulo and len(texto) >= 80:
            return titulo, texto[:max_chars], "indice_de_busca"

    slug = Path(unquote(parsed.path)).stem
    titulo_slug = re.sub(r"[-_]+", " ", slug).strip().capitalize()
    if len(titulo_slug) >= 40 and len(titulo_slug.split()) >= 5:
        return titulo_slug, titulo_slug[:max_chars], "titulo_da_url"
    raise ValueError(
        "A página bloqueou o acesso, a busca não retornou trecho suficiente e o endereço não contém um título legível."
    )


def extrair_texto(url: str, timeout: int, max_chars: int) -> tuple[str, str, str]:
    """Baixa o HTML ou usa título/trecho indexados em caso de bloqueio."""
    validar_url_publica(url)
    try:
        resposta = requisitar_url_publica(url, timeout)
        resposta.raise_for_status()
    except requests.RequestException as erro_direto:
        try:
            return extrair_texto_da_busca(url, max_chars)
        except (ValueError, RuntimeError) as erro_busca:
            codigo = getattr(getattr(erro_direto, "response", None), "status_code", None)
            detalhe = f"HTTP {codigo}" if codigo else str(erro_direto)
            raise RuntimeError(
                f"Não foi possível acessar a página diretamente ({detalhe}) nem recuperar seu trecho pela busca. "
                f"Detalhe da busca: {erro_busca}"
            ) from erro_direto
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
        return extrair_texto_da_busca(url, max_chars)
    return titulo, texto[:max_chars], "html_direto"


def validar_url_publica(url: str) -> None:
    """Recusa destinos locais, privados e portas não web antes de fazer a requisição."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("A URL deve começar com http:// ou https:// e conter um host.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs com credenciais embutidas não são aceitas.")
    try:
        porta = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as erro:
        raise ValueError("A URL contém uma porta inválida.") from erro
    porta_padrao = 443 if parsed.scheme == "https" else 80
    if porta != porta_padrao:
        raise ValueError("Somente as portas padrão HTTP (80) e HTTPS (443) são aceitas.")

    host = parsed.hostname.rstrip(".")
    try:
        enderecos = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            enderecos = {
                ipaddress.ip_address(info[4][0])
                for info in socket.getaddrinfo(host, porta, type=socket.SOCK_STREAM)
            }
        except (OSError, ValueError) as erro:
            raise ValueError("Não foi possível validar o host da URL.") from erro
    if not enderecos or any(not endereco.is_global for endereco in enderecos):
        raise ValueError("A URL precisa apontar somente para endereços IP públicos.")


def requisitar_url_publica(url: str, timeout: int, max_redirecionamentos: int = 5) -> requests.Response:
    """Segue redirecionamentos manualmente, validando cada destino."""
    atual = url
    for _ in range(max_redirecionamentos + 1):
        validar_url_publica(atual)
        resposta = requests.get(
            atual,
            timeout=timeout,
            headers=BROWSER_HEADERS,
            allow_redirects=False,
        )
        status_code = getattr(resposta, "status_code", None)
        if not isinstance(status_code, int) or not 300 <= status_code < 400:
            return resposta
        destino = resposta.headers.get("Location")
        if not destino:
            return resposta
        atual = urljoin(atual, destino)
    raise requests.TooManyRedirects(f"A URL excedeu {max_redirecionamentos} redirecionamentos.")


def host_normalizado(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def url_normalizada(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme.lower()}://{host_normalizado(url)}{parsed.path.rstrip('/')}"


def filtrar_fontes_com_auditoria(
    fontes: list[dict[str, str]], url_original: str
) -> tuple[list[dict[str, str]], dict]:
    """Remove fontes inadequadas e registra por que cada resultado foi descartado."""
    original = url_normalizada(url_original)
    host_original = host_normalizado(url_original)
    vistas, dominios, saida = set(), set(), []
    descartes: Counter[str] = Counter()
    for fonte in fontes:
        url = fonte.get("url", "").strip()
        host = host_normalizado(url)
        normalizada = url_normalizada(url) if url else ""
        motivo = None
        if not url or not host:
            motivo = "sem_url_valida"
        elif normalizada == original or host == host_original:
            motivo = "mesmo_site_da_noticia"
        elif any(host == social or host.endswith(f".{social}") for social in DOMINIOS_SOCIAIS):
            motivo = "rede_social"
        elif normalizada in vistas:
            motivo = "url_repetida"
        elif host in dominios:
            motivo = "dominio_repetido"
        if motivo:
            descartes[motivo] += 1
            continue
        vistas.add(normalizada)
        dominios.add(host)
        saida.append(fonte)
    def prioridade(fonte: dict[str, str]) -> int:
        host = host_normalizado(fonte.get("url", ""))
        for indice, dominio in enumerate(DOMINIOS_PRIORITARIOS):
            dominio_limpo = dominio.lstrip(".")
            if host == dominio_limpo or host.endswith(f".{dominio_limpo}"):
                return indice
        return len(DOMINIOS_PRIORITARIOS)

    ordenadas = sorted(saida, key=prioridade)
    return ordenadas, {
        "resultados_brutos": len(fontes),
        "fontes_aceitas_apos_filtros": len(ordenadas),
        "fontes_descartadas": sum(descartes.values()),
        "descartes_por_motivo": dict(descartes),
    }


def filtrar_fontes(fontes: list[dict[str, str]], url_original: str) -> list[dict[str, str]]:
    """Compatibilidade: retorna apenas as fontes aceitas."""
    return filtrar_fontes_com_auditoria(fontes, url_original)[0]


def extrair_alegacao(titulo: str, texto: str, max_chars: int = 500) -> str:
    """Usa titulo e primeiro periodo como representacao curta da alegacao."""
    titulo_limpo = re.split(r"\s+[|–—-]\s+", titulo, maxsplit=1)[0].strip()
    primeira_frase = re.split(r"(?<=[.!?])\s+", texto.strip(), maxsplit=1)[0]
    partes = [parte for parte in (titulo_limpo, primeira_frase) if parte]
    alegacao = " ".join(dict.fromkeys(partes))
    return alegacao[:max_chars].rsplit(" ", 1)[0] if len(alegacao) > max_chars else alegacao


def montar_auditoria(verificacao: dict, fontes: list[dict[str, str]], minimo_fontes: int) -> dict:
    """Explica a regra aplicada sem inventar cadeia de raciocinio do modelo."""
    citadas = []
    for indice in verificacao["fontes_usadas"]:
        fonte = fontes[indice - 1]
        citadas.append({
            "indice": indice,
            "titulo": fonte.get("titulo", ""),
            "url": fonte.get("url", ""),
            "trecho": fonte.get("trecho", ""),
        })
    veredito = verificacao["veredito"]
    requisito_especifico = {
        "FALSA": "A fonte citada deve contradizer objetivamente a alegacao.",
        "VERDADEIRA": "A fonte citada deve confirmar objetivamente a alegacao.",
        "FORA_DE_CONTEXTO": "A fonte citada deve mostrar o contexto original e a alteracao de data, local ou sentido.",
        "INCONCLUSIVA": "As fontes sao ausentes, insuficientes, conflitantes ou nao permitem concluir.",
    }[veredito]
    alertas = []
    if verificacao.get("fontes_insuficientes"):
        alertas.append(
            "O modelo citou menos fontes do que o mínimo configurado; a resposta foi rebaixada para INCONCLUSIVA."
        )
    if verificacao.get("resposta_conclusiva_sem_citacoes_validas"):
        alertas.append(
            "A resposta citou índices inválidos; o veredito foi rebaixado para INCONCLUSIVA."
        )
    if len(fontes) < minimo_fontes:
        alertas.append("Ha menos fontes independentes que o minimo configurado.")
    alertas.append(
        "A justificativa e a relacao entre alegacao e fonte sao geradas pelo modelo; "
        "este relatorio nao prova automaticamente que a citacao sustenta o veredito."
    )
    return {
        "regra_geral": (
            f"Um veredito conclusivo exige pelo menos {minimo_fontes} fontes independentes citadas e válidas."
        ),
        "requisito_para_o_veredito_emitido": requisito_especifico,
        "minimo_fontes_independentes_configurado": minimo_fontes,
        "fontes_independentes_recuperadas": len(fontes),
        "justificativa_gerada_pelo_modelo": verificacao["justificativa"],
        "fontes_que_o_modelo_disse_ter_usado": citadas,
        "alertas": alertas,
    }


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
    if args.visual:
        cabecalho("Verificação de notícia por link", f"Adaptador: {args.adapters_dir.name}")
        console.print("[cyan]1/4[/cyan] Extraindo conteúdo da página...")
    if not args.visual:
        print("Baixando e extraindo a notícia...", flush=True)
    titulo, texto, metodo_extracao = extrair_texto(args.url, args.timeout, args.max_chars)
    alegacao = extrair_alegacao(titulo, texto, args.max_claim_chars)
    if args.visual:
        if metodo_extracao == "indice_de_busca":
            console.print(
                "[yellow]O site bloqueou a leitura direta. A alegação foi extraída do título e do trecho indexados na busca.[/yellow]"
            )
        elif metodo_extracao == "titulo_da_url":
            console.print(
                "[bold yellow]O site e a busca bloquearam a leitura. A alegação foi inferida somente do título presente na URL.[/bold yellow]"
            )
        console.print(f"[bold]Alegação analisada:[/bold] {alegacao}")
        console.print("[cyan]2/4[/cyan] Buscando fontes independentes...")
    if not args.visual:
        print("Pesquisando evidências na web...", flush=True)
    consulta = alegacao
    fontes_brutas = buscar_evidencias(consulta, args.max_resultados * 3)
    fontes_filtradas, auditoria_recuperacao = filtrar_fontes_com_auditoria(fontes_brutas, args.url)
    fontes = fontes_filtradas[:args.max_resultados]
    auditoria_recuperacao["fontes_apresentadas_ao_modelo"] = len(fontes)
    auditoria_recuperacao["dominios_apresentados"] = [host_normalizado(fonte.get("url", "")) for fonte in fontes]
    if not (args.adapters_dir / "adapters.safetensors").exists():
        raise FileNotFoundError(f"Adaptador não encontrado em '{args.adapters_dir}'.")
    if len(fontes) < args.min_fontes_independentes:
        resposta = json.dumps({
            "veredito": "INCONCLUSIVA",
            "justificativa": "Nao foram recuperadas fontes independentes suficientes.",
            "fontes_usadas": [],
        }, ensure_ascii=False)
    else:
        if args.visual:
            console.print("[cyan]3/4[/cyan] Carregando adaptador e gerando veredito...")
        if not args.visual:
            print("Carregando o modelo fine-tunado e gerando veredito...", flush=True)
        from mlx_lm import generate, load

        modelo, tokenizer = load(args.model, adapter_path=str(args.adapters_dir))
        resposta = generate(
            modelo,
            tokenizer,
            prompt=montar_prompt_evidencias(
                tokenizer,
                alegacao,
                formatar_evidencias(fontes),
                minimo_fontes_conclusivas=args.min_fontes_independentes,
            ),
            max_tokens=args.max_tokens,
            verbose=False,
        )
    verificacao = validar_resposta(
        resposta,
        len(fontes),
        minimo_fontes_conclusivas=args.min_fontes_independentes,
    )
    auditoria = montar_auditoria(verificacao, fontes, args.min_fontes_independentes)
    metricas_auditoria = calcular_metricas_auditoria(
        alegacao,
        verificacao,
        fontes,
        args.min_fontes_independentes,
        metodo_extracao,
    )
    metricas_auditoria["resultados_brutos_da_busca"] = len(fontes_brutas)
    metricas_auditoria["fontes_descartadas_pelos_filtros"] = auditoria_recuperacao["fontes_descartadas"]
    metricas_auditoria["fontes_nao_apresentadas_por_limite"] = max(0, len(fontes_filtradas) - len(fontes))
    metricas_auditoria["taxa_descarte_da_busca"] = (
        auditoria_recuperacao["fontes_descartadas"] / len(fontes_brutas) if fontes_brutas else 0.0
    )
    rastro_decisao = montar_rastro_decisao(
        alegacao,
        verificacao,
        fontes,
        args.min_fontes_independentes,
        metodo_extracao,
        auditoria_recuperacao,
    )
    resultado = {
        "url_analisada": args.url,
        "titulo": titulo,
        "metodo_extracao": metodo_extracao,
        "alegacao_analisada": alegacao,
        "consulta_de_evidencias": consulta,
        "veredito_modelo_fine_tunado": verificacao["veredito"],
        "justificativa": verificacao["justificativa"],
        "fontes_usadas": verificacao["fontes_usadas"],
        "resposta_json_valida": verificacao["json_valido"],
        "citacoes_validas": verificacao["citacoes_validas"],
        "fontes_suficientes_para_conclusao": not verificacao["fontes_insuficientes"],
        "resposta_bruta_modelo": resposta,
        "fontes_recuperadas": fontes,
        "fontes_descartadas": auditoria_recuperacao["fontes_descartadas"],
        "auditoria_da_recuperacao": auditoria_recuperacao,
        "auditoria_do_veredito": auditoria,
        "metricas_de_auditoria": metricas_auditoria,
        "rastro_da_decisao": rastro_decisao,
        "aviso": "Veredito automatizado baseado nos snippets recuperados; mantenha revisao humana para decisoes importantes.",
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
            "metodo_extracao": metodo_extracao,
        })
        mlflow.log_metrics({
            "caracteres_noticia": len(texto),
            "minimo_fontes_exigido": args.min_fontes_independentes,
            **metricas_auditoria,
        })
        mlflow.log_dict(resultado, "verification/resultado.json")
    texto_saida = json.dumps(resultado, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(texto_saida, encoding="utf-8")
        if args.visual:
            console.print(f"[green]✓[/green] Relatório JSON salvo em: {args.output}")
        else:
            print(f"Resultado salvo em: {args.output}")
    else:
        print(texto_saida)
    if args.visual:
        console.print("[cyan]4/4[/cyan] Verificação concluída.")
        painel_veredito(
            resultado["veredito_modelo_fine_tunado"],
            resultado["justificativa"],
            titulo="Veredito com evidências",
        )
        tabela_fontes(fontes, verificacao["fontes_usadas"])
        tabela_metricas({
            "fontes_citadas": metricas_auditoria["fontes_citadas"],
            "cobertura_fontes": metricas_auditoria["cobertura_fontes_recuperadas"],
            "mínimo_atendido": metricas_auditoria["requisito_minimo_fontes_atendido"],
            "justificativa_evidências": metricas_auditoria["sobreposicao_justificativa_evidencias"],
            "fontes_prioritárias_citadas": metricas_auditoria["taxa_fontes_prioritarias_citadas"],
            "auditabilidade_heurística": metricas_auditoria["indice_auditabilidade_heuristico"],
        }, "Auditoria da resposta")
        console.print(
            f"[dim]JSON válido: {'sim' if verificacao['json_valido'] else 'não'} | "
            f"Citações válidas: {'sim' if verificacao['citacoes_validas'] else 'não'}[/dim]"
        )


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Extrai uma notícia por URL e a verifica com evidências web.")
    parser.add_argument("url", help="URL pública da notícia.")
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_REPO") or DEFAULT_MODEL)
    parser.add_argument("--adapters-dir", type=Path, default=DEFAULT_ADAPTERS)
    parser.add_argument("--max-resultados", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--max-claim-chars", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--min-fontes-independentes", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, help="Salva o relatório JSON neste caminho.")
    parser.add_argument("--visual", action="store_true", help="Mostra painel, etapas e fontes no terminal.")
    parser.add_argument("--experiment-name", default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or "fakebr-verificacao")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI") or arquivo_uri(ROOT_DIR / "mlflow.db"))
    args = parser.parse_args()
    if any(valor <= 0 for valor in (
        args.max_resultados, args.max_chars, args.max_claim_chars, args.max_tokens,
        args.min_fontes_independentes, args.timeout,
    )):
        parser.error("Os limites numericos devem ser positivos.")
    main(args)
