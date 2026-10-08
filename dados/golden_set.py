"""Gera o golden set a partir da planilha de anotação.

Lê a aba principal de anotação e a aba de gabarito da rodada de concordância, normaliza os valores,
valida cada item contra as regras do guia de anotação (`references/guia-anotacao-golden-set.md`),
calcula o kappa de Cohen entre os dois anotadores da rodada de concordância e grava:

- `data/processed/golden_set_v1.jsonl`: um item por linha;
- `reports/golden-set-v1.md`: composição, concordância e pendências.

Uso: python -m dados.golden_set [planilha.xlsx]
"""

import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from dados.caminhos import PROCESSED, RAW, REPORTS

VERSAO = "v1"
PLANILHA = RAW / "golden_set" / "golden_set_v1_anotacao.xlsx"
ABA_PRINCIPAL = "anotacao"
# A rodada de concordância tem duas abas; a primeira fornece o gabarito, as duas entram no kappa.
ABAS_CONCORDANCIA = ("av_FilipeLacerda", "av_FelipeRodrigues")

NATUREZAS = {"fato verificavel": "fato verificável", "opiniao": "opinião", "previsao": "previsão",
             "satira": "sátira", "fora de contexto": "fora de contexto"}
TEMAS = {"saude": "saúde", "politica": "política", "economia": "economia", "ciencia": "ciência", "outro": "outro"}
TECNICAS = {
    "urgencia": "tec_urgencia",
    "pedido_compartilhamento": "tec_pedido_compartilhamento",
    "autoridade_anonima": "tec_autoridade_anonima",
    "apelo_emocional": "tec_apelo_emocional",
}
SINALIZACOES = ["sensivel_ao_tempo", "precisa_contexto", "tema_de_risco"]
CAMPOS_KAPPA = ["natureza_1", "tema", *TECNICAS.values(), *SINALIZACOES]
VAZIOS = {"", "-", "nan", "none", "nao se aplica"}


def chave(valor) -> str | None:
    """Forma comparável de um valor da planilha: minúsculas, sem acento, com grafias comuns unificadas."""
    texto = str(valor).strip().lower()
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    if sem_acento in VAZIOS:
        return None
    return sem_acento.replace("opniao", "opiniao")


def texto_livre(valor) -> str | None:
    if chave(valor) is None:
        return None
    return " ".join(str(valor).split())


def sim_nao(valor) -> bool | None:
    return {"sim": True, "nao": False}.get(chave(valor))


def para_item(linha: pd.Series) -> tuple[dict, list[str]]:
    """Converte uma linha da planilha num item do golden set e lista as pendências encontradas."""
    pendencias = []
    afirmacoes = []
    for k in (1, 2, 3):
        afirmacao = texto_livre(linha.get(f"afirmacao_{k}"))
        natureza_bruta = chave(linha.get(f"natureza_{k}"))
        natureza = NATUREZAS.get(natureza_bruta)
        if afirmacao and not natureza:
            pendencias.append(f"afirmacao_{k} sem natureza válida ({linha.get(f'natureza_{k}')!r})")
        if natureza and not afirmacao:
            pendencias.append(f"natureza_{k} preenchida sem afirmação")
        if afirmacao:
            afirmacoes.append({"texto": afirmacao, "natureza": natureza})
    if not afirmacoes:
        pendencias.append("nenhuma afirmação")

    tema = TEMAS.get(chave(linha.get("tema")))
    if tema is None:
        pendencias.append(f"tema inválido ({linha.get('tema')!r})")

    tecnicas = {}
    for nome, coluna in TECNICAS.items():
        tecnicas[nome] = sim_nao(linha.get(coluna))
        if tecnicas[nome] is None:
            pendencias.append(f"{coluna} vazio ou inválido")
    tecnicas["outra"] = texto_livre(linha.get("tec_outra"))

    sinalizacoes = {}
    for coluna in SINALIZACOES:
        sinalizacoes[coluna] = sim_nao(linha.get(coluna))
        if sinalizacoes[coluna] is None:
            pendencias.append(f"{coluna} vazio ou inválido")

    dataset, _, id_original = str(linha["origem"]).partition(":")
    item = {
        "id": linha["id"],
        "versao": VERSAO,
        "origem": {"dataset": dataset, "id_original": id_original or None},
        "texto": " ".join(str(linha["texto"]).split()),
        "fonte_esperada": texto_livre(linha.get("fonte_esperada")),
        "afirmacoes": afirmacoes,
        "tema": tema,
        "tecnicas": tecnicas,
        **sinalizacoes,
        "anotador": texto_livre(linha.get("anotador")),
        "observacoes": texto_livre(linha.get("observacoes")),
    }
    return item, pendencias


def concordancia(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Kappa de Cohen por campo, só sobre os itens anotados pelos dois (pareados pelo `id`)."""
    comuns = a["id"][a["id"].isin(b["id"])]
    a, b = a.set_index("id").loc[comuns], b.set_index("id").loc[comuns]
    linhas = []
    for campo in CAMPOS_KAPPA:
        x, y = a[campo].map(chave), b[campo].map(chave)
        validos = x.notna() & y.notna()
        x, y = x[validos], y[validos]
        kappa = cohen_kappa_score(x, y) if len(set(x) | set(y)) > 1 else float("nan")
        linhas.append({"campo": campo, "itens": int(validos.sum()),
                       "concordancia_bruta": (x == y).mean(), "kappa": kappa, "leitura": leitura_kappa(kappa)})
    return pd.DataFrame(linhas).set_index("campo")


def leitura_kappa(kappa: float) -> str:
    """Faixas de Landis e Koch (1977)."""
    if pd.isna(kappa):
        return "indefinido"
    if kappa < 0:
        return "pior que o acaso"
    for limite, rotulo in [(0.2, "leve"), (0.4, "razoável"), (0.6, "moderada"), (0.8, "substancial")]:
        if kappa <= limite:
            return rotulo
    return "quase perfeita"


def montar(planilha: Path = PLANILHA) -> tuple[list[dict], dict[str, list[str]], pd.DataFrame]:
    """Itens do golden set, pendências por item e tabela de concordância."""
    principal = pd.read_excel(planilha, sheet_name=ABA_PRINCIPAL)
    gabarito_piloto, segundo_anotador = (pd.read_excel(planilha, sheet_name=aba) for aba in ABAS_CONCORDANCIA)

    repetidos = set(principal["id"]) & set(gabarito_piloto["id"])
    if repetidos:
        raise ValueError(f"Itens presentes na aba principal e na rodada de concordância: {sorted(repetidos)}")

    linhas = pd.concat([principal, gabarito_piloto], ignore_index=True)
    linhas = linhas[linhas["manter"].map(chave) == "sim"].sort_values("id")
    itens, pendencias = [], {}
    for _, linha in linhas.iterrows():
        item, problemas = para_item(linha)
        itens.append(item)
        if problemas:
            pendencias[item["id"]] = problemas
    return itens, pendencias, concordancia(gabarito_piloto, segundo_anotador)


def relatorio(itens: list[dict], pendencias: dict[str, list[str]], kappa: pd.DataFrame, planilha: Path) -> str:
    df = pd.json_normalize(itens)
    naturezas = pd.Series([a["natureza"] for i in itens for a in i["afirmacoes"]]).value_counts()
    tecnicas = {nome: int(df[f"tecnicas.{nome}"].fillna(False).sum()) for nome in TECNICAS}
    sinalizacoes = {c: int(df[c].fillna(False).sum()) for c in SINALIZACOES}
    linhas = [
        f"# Golden set {VERSAO}",
        "",
        f"Gerado por `python -m dados.golden_set` (`make golden-set`) a partir de `{planilha.name}`. "
        "O arquivo de dados fica em `data/processed/` e não é versionado no git.",
        "",
        "## Composição",
        "",
        f"- Itens: **{len(itens)}** ({', '.join(f'{n} de {o}' for o, n in df['origem.dataset'].value_counts().items())}).",
        f"- Afirmações anotadas: {sum(len(i['afirmacoes']) for i in itens)} "
        f"({', '.join(f'{n} {k}' for k, n in naturezas.items())}).",
        f"- Temas: {', '.join(f'{n} {k}' for k, n in df['tema'].value_counts().items())}.",
        f"- Técnicas presentes: {', '.join(f'{k} em {n}' for k, n in tecnicas.items())}.",
        f"- Sinalizações: {', '.join(f'{k} em {n}' for k, n in sinalizacoes.items())}.",
        "",
        "## Concordância entre anotadores",
        "",
        f"Rodada com as abas `{ABAS_CONCORDANCIA[0]}` (gabarito) e `{ABAS_CONCORDANCIA[1]}`, "
        f"{int(kappa['itens'].max())} itens em comum. Kappa de Cohen por campo; leitura pelas faixas de Landis e Koch (1977).",
        "",
        kappa.round(2).to_markdown(),
        "",
        "## Limitações",
        "",
        "- Os itens da rodada de concordância foram discutidos entre os anotadores antes da anotação ajustada, "
        "o que tende a elevar o kappa.",
        "- Campos em que quase todos os itens têm o mesmo valor (como `natureza_1`) podem ter kappa baixo "
        "com concordância bruta alta.",
        "- A maior parte dos itens tem um único anotador.",
    ]
    if "manual" not in set(df["origem.dataset"]):
        linhas.append("- Não há casos escritos à mão para sátira, opinião e previsão: as métricas dessas naturezas "
                      "não são confiáveis nesta versão.")
    linhas += ["", "## Pendências", ""]
    if pendencias:
        linhas += [f"- **{i}**: {'; '.join(p)}" for i, p in pendencias.items()]
    else:
        linhas.append("Nenhuma.")
    return "\n".join(linhas) + "\n"


def carregar(caminho: Path = PROCESSED / f"golden_set_{VERSAO}.jsonl") -> list[dict]:
    """Lê o golden set exportado."""
    with caminho.open(encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def main() -> None:
    planilha = Path(sys.argv[1]) if len(sys.argv) > 1 else PLANILHA
    itens, pendencias, kappa = montar(planilha)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    saida = PROCESSED / f"golden_set_{VERSAO}.jsonl"
    with saida.open("w", encoding="utf-8") as arquivo:
        for item in itens:
            arquivo.write(json.dumps(item, ensure_ascii=False) + "\n")

    texto = relatorio(itens, pendencias, kappa, planilha)
    (REPORTS / f"golden-set-{VERSAO}.md").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"{len(itens)} itens gravados em {saida}")


if __name__ == "__main__":
    main()
