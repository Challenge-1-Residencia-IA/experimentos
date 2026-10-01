"""Descobre URLs de checagens via Google Fact Check Tools API."""

import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


def discover_claims(consulta: str, paginas: int = 1, saida: Path = Path("data/candidatas_checagens.jsonl")) -> int:
    load_dotenv()
    api_key = os.getenv("FACT_CHECK_API_KEY")
    if not api_key:
        raise ValueError("Defina FACT_CHECK_API_KEY no arquivo .env")
    if not 1 <= paginas <= 10:
        raise ValueError("paginas deve estar entre 1 e 10")
    saida.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if saida.exists():
        with saida.open(encoding="utf-8") as existing:
            seen = {json.loads(line)["url"] for line in existing if line.strip()}
    added = 0
    page_token = None
    with saida.open("a", encoding="utf-8") as output:
        for _ in range(paginas):
            params = {"query": consulta, "languageCode": "pt", "pageSize": 100, "key": api_key}
            if page_token:
                params["pageToken"] = page_token
            try:
                response = requests.get(
                    "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                    params=params,
                    timeout=30,
                )
            except requests.RequestException:
                raise RuntimeError("Falha de conexão com a API de checagens") from None
            if not response.ok:
                raise RuntimeError(f"A API de checagens retornou HTTP {response.status_code}")
            result = response.json()
            for claim in result.get("claims", []):
                for review in claim.get("claimReview", []):
                    url = review.get("url")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    added += 1
                    output.write(json.dumps({
                        "alegacao": claim.get("text", ""),
                        "url": url,
                        "titulo": review.get("title", ""),
                        "veiculo": review.get("publisher", {}).get("name", ""),
                        "avaliacao_do_veiculo": review.get("textualRating", ""),
                        "data_revisao": review.get("reviewDate", ""),
                        "status": "pendente_revisao_de_fonte_e_licenca",
                    }, ensure_ascii=False) + "\n")
            page_token = result.get("nextPageToken")
            if not page_token:
                break
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consulta", help="Termo ou alegação a pesquisar")
    parser.add_argument("--paginas", type=int, default=1)
    parser.add_argument("--saida", type=Path, default=Path("data/candidatas_checagens.jsonl"))
    args = parser.parse_args()
    added = discover_claims(args.consulta, args.paginas, args.saida)
    print(f"{added} URLs candidatas novas; revise antes de indexar.")


if __name__ == "__main__":
    main()
