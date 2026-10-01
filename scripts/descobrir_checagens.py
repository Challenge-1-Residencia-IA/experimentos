"""Descobre URLs de checagens via Google Fact Check Tools API."""

import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consulta", help="Termo ou alegação a pesquisar")
    parser.add_argument("--paginas", type=int, default=1)
    parser.add_argument("--saida", type=Path, default=Path("data/candidatas_checagens.jsonl"))
    args = parser.parse_args()
    load_dotenv()
    api_key = os.getenv("FACT_CHECK_API_KEY")
    if not api_key:
        parser.error("Defina FACT_CHECK_API_KEY no ambiente")
    if not 1 <= args.paginas <= 10:
        parser.error("--paginas deve estar entre 1 e 10")
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.saida.exists():
        with args.saida.open(encoding="utf-8") as existing:
            seen = {json.loads(line)["url"] for line in existing if line.strip()}
    added = 0
    page_token = None
    with args.saida.open("a", encoding="utf-8") as output:
        for _ in range(args.paginas):
            params = {"query": args.consulta, "languageCode": "pt", "pageSize": 100, "key": api_key}
            if page_token:
                params["pageToken"] = page_token
            response = requests.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params=params,
                timeout=30,
            )
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
    print(f"{added} URLs candidatas novas; revise antes de indexar.")


if __name__ == "__main__":
    main()
