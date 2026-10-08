import os
import json
import random
import re
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data/raw/full_texts"
OUTPUT_DIR = ROOT_DIR / "data/processed"
TRAIN_RATIO = 0.8

PROMPT_SISTEMA = """Você é um analista de dados estruturados para um copiloto investigativo de alfabetização midiática.
Sua tarefa é DECOMPOR a mensagem do usuário nas suas afirmações principais e classificar a sua natureza.

REGRAS DE EXTRAÇÃO (MUITO IMPORTANTE):
1. Limite-se a extrair no máximo 1 a 2 afirmações centrais por mensagem. Não fragmente o texto.
2. Não invente informações que não estão explicitamente escritas.

DEFINIÇÃO DE CATEGORIAS:
- "fato verificavel": Alegações objetivas que podem ser provadas ou refutadas (inclui notícias falsas ou verdadeiras).
- "fora de contexto": O facto ocorreu, mas a data, local ou intenção original foram alterados para enganar.
- "opiniao": Juízos de valor, crenças ou interpretações pessoais sem base em factos objetivos.
- "satira": Exageros humorísticos, ironia ou paródia evidentes que não devem ser levados à letra.
- "previsao": Especulações sobre eventos futuros.

Retorne EXCLUSIVAMENTE um objeto JSON válido.
{
  "afirmacoes": [
    {"afirmacao": "[texto da alegação principal]", "tipo": "[categoria]", "verificavel": true}
  ]
}"""

def extrair_primeira_frase(texto):
    """Heurística simples para simular a extração da afirmação central para o treino."""
    # Divide o texto no primeiro ponto final, exclamação ou interrogação
    match = re.split(r'(?<=[.!?])\s+', texto.strip())
    frase = match[0] if match else texto[:150]
    # Remove quebras de linha que podem sujar o JSON
    return frase.replace('\n', ' ').strip()

def processar_noticias(dataset_path: Path):
    amostras = []

    # Ambas as pastas (fake e true) contêm alegações factuais que podem ser verificadas.
    # O bot não dirá se é falso/verdadeiro, apenas classificará a natureza do texto.
    categorias = {"fake": "fato verificavel", "true": "fato verificavel"}

    for pasta, tipo_json in categorias.items():
        caminho_pasta = dataset_path / pasta
        if not caminho_pasta.exists():
            print(f"Aviso: Pasta {caminho_pasta} não encontrada.")
            continue

        for caminho_arquivo in caminho_pasta.iterdir():
            if caminho_arquivo.suffix == ".txt":
                with caminho_arquivo.open("r", encoding="utf-8", errors="ignore") as f:
                    conteudo = f.read().strip()

                if conteudo:
                    afirmacao_central = extrair_primeira_frase(conteudo)

                    # Cria o objeto JSON que o modelo DEVE aprender a gerar
                    resposta_alvo = {
                        "afirmacoes": [
                            {
                                "afirmacao": afirmacao_central,
                                "tipo": tipo_json,
                                "verificavel": True
                            }
                        ]
                    }

                    # Formato nativo OpenAl/HuggingFace/MLX
                    item = {
                        "messages": [
                            {"role": "system", "content": PROMPT_SISTEMA},
                            {"role": "user", "content": f"Mensagem a ser analisada:\n{conteudo}"},
                            {"role": "assistant", "content": json.dumps(resposta_alvo, ensure_ascii=False, indent=2)}
                        ]
                    }
                    amostras.append(item)

    return amostras

def salvar_jsonl(dados, caminho_arquivo: Path):
    with caminho_arquivo.open("w", encoding="utf-8") as f:
        for item in dados:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara JSONL para fine-tuning de extração de afirmações.")
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio deve estar entre 0 e 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dados = processar_noticias(args.dataset_path)
    random.seed(42)
    random.shuffle(dados)

    limite = int(len(dados) * args.train_ratio)
    dados_treino = dados[:limite]
    dados_validacao = dados[limite:]

    salvar_jsonl(dados_treino, args.output_dir / "train.jsonl")
    salvar_jsonl(dados_validacao, args.output_dir / "valid.jsonl")

    print(f"Dataset de treino gerado com sucesso em '{args.output_dir}'!")
    print(f"- Treino: {len(dados_treino)} exemplos")
    print(f"- Validação: {len(dados_validacao)} exemplos")
