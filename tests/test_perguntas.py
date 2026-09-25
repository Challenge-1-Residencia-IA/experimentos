import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from perguntas_utils import (
    DEFAULT_PROMPTS_DIR, montar_mensagens_pergunta, validar_resposta_conversa,
    validar_resposta_factual,
)
from preparar_dataset_perguntas import exemplo_leitura, exemplo_revisado, preparar
from responder_pergunta import validar_adapter_perguntas


class ContratoPerguntasTest(unittest.TestCase):
    def test_prompts_sao_lidos_de_arquivos_customizados(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            for origem in DEFAULT_PROMPTS_DIR.glob("perguntas_*.txt"):
                (destino / origem.name).write_text(origem.read_text(encoding="utf-8"), encoding="utf-8")
            arquivo = destino / "perguntas_conversa_sistema.txt"
            arquivo.write_text("Instrução personalizada para conversa.", encoding="utf-8")
            mensagens = montar_mensagens_pergunta("Olá", "conversa", prompts_dir=destino)
            self.assertEqual(mensagens[0]["content"], "Instrução personalizada para conversa.")
            self.assertEqual(mensagens[1]["content"], "Olá")

    def test_template_exige_marcadores(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            (destino / "perguntas_conversa_sistema.txt").write_text("Sistema", encoding="utf-8")
            (destino / "perguntas_conversa_usuario.txt").write_text("Sem marcador", encoding="utf-8")
            with self.assertRaises(ValueError):
                montar_mensagens_pergunta("Olá", "conversa", prompts_dir=destino)

    def test_bloqueia_adapter_de_classificacao_em_perguntas(self):
        with self.assertRaises(ValueError):
            validar_adapter_perguntas(Path("models/adapters-evidencias-v3/best"))
        validar_adapter_perguntas(Path("models/adapters-perguntas-v2"))
        validar_adapter_perguntas(Path("models/adapters-evidencias-v3/best"), permitir_incompativel=True)

    def test_resposta_conclusiva_exige_justificativa_e_citacoes(self):
        texto = json.dumps({
            "status": "RESPONDIDA", "resposta": "Sim [1] [2]",
            "justificativa": "Duas fontes apoiam [1] [2]", "fontes_usadas": [1, 2],
        })
        self.assertEqual(validar_resposta_factual(texto, 2, 2)["status"], "RESPONDIDA")
        sem_citacao = texto.replace("Duas fontes apoiam [1] [2]", "Duas fontes apoiam")
        self.assertEqual(validar_resposta_factual(sem_citacao, 2, 2)["status"], "INCONCLUSIVA")
        inventada = texto.replace("Sim [1] [2]", "Sim [1] [2] [9]")
        self.assertEqual(validar_resposta_factual(inventada, 2, 2)["status"], "INCONCLUSIVA")

    def test_conversa_rejeita_justificativa_ausente(self):
        saida = validar_resposta_conversa('{"resposta":"Olá","fontes_usadas":[]}')
        self.assertFalse(saida["json_valido"])
        self.assertIn("não forneceu", saida["justificativa"])

    def test_exemplo_leitura_nao_transforma_rotulo_em_fato(self):
        exemplo = {"messages": [
            {"role": "user", "content": "ALEGACAO:\nUma afirmação\n\nEVIDENCIAS:\n[1] Fonte: Site\nURL: https://exemplo.org/a\nTrecho: Uma pessoa afirmou uma coisa em entrevista, mas isto não prova a alegação."},
            {"role": "assistant", "content": '{"veredito":"FALSA"}'},
        ]}
        convertido = exemplo_leitura(exemplo)
        resposta = json.loads(convertido["messages"][-1]["content"])
        self.assertNotIn("FALSA", resposta["resposta"])
        self.assertIn("não uma confirmação independente", resposta["justificativa"])

    def test_exemplo_revisado_exige_citacoes(self):
        with self.assertRaises(ValueError):
            exemplo_revisado({
                "split": "train", "modo": "evidencias", "pergunta": "P?", "resposta": "Sim",
                "justificativa": "A fonte comprova", "status": "RESPONDIDA",
                "fontes": [{"url": "https://exemplo.org", "trecho": "texto"}], "fontes_usadas": [1],
            })

    def test_preparo_gera_splits_sem_sobrescrever(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            revisados = raiz / "revisados.jsonl"
            exemplos = [
                {"split": split, "modo": "conversa", "pergunta": f"Pergunta {split}?",
                 "resposta": "Resposta", "justificativa": "Porque é assim."}
                for split in ("train", "valid", "test")
            ]
            revisados.write_text("\n".join(json.dumps(e) for e in exemplos) + "\n", encoding="utf-8")
            args = argparse.Namespace(evidence_data_dir=None, curated_jsonl=revisados,
                                      output_dir=raiz / "out", prompts_dir=DEFAULT_PROMPTS_DIR,
                                      overwrite=False)
            preparar(args)
            self.assertEqual(len((args.output_dir / "test.jsonl").read_text().splitlines()), 1)
            with self.assertRaises(FileExistsError):
                preparar(args)


class MlflowPerguntasTest(unittest.TestCase):
    def test_registra_resposta_justificativa_e_fontes(self):
        from responder_pergunta import registrar_pergunta_mlflow
        import mlflow

        with tempfile.TemporaryDirectory() as pasta:
            uri = f"sqlite:///{Path(pasta) / 'mlflow.db'}"
            args = argparse.Namespace(tracking_uri=uri, experiment_name="teste-perguntas",
                                      modo="evidencias", model="modelo-teste", adapters_dir=None,
                                      max_tokens=100, min_fontes=1, prompts_dir=DEFAULT_PROMPTS_DIR)
            resultado = {"pergunta": "P?", "status": "RESPONDIDA", "resposta": "R [1]",
                         "justificativa": "Evidência [1]", "fontes_usadas": [1],
                         "fontes_recuperadas": [{"url": "https://exemplo.org"}],
                         "resposta_bruta_modelo": "bruta", "prompt_modelo": "prompt", "json_valido": True,
                         "citacoes_validas": True}
            with patch("responder_pergunta.ROOT_DIR", Path(pasta)):
                run_id = registrar_pergunta_mlflow(resultado, args)
            client = mlflow.tracking.MlflowClient(tracking_uri=uri)
            artefato = client.download_artifacts(run_id, "pergunta/resultado.json")
            self.assertEqual(json.loads(Path(artefato).read_text())["justificativa"], "Evidência [1]")


class NovaTentativaPerguntasTest(unittest.TestCase):
    def test_corrige_citacoes_em_segunda_tentativa(self):
        from responder_pergunta import responder

        primeira = json.dumps({
            "status": "RESPONDIDA", "resposta": "Sim", "justificativa": "A fonte confirma",
            "fontes_usadas": [1],
        })
        segunda = json.dumps({
            "status": "RESPONDIDA", "resposta": "Sim [1] [2]",
            "justificativa": "As duas fontes confirmam [1] [2]", "fontes_usadas": [1, 2],
        })
        gerar = Mock(side_effect=[primeira, segunda])
        modulo_mlx = types.SimpleNamespace(generate=gerar)
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda mensagens, **_: json.dumps(mensagens)
        fontes = [
            {"titulo": "A", "url": "https://a.example/x", "trecho": "confirma"},
            {"titulo": "B", "url": "https://b.example/x", "trecho": "confirma"},
        ]
        args = argparse.Namespace(
            modo="evidencias", pergunta="É verdade?", model="modelo", adapters_dir=None,
            permitir_adapter_incompativel=False, prompts_dir=DEFAULT_PROMPTS_DIR,
            max_resultados=5, min_fontes=2, max_tokens=200, max_tentativas=2,
        )
        with patch.dict(sys.modules, {"mlx_lm": modulo_mlx}), \
             patch("responder_pergunta.buscar_evidencias", return_value=fontes), \
             patch("responder_pergunta.carregar_modelo", return_value=(object(), tokenizer)):
            resultado = responder(args)
        self.assertEqual(resultado["status"], "RESPONDIDA")
        self.assertEqual(resultado["fontes_usadas"], [1, 2])
        self.assertEqual(len(resultado["tentativas"]), 2)
        self.assertIn("citacoes_invalidas", resultado["tentativas"][0]["ajustes_aplicados"])


if __name__ == "__main__":
    unittest.main()
