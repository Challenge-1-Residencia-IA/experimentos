import json
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from evidencias_utils import extrair_json, normalizar_veredito, validar_resposta
from preparar_dataset_evidencias import balancear_treino, reduzir_vazamento_rotulo
from verificar_link import filtrar_fontes, montar_auditoria


class EvidenciasUtilsTest(unittest.TestCase):
    def test_extrai_json_cercado_por_texto(self):
        resposta = '```json\n{"veredito":"VERDADEIRA","fontes_usadas":[1]}\n```'
        self.assertEqual(extrair_json(resposta)["veredito"], "VERDADEIRA")

    def test_veredito_conclusivo_sem_citacao_vira_inconclusivo(self):
        resposta = json.dumps({"veredito": "FALSA", "justificativa": "teste", "fontes_usadas": [9]})
        validada = validar_resposta(resposta, total_fontes=2)
        self.assertEqual(validada["veredito"], "INCONCLUSIVA")
        self.assertEqual(validada["fontes_usadas"], [])

    def test_normaliza_fora_de_contexto(self):
        self.assertEqual(normalizar_veredito("fora de contexto"), "FORA_DE_CONTEXTO")

    def test_filtra_autocitacao_social_e_dominio_repetido(self):
        fontes = [
            {"url": "https://g1.globo.com/a", "titulo": "original", "trecho": ""},
            {"url": "https://instagram.com/p/1", "titulo": "social", "trecho": ""},
            {"url": "https://gov.br/fonte-a", "titulo": "oficial", "trecho": ""},
            {"url": "https://gov.br/fonte-b", "titulo": "repetida", "trecho": ""},
            {"url": "https://reuters.com/mundo/x", "titulo": "independente", "trecho": ""},
        ]
        filtradas = filtrar_fontes(fontes, "https://g1.globo.com/a")
        self.assertEqual([f["titulo"] for f in filtradas], ["oficial", "independente"])

    def test_oversample_preserva_todos_os_ids(self):
        registros = [
            {"id": "f1", "label": "FALSA"},
            {"id": "f2", "label": "FALSA"},
            {"id": "v1", "label": "VERDADEIRA"},
        ]
        balanceados = balancear_treino(registros, seed=42, estrategia="oversample", tamanho_alvo=2)
        self.assertTrue({"f1", "f2", "v1"}.issubset({r["id"] for r in balanceados}))
        self.assertEqual(len(balanceados), 4)

    def test_remove_vazamento_explicito(self):
        self.assertNotIn("é falsa", reduzir_vazamento_rotulo("A informação é falsa porque os dados divergem."))

    def test_auditoria_exibe_regra_e_fonte_citada(self):
        auditoria = montar_auditoria(
            {"veredito": "FALSA", "justificativa": "A fonte contradiz.", "fontes_usadas": [1]},
            [{"titulo": "Fonte", "url": "https://gov.br/dado", "trecho": "Dado oficial"}],
            minimo_fontes=1,
        )
        self.assertIn("contradizer", auditoria["requisito_para_o_veredito_emitido"])
        self.assertEqual(auditoria["fontes_que_o_modelo_disse_ter_usado"][0]["titulo"], "Fonte")


if __name__ == "__main__":
    unittest.main()
