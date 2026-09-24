import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from auditoria_evidencias import calcular_metricas_auditoria, montar_rastro_decisao
from evidencias_utils import extrair_json, normalizar_veredito, validar_resposta
from preparar_dataset_evidencias import balancear_treino, grupo_url, reduzir_vazamento_rotulo, separar
from verificar_link import extrair_texto, filtrar_fontes, filtrar_fontes_com_auditoria, montar_auditoria


class EvidenciasUtilsTest(unittest.TestCase):
    def test_extrai_json_cercado_por_texto(self):
        resposta = '```json\n{"veredito":"VERDADEIRA","fontes_usadas":[1]}\n```'
        self.assertEqual(extrair_json(resposta)["veredito"], "VERDADEIRA")

    def test_veredito_conclusivo_sem_citacao_vira_inconclusivo(self):
        resposta = json.dumps({"veredito": "FALSA", "justificativa": "teste", "fontes_usadas": [9]})
        validada = validar_resposta(resposta, total_fontes=2)
        self.assertEqual(validada["veredito"], "INCONCLUSIVA")
        self.assertEqual(validada["fontes_usadas"], [])

    def test_veredito_exige_minimo_configurado_de_fontes(self):
        resposta = json.dumps({
            "veredito": "VERDADEIRA",
            "justificativa": "A fonte confirma.",
            "fontes_usadas": [1],
        })
        validada = validar_resposta(resposta, total_fontes=5, minimo_fontes_conclusivas=2)
        self.assertEqual(validada["veredito"], "INCONCLUSIVA")
        self.assertTrue(validada["fontes_insuficientes"])
        self.assertEqual(validada["fontes_usadas"], [1])

    def test_veredito_aceita_duas_fontes_quando_configurado(self):
        resposta = json.dumps({
            "veredito": "VERDADEIRA",
            "justificativa": "As fontes confirmam.",
            "fontes_usadas": [1, 2],
        })
        validada = validar_resposta(resposta, total_fontes=5, minimo_fontes_conclusivas=2)
        self.assertEqual(validada["veredito"], "VERDADEIRA")
        self.assertFalse(validada["fontes_insuficientes"])

    def test_metricas_e_rastro_da_justificativa(self):
        fontes = [
            {"titulo": "Órgão confirma dado", "url": "https://gov.br/dado", "trecho": "O órgão confirmou o dado oficial."},
            {"titulo": "Agência confirma dado", "url": "https://reuters.com/dado", "trecho": "A agência também confirmou o dado."},
        ]
        verificacao = validar_resposta(json.dumps({
            "veredito": "VERDADEIRA",
            "justificativa": "As fontes [1] e [2] confirmam o dado oficial.",
            "fontes_usadas": [1, 2],
        }), total_fontes=2, minimo_fontes_conclusivas=2)
        metricas = calcular_metricas_auditoria("O dado foi confirmado", verificacao, fontes, minimo_fontes=2)
        rastro = montar_rastro_decisao("O dado foi confirmado", verificacao, fontes, minimo_fontes=2)

        self.assertEqual(metricas["fontes_citadas"], 2)
        self.assertEqual(metricas["requisito_minimo_fontes_atendido"], 1)
        self.assertEqual(metricas["taxa_fontes_citadas_mencionadas_na_justificativa"], 1.0)
        self.assertGreater(metricas["sobreposicao_justificativa_evidencias"], 0)
        self.assertEqual(len(rastro["fontes_citadas_com_trechos"]), 2)

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
        _, auditoria = filtrar_fontes_com_auditoria(fontes, "https://g1.globo.com/a")
        self.assertEqual(auditoria["descartes_por_motivo"]["mesmo_site_da_noticia"], 1)
        self.assertEqual(auditoria["descartes_por_motivo"]["rede_social"], 1)
        self.assertEqual(auditoria["descartes_por_motivo"]["dominio_repetido"], 1)

    def test_oversample_preserva_todos_os_ids(self):
        registros = [
            {"id": "f1", "label": "FALSA"},
            {"id": "f2", "label": "FALSA"},
            {"id": "v1", "label": "VERDADEIRA"},
        ]
        balanceados = balancear_treino(registros, seed=42, estrategia="oversample", tamanho_alvo=2)
        self.assertTrue({"f1", "f2", "v1"}.issubset({r["id"] for r in balanceados}))
        self.assertEqual(len(balanceados), 4)

    def test_oversample_nao_descarta_maioria_acima_do_alvo(self):
        registros = [{"id": f"f{i}", "label": "FALSA"} for i in range(3)]
        registros.append({"id": "v1", "label": "VERDADEIRA"})
        balanceados = balancear_treino(registros, seed=42, estrategia="oversample", tamanho_alvo=2)
        self.assertTrue({r["id"] for r in registros}.issubset({r["id"] for r in balanceados}))
        self.assertEqual(len(balanceados), 6)

    def test_remove_vazamento_explicito(self):
        self.assertNotIn("é falsa", reduzir_vazamento_rotulo("A informação é falsa porque os dados divergem."))

    def test_split_mantem_checagem_no_mesmo_conjunto(self):
        registros = [
            {"id": f"{i}-{j}", "url": f"https://www.exemplo.com/checagem/{i}/" + ("?origem=busca" if j else ""),
             "label": "FALSA" if j == 0 else "VERDADEIRA"}
            for i in range(14) for j in range(2)
        ]
        splits = separar(registros, seed=42)
        grupos = [{grupo_url(r["url"]) for r in splits[nome]} for nome in ("train", "valid", "test")]
        self.assertTrue(all(grupos))
        self.assertTrue(grupos[0].isdisjoint(grupos[1]))
        self.assertTrue(grupos[0].isdisjoint(grupos[2]))
        self.assertTrue(grupos[1].isdisjoint(grupos[2]))
        self.assertEqual(sum(map(len, splits.values())), len(registros))
        self.assertEqual(splits, separar(registros, seed=42))

    def test_auditoria_exibe_regra_e_fonte_citada(self):
        auditoria = montar_auditoria(
            {"veredito": "FALSA", "justificativa": "A fonte contradiz.", "fontes_usadas": [1]},
            [{"titulo": "Fonte", "url": "https://gov.br/dado", "trecho": "Dado oficial"}],
            minimo_fontes=1,
        )
        self.assertIn("contradizer", auditoria["requisito_para_o_veredito_emitido"])
        self.assertEqual(auditoria["fontes_que_o_modelo_disse_ter_usado"][0]["titulo"], "Fonte")

    @patch("verificar_link.buscar_evidencias")
    @patch("verificar_link.requests.get")
    def test_extracao_usa_busca_quando_site_bloqueia(self, requisicao, busca):
        resposta = Mock()
        resposta.raise_for_status.side_effect = requests.HTTPError("403", response=Mock(status_code=403))
        requisicao.return_value = resposta
        busca.return_value = [{
            "titulo": "Título recuperado pela busca",
            "url": "https://noticias.exemplo.com/noticia/teste.ghtml",
            "trecho": "Este trecho indexado possui conteúdo suficiente para representar a notícia bloqueada e extrair sua alegação principal.",
        }]

        titulo, texto, metodo = extrair_texto(
            "https://noticias.exemplo.com/noticia/teste.ghtml", timeout=5, max_chars=500
        )

        self.assertEqual(titulo, "Título recuperado pela busca")
        self.assertIn("conteúdo suficiente", texto)
        self.assertEqual(metodo, "indice_de_busca")

    @patch("verificar_link.buscar_evidencias", return_value=[])
    @patch("verificar_link.requests.get")
    def test_extracao_usa_titulo_da_url_quando_site_e_busca_bloqueiam(self, requisicao, _busca):
        resposta = Mock()
        resposta.raise_for_status.side_effect = requests.HTTPError("403", response=Mock(status_code=403))
        requisicao.return_value = resposta

        titulo, texto, metodo = extrair_texto(
            "https://noticias.exemplo.com/quem-e-empresario-ligado-a-banco-e-alvo-do-ministerio-publico.ghtml",
            timeout=5,
            max_chars=500,
        )

        self.assertEqual(titulo, texto)
        self.assertIn("empresario ligado", titulo.lower())
        self.assertEqual(metodo, "titulo_da_url")


if __name__ == "__main__":
    unittest.main()
