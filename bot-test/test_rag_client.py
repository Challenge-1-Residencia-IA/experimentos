import unittest
from unittest.mock import patch

import rag_client


class RagClientTests(unittest.IsolatedAsyncioTestCase):
    def test_format_separates_cited_and_consulted_sources(self):
        answer = {
            "resposta_usuario": "A nota explica a condição [1].",
            "fontes": [
                {"numero": 1, "titulo": "Nota", "origem": "Agência", "data": "2026-10-01",
                 "url": "https://example.org/nota"},
                {"numero": 2, "url": "https://example.org/nao-citada"},
            ],
            "fontes_encontradas": [{"url": "https://example.org/consultada"}],
        }
        output = rag_client.format_answer(answer)
        self.assertIn("https://example.org/nota", output)
        self.assertNotIn("nao-citada", output)
        self.assertIn("https://example.org/consultada", output)
        self.assertIn("Páginas consultadas (não comprovam", output)

    def test_missing_citation_is_rejected(self):
        with self.assertRaises(rag_client.BridgeError):
            rag_client.format_answer({"sintese": "Veja [2].", "fontes": []})

    def test_inconclusive_answer_shows_search_scope_without_claiming_proof(self):
        output = rag_client.format_answer({
            "resposta_usuario": "Esta busca não permitiu concluir.",
            "avaliacao_evidencias": "EVIDENCIAS_INSUFICIENTES",
            "fontes": [],
            "pesquisa_web": {"resultados": 6, "paginas_lidas": 2, "erro_busca": ""},
            "fontes_encontradas": [{"titulo": "Página consultada", "url": "https://example.org/pagina"}],
        })
        self.assertIn("6 resultados, 2 páginas lidas", output)
        self.assertIn("não comprovam a alegação", output)
        self.assertIn("https://example.org/pagina", output)

    def test_all_links_are_plain_urls_once_and_labeled(self):
        output = rag_client.format_answer({
            "resposta_usuario": "Não há trechos suficientes [1].",
            "fontes": [{"numero": 1, "url": "https://example.org/citada"}],
            "fontes_encontradas": [{"url": "https://example.org/citada"},
                                   {"url": "https://example.org/consultada"}],
            "fontes_descartadas": [{"url": "https://example.org/outra"}],
            "checagens_encontradas": [{"url": "https://example.org/checagem"}],
            "pesquisa_web": {"links_resultados": ["https://example.org/citada",
                                                "https://example.org/consultada",
                                                "https://example.org/nao-lida"]},
        })
        for suffix in ("citada", "consultada", "outra", "checagem", "nao-lida"):
            self.assertEqual(output.count(f"https://example.org/{suffix}"), 1)
        self.assertIn("Resultados da busca não verificados", output)
        self.assertNotIn("[https://", output)

    def test_search_results_are_not_truncated_to_two_links(self):
        urls = [f"https://example.org/resultado-{index}" for index in range(17)]
        output = rag_client.format_answer({
            "resposta_usuario": "A busca não trouxe trechos suficientes.",
            "fontes": [], "pesquisa_web": {"links_resultados": urls},
        })
        assert all(url in output for url in urls)

    def test_multiple_claims_keep_each_citation_local(self):
        answer = {"analises": [
            {"afirmacao": "Projeto aprovado", "sintese": "Há registro [1].",
             "fontes": [{"numero": 1, "url": "https://example.org/primeiro"}]},
            {"afirmacao": "Projeto rejeitado", "sintese": "Outra etapa [1].",
             "fontes": [{"numero": 1, "url": "https://example.org/segundo"}]},
        ]}
        output = rag_client.format_answer(answer)
        self.assertIn("Afirmação 1: Projeto aprovado", output)
        self.assertIn("Afirmação 2: Projeto rejeitado", output)
        self.assertIn("https://example.org/primeiro", output)
        self.assertIn("https://example.org/segundo", output)

    async def test_request_id_must_match(self):
        with patch.object(rag_client, "_request", return_value={"request_id": "outro", "answer": {}}):
            with self.assertRaises(rag_client.BridgeError):
                await rag_client.analyze("esperado", "Pergunta")


if __name__ == "__main__":
    unittest.main()
