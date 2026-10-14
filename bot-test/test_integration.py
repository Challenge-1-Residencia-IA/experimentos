"""Testes simulados, sem WhatsApp, Ollama ou pesquisa externa. Não executados na reconstrução."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from delivery_store import DeliveryStore, message_key
from rag_client import BridgeError, RagClient, format_analysis


class FormatTests(unittest.TestCase):
    def test_cited_and_consulted_are_distinct(self):
        text = format_analysis({"sintese": "Trecho [2].", "fontes": [
            {"numero": 2, "titulo": "Fonte", "url": "https://example.org/prova"}],
            "fontes_encontradas": [{"numero": 1, "url": "https://example.org/consulta"}]})
        self.assertIn("[2] Fonte\nhttps://example.org/prova", text)
        self.assertIn("Consultadas (não são prova):\nhttps://example.org/consulta", text)

    def test_consulted_cannot_resolve_citation(self):
        with self.assertRaises(BridgeError):
            format_analysis({"sintese": "Texto [1]", "fontes_encontradas": [{"numero": 1, "url": "https://example.org"}]})

    def test_invented_url_rejected(self):
        with self.assertRaises(BridgeError):
            format_analysis({"sintese": "Veja https://example.org/inventado"})

    def test_simulation_not_analysis(self):
        with self.assertRaises(BridgeError):
            format_analysis({"sintese": "[Simulação] Sem chave"})

    def test_non_factual_and_inconclusive(self):
        for text in ("Não há evidência suficiente.", "Trata-se de opinião.", "O material é uma sátira."):
            with self.subTest(text=text):
                self.assertIn(text, format_analysis({"veredito": text, "fontes": []}))

    def test_multiple_claims(self):
        text = format_analysis({"analises": [{"sintese": "Primeira inconclusiva."}, {"sintese": "Segunda inconclusiva."}]})
        self.assertIn("Alegação 1", text)
        self.assertIn("Alegação 2", text)

    def test_fallback_findings_not_promoted_to_evidence(self):
        text = format_analysis({"veredito": "Inconclusivo.", "achados": ["Fonte [1] — trecho"],
                                "fontes_encontradas": [{"numero": 1, "url": "https://example.org/consulta"}]})
        self.assertNotIn("Fonte [1]", text)
        self.assertIn("Consultadas (não são prova)", text)

    def test_all_consulted_links_retained(self):
        urls = [f"https://example.org/{i}" for i in range(17)]
        text = format_analysis({"sintese": "Inconclusivo.", "fontes_encontradas": [{"url": url} for url in urls]})
        for url in urls:
            self.assertIn(url + "\n", text + "\n")


class DeliveryTests(unittest.TestCase):
    def test_identity_reservation_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deliveries.db"
            store = DeliveryStore(path)
            a, b = message_key("chat A", "same-id"), message_key("chat B", "same-id")
            try:
                self.assertNotEqual(a, b)
                self.assertTrue(store.claim(a))
                self.assertFalse(store.claim(a))
                self.assertTrue(store.claim(b))
                store.state(a, "ready")
                self.assertTrue(store.reserve_send(a))
                self.assertFalse(store.reserve_send(a))
            finally:
                store.close()
            reopened = DeliveryStore(path)
            try:
                self.assertTrue(reopened.known(a))
                self.assertFalse(reopened.reserve_send(a))
                self.assertEqual(reopened.db.execute("SELECT state FROM deliveries WHERE key=?", (a,)).fetchone()[0], "interrupted")
            finally:
                reopened.close()


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def client_with(self, handler):
        client = RagClient()
        await client.http.aclose()
        client.http = httpx.AsyncClient(base_url="http://127.0.0.1:8765", transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.close)
        return client

    async def test_contract_success(self):
        client = await self.client_with(lambda request: httpx.Response(200, json={
            "schema_version": 1, "request_id": "abc", "analysis": {"sintese": "Inconclusivo."}}))
        self.assertIn("Inconclusivo", await client.analyze("Notícia pública", "abc"))

    async def test_wrong_request_identity(self):
        client = await self.client_with(lambda request: httpx.Response(200, json={
            "schema_version": 1, "request_id": "outro", "analysis": {"sintese": "Texto"}}))
        with self.assertRaises(BridgeError):
            await client.analyze("Notícia", "abc")

    async def test_http_failures_are_not_answers(self):
        for status in (400, 429, 502, 503, 504):
            client = await self.client_with(lambda request, status=status: httpx.Response(status, json={"error": "failed"}))
            with self.subTest(status=status), self.assertRaises(BridgeError):
                await client.analyze("Notícia", "abc")

    async def test_transport_timeout(self):
        def timeout(request):
            raise httpx.ReadTimeout("simulado")
        client = await self.client_with(timeout)
        with self.assertRaises(BridgeError):
            await client.analyze("Notícia", "abc")

    async def test_health_not_ready(self):
        client = await self.client_with(lambda request: httpx.Response(503, json={"ready": False}))
        with self.assertRaises(BridgeError):
            await client.health()


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_never_opens_or_sends(self):
        import botfinal
        outgoing = asyncio.Queue()
        message = botfinal.Message("Contato", "abc", "Texto")
        outgoing.put_nowait((message, "Resposta"))
        store = unittest.mock.Mock()
        with patch.object(botfinal, "send_confirmed", new_callable=AsyncMock) as send:
            task = asyncio.create_task(botfinal.dispatcher(None, outgoing, asyncio.Lock(), store, False, set()))
            try:
                await asyncio.wait_for(outgoing.join(), 1)
                send.assert_not_awaited()
                store.state.assert_called_with("abc", "dry_run")
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_full_queue_does_not_claim_more_messages(self):
        import botfinal
        incoming = asyncio.Queue(1)
        incoming.put_nowait(botfinal.Message("Contato", "old", "Texto"))
        store = unittest.mock.Mock()
        with patch.object(botfinal, "open_chat", new_callable=AsyncMock) as open_chat:
            task = asyncio.create_task(botfinal.scanner(None, ["Contato"], incoming, asyncio.Lock(), store))
            try:
                await asyncio.sleep(0)
                open_chat.assert_not_awaited()
                store.claim.assert_not_called()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_media_skips_model_and_failure_is_safe_notice(self):
        import botfinal
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        incoming.put_nowait(botfinal.Message("A", "a", "", True))
        incoming.put_nowait(botfinal.Message("B", "b", "Pergunta"))
        client = unittest.mock.Mock(analyze=AsyncMock(side_effect=BridgeError("failed")))
        store = unittest.mock.Mock()
        task = asyncio.create_task(botfinal.worker(incoming, outgoing, client, store))
        try:
            await asyncio.wait_for(incoming.join(), 1)
            first, second = outgoing.get_nowait(), outgoing.get_nowait()
            self.assertEqual(first[1], botfinal.UNSUPPORTED)
            self.assertEqual(second[1], botfinal.UNAVAILABLE)
            self.assertEqual(client.analyze.await_count, 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
