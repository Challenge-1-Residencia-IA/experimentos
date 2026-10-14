"""Contrato da ponte sem carregar modelo/índice. Criados, mas não executados na reconstrução."""
import unittest
from unittest.mock import Mock

from scripts.rag_bridge import Bridge, validate


class ContractTests(unittest.TestCase):
    def test_minimal_and_history(self):
        payload = validate({"request_id": "abc_123", "message": "Notícia pública", "history": [{"pergunta": "Contexto público"}]})
        self.assertEqual(payload["message"], "Notícia pública")

    def test_invalid_inputs(self):
        for payload in (None, [], {}, {"request_id": "a", "message": ""},
                        {"request_id": "a", "message": "x" * 6001},
                        {"request_id": "a", "message": "notícia", "chat": "contato"},
                        {"request_id": "a", "message": "notícia", "history": [{}]},
                        {"request_id": "../a", "message": "notícia"}):
            with self.subTest(payload_type=type(payload).__name__), self.assertRaises(ValueError):
                validate(payload)

    def test_personal_data_before_search(self):
        for text in ("email pessoa@example.org", "CPF 123.456.789-00", "telefone (11) 99999-1234"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "personal_data_detected"):
                validate({"request_id": "a", "message": text})

    def test_busy_and_not_ready(self):
        bridge = Bridge()
        self.assertEqual(bridge.analyze({})[0], 503)
        bridge.lock.acquire()
        try:
            self.assertEqual(bridge.analyze({})[0], 429)
        finally:
            bridge.lock.release()

    def test_timeout_terminates_worker_and_releases_lock(self):
        bridge = Bridge(timeout=1)
        bridge.ready = True
        bridge.process = Mock()
        bridge.process.is_alive.return_value = True
        bridge.connection = Mock()
        bridge.connection.poll.return_value = False
        status, result = bridge.analyze({"message": "Notícia"})
        self.assertEqual((status, result["error"]), (504, "analysis_timeout"))
        self.assertFalse(bridge.ready)
        self.assertFalse(bridge.lock.locked())
        bridge.process.terminate.assert_called_once()

    def test_model_failure_is_not_success(self):
        bridge = Bridge()
        bridge.ready = True
        bridge.process = Mock()
        bridge.connection = Mock()
        bridge.connection.recv.return_value = {"error": "analysis_failed"}
        self.assertEqual(bridge.analyze({})[0], 502)
        self.assertFalse(bridge.lock.locked())


if __name__ == "__main__":
    unittest.main()
