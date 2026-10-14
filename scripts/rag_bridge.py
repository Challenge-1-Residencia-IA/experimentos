"""Ponte local WhatsApp/RAG. Execute: python -m scripts.rag_bridge."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


MAX_BODY = 32768
MAX_MESSAGE = 6000
PRIVATE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b|\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b|(?<!\w)(?:\+55\s*)?\(?\d{2}\)?[ -]?9?\d{4}[ -]?\d{4}(?!\w)")


def runtime(connection):
    """Um processo persistente; timeout descarta o processamento local pendente."""
    try:
        from dotenv import load_dotenv
        from rag.corpus import ROOT
        load_dotenv(ROOT / ".env")
        # Definir antes de importar bibliotecas que leem o modo offline no import.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import requests
        from rag.store import VectorStore
        from rag.tutor import ask_online
        from rag.web import WebResearcher

        endpoint = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama deve ser local")
        tags = requests.get(f"{parsed.scheme}://{parsed.netloc}/api/tags", timeout=5)
        tags.raise_for_status()
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        if model not in {item.get("name") for item in tags.json().get("models", [])}:
            raise ValueError("Modelo configurado não instalado")
        # Não baixa embeddings durante o startup: o cache deve estar preparado.
        store = VectorStore()
        if not all(collection.count() for collection in store.collections.values()):
            raise ValueError("Índice local incompleto")
        connection.send({"ready": True})
        while True:
            payload = connection.recv()
            try:
                # Não manter mensagens pessoais no cache entre requisições.
                researcher = WebResearcher()
                result = ask_online(payload["message"], researcher, store,
                                    history=payload["history"], track=False, model_name=model,
                                    fail_on_transport_error=True)
                connection.send({"analysis": result})
            except Exception:
                connection.send({"error": "analysis_failed"})
    except Exception:
        connection.send({"ready": False, "error": "runtime_unavailable"})
    finally:
        connection.close()


class Bridge:
    def __init__(self, timeout=180, startup_timeout=120):
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.lock = threading.Lock()
        self.ready = False
        self.process = None
        self.connection = None

    def initialize(self):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=runtime, args=(child,), daemon=True)
        self.process.start()
        child.close()
        try:
            self.ready = (self.connection.poll(self.startup_timeout)
                          and self.connection.recv().get("ready") is True)
        except (EOFError, OSError):
            self.ready = False
        if not self.ready:
            self.close()

    def close(self):
        self.ready = False
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
        if self.connection is not None:
            self.connection.close()

    def analyze(self, payload):
        if not self.lock.acquire(blocking=False):
            return 429, {"error": "busy"}
        try:
            if not self.ready or not self.process.is_alive():
                return 503, {"error": "not_ready"}
            self.connection.send(payload)
            if not self.connection.poll(self.timeout):
                self.close()
                return 504, {"error": "analysis_timeout"}
            result = self.connection.recv()
            return (502 if "error" in result else 200), result
        except (EOFError, OSError, BrokenPipeError):
            self.close()
            return 503, {"error": "runtime_unavailable"}
        finally:
            self.lock.release()


def validate(payload):
    if not isinstance(payload, dict) or set(payload) - {"request_id", "message", "history"}:
        raise ValueError("invalid_request")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", request_id):
        raise ValueError("invalid_request_id")
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE:
        raise ValueError("invalid_message")
    history = payload.get("history", [])
    if not isinstance(history, list) or len(history) > 2:
        raise ValueError("invalid_history")
    for entry in history:
        if (not isinstance(entry, dict) or set(entry) != {"pergunta"}
                or not isinstance(entry["pergunta"], str) or len(entry["pergunta"]) > 1000):
            raise ValueError("invalid_history")
    # A pesquisa web envia a alegação a terceiros. Filtro conservador, não DLP completo.
    if PRIVATE.search(message) or any(PRIVATE.search(item["pergunta"]) for item in history):
        raise ValueError("personal_data_detected")
    return {"request_id": request_id, "message": message.strip(), "history": history}


def handler_for(bridge, port):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass  # Não gravar corpo, identificadores ou URLs pessoais.

        def respond(self, status, payload):
            data = json.dumps({"schema_version": 1, **payload}, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (OSError, TimeoutError):
                pass

        def local_request(self):
            # Rejeita chamadas de páginas web e DNS rebinding; não habilita CORS.
            return (not self.headers.get("Origin")
                    and self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"})

        def do_GET(self):
            if not self.local_request():
                return self.respond(403, {"error": "forbidden"})
            if self.path != "/health":
                return self.respond(404, {"error": "not_found"})
            ready = bridge.ready and bridge.process is not None and bridge.process.is_alive()
            self.respond(200 if ready else 503, {"ready": ready, "busy": bridge.lock.locked()})

        def do_POST(self):
            request_id = uuid.uuid4().hex
            if not self.local_request():
                return self.respond(403, {"error": "forbidden"})
            if self.path != "/analyze":
                return self.respond(404, {"error": "not_found"})
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("invalid_request")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    return self.respond(413, {"error": "input_too_large", "request_id": request_id})
                if self.headers.get_content_type() != "application/json":
                    return self.respond(415, {"error": "invalid_content_type", "request_id": request_id})
                payload = validate(json.loads(self.rfile.read(length).decode("utf-8")))
                request_id = payload["request_id"]
            except (ValueError, UnicodeError, OSError) as error:
                code = str(error)
                if code not in {"invalid_request", "invalid_request_id", "invalid_message",
                                "invalid_history", "personal_data_detected"}:
                    code = "invalid_json"
                return self.respond(400, {"error": code, "request_id": request_id})
            status, result = bridge.analyze(payload)
            self.respond(status, {"request_id": request_id, **result})
    return Handler


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.timeout <= 600:
        parser.error("Porta ou timeout inválido")
    bridge = Bridge(timeout=args.timeout)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(bridge, args.port))
    initializer = threading.Thread(target=bridge.initialize, daemon=True)
    initializer.start()
    print(f"Ponte local: http://127.0.0.1:{args.port}; consulte /health antes de iniciar o bot.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        initializer.join(timeout=bridge.startup_timeout + 5)
        bridge.close()


if __name__ == "__main__":
    main()
