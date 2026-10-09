"""Atualização periódica do corpus e do índice ao abrir a conversa."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rag.corpus import CORPUS, ROOT, ingest
from rag.splitter import CHUNKING_VERSION

STATE = ROOT / "data" / "rag_auto_refresh.json"
INTERVAL = timedelta(hours=24)


def config_signature(config_path: Path) -> str:
    """Inclui no hash a configuração e o conteúdo das fontes locais."""
    digest = hashlib.sha256(config_path.read_bytes())
    digest.update(CHUNKING_VERSION.encode("utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for entry in config.get("documents", []):
        location = entry.get("path")
        if not location:
            continue
        path = (config_path.parent / location).resolve()
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"<arquivo-ausente>")
    return digest.hexdigest()


def refresh_if_due(store, config_path: Path, state_path: Path = STATE) -> dict | None:
    """Coleta e indexa se passou um dia ou se a configuração mudou."""
    signature = config_signature(config_path)
    now = datetime.now(timezone.utc)
    try:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        last = datetime.fromisoformat(previous["updated_at"])
        if (
            previous.get("config_hash") == signature
            and last.tzinfo is not None
            and now - last < INTERVAL
            and CORPUS.exists()
            and any(collection.count() for collection in store.collections.values())
        ):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        pass

    collected = ingest(config_path)
    indexed = store.index()
    result = {"collected": collected, "indexed": indexed}
    if collected["failed"] == 0:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": signature,
        }), encoding="utf-8")
        temporary.replace(state_path)
    return result
