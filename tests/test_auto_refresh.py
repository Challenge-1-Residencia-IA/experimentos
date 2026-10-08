import json

from rag.auto_refresh import config_signature


def test_assinatura_muda_quando_fonte_local_muda(tmp_path):
    source = tmp_path / "guia.md"
    source.write_text("primeira versão", encoding="utf-8")
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({
        "documents": [{"path": "guia.md"}],
        "feeds": [],
    }), encoding="utf-8")

    before = config_signature(config)
    source.write_text("segunda versão", encoding="utf-8")

    assert config_signature(config) != before
