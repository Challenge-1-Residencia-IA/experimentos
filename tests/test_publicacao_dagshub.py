"""Verificações locais do preparo da publicação, sem acesso ao DagsHub."""

import pytest

from scripts import publicar_dataset_rag_mlflow, registrar_modelo_rag_mlflow


class PreparoConcluido(Exception):
    """Interrompe o teste antes de qualquer operação remota."""


def test_registro_nao_bloqueia_license_verificar_antes_do_ollama(tmp_path, monkeypatch):
    corpus = tmp_path / "documents.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    store = tmp_path / "chroma"
    store.mkdir()
    monkeypatch.setattr(registrar_modelo_rag_mlflow, "CORPUS", corpus)
    monkeypatch.setattr(registrar_modelo_rag_mlflow, "STORE", store)
    monkeypatch.setattr(registrar_modelo_rag_mlflow, "read_corpus", lambda: {
        "documento": {"license": "verificar"}
    })
    monkeypatch.setattr(registrar_modelo_rag_mlflow, "ollama_digest", lambda _model: (
        _ for _ in ()).throw(PreparoConcluido)
    )

    with pytest.raises(PreparoConcluido):
        registrar_modelo_rag_mlflow.main()


def test_dataset_monta_manifesto_sem_bloqueio_e_sem_envio(tmp_path, monkeypatch):
    corpus = tmp_path / "documents.jsonl"
    cases = tmp_path / "casos.jsonl"
    manifest = tmp_path / "manifest.json"
    for path in (corpus, cases, manifest):
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(publicar_dataset_rag_mlflow, "CORPUS", corpus)
    monkeypatch.setattr(publicar_dataset_rag_mlflow, "COLLEAGUE_CASES", cases)
    monkeypatch.setattr(publicar_dataset_rag_mlflow, "COLLEAGUE_MANIFEST", manifest)
    monkeypatch.setattr(publicar_dataset_rag_mlflow, "BENCHMARK", tmp_path / "benchmark")
    monkeypatch.setattr(publicar_dataset_rag_mlflow, "read_corpus", lambda _path: {
        "documento": {"kind": "evidencia", "license": "verificar"}
    })
    monkeypatch.setattr(publicar_dataset_rag_mlflow, "configure_tracking", lambda **_kwargs: (
        _ for _ in ()).throw(PreparoConcluido)
    )

    with pytest.raises(PreparoConcluido):
        publicar_dataset_rag_mlflow.main()
