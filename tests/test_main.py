import os

import main


def test_talk_usa_experimento_escolhido_e_restaura_configuracao(monkeypatch):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "Experimento-Padrao")
    observed = {}

    monkeypatch.setattr(main.Prompt, "ask", lambda *_args, **_kwargs: "  Teste-RAG-Equipe  ")
    monkeypatch.setattr(
        main,
        "_talk",
        lambda: observed.update(experiment=os.environ["MLFLOW_EXPERIMENT_NAME"]),
    )

    main.talk()

    assert observed["experiment"] == "Teste-RAG-Equipe"
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "Experimento-Padrao"


def test_menu_sincroniza_todos_os_experimentos_locais_confirmados(monkeypatch):
    from scripts import sincronizar_runs_mlflow_dagshub as sync_module

    available = [
        {"name": "NexoVerdict-RAG", "runs": 14, "runs_with_metrics": 14},
        {"name": "teste-novos-prompts", "runs": 2, "runs_with_metrics": 2},
    ]
    sent = []
    monkeypatch.setattr(main.Prompt, "ask", lambda *_args, **_kwargs: "experimentos")
    monkeypatch.setattr(main.Confirm, "ask", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sync_module, "local_experiments", lambda _database: available)
    monkeypatch.setattr(sync_module, "sync", lambda database, name, run_ids: (
        sent.append((database, name, run_ids)) or [{"status": "enviado"}]
    ))

    main.publish_dagshub()

    assert [name for _database, name, _run_ids in sent] == [item["name"] for item in available]
    assert all(database == main.ROOT / "mlflow.db" and run_ids == []
               for database, _name, run_ids in sent)


def test_menu_nao_envia_experimentos_sem_confirmacao(monkeypatch):
    from scripts import sincronizar_runs_mlflow_dagshub as sync_module

    monkeypatch.setattr(main.Prompt, "ask", lambda *_args, **_kwargs: "experimentos")
    monkeypatch.setattr(main.Confirm, "ask", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(sync_module, "local_experiments", lambda _database: [
        {"name": "teste", "runs": 1, "runs_with_metrics": 1}
    ])
    monkeypatch.setattr(sync_module, "sync", lambda *_args: (_ for _ in ()).throw(
        AssertionError("Não deveria enviar sem confirmação")))

    main.publish_dagshub()


def test_opcao_indexar_prepara_neon_antes_da_carga(tmp_path, monkeypatch):
    from rag import store

    corpus = tmp_path / "documents.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(main, "CORPUS", corpus)
    calls = []

    class FakeNeonStore:
        def initialize(self):
            calls.append("initialize")

        def index(self):
            calls.append("index")
            return {"indexed": 1, "unchanged": 0, "removed_chunks": 0}

    monkeypatch.setattr(store, "create_vector_store", FakeNeonStore)

    main.index()

    assert calls == ["initialize", "index"]
