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
