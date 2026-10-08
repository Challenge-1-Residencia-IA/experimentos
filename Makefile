PROJECT_NAME = experimentos
PYTHON_INTERPRETER = python3

.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> Ambiente criado. Ative com: source .venv/bin/activate"

.PHONY: requirements
requirements:
	pip install -r requirements.txt
	python -m nltk.downloader -d .venv/nltk_data stopwords

.PHONY: data
data:
	python -m dados.preparar

.PHONY: golden-set
golden-set:
	python -m dados.golden_set

.PHONY: pipeline
pipeline:
	dvc repro

.PHONY: test
test:
	python -m pytest tests

.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.DEFAULT_GOAL := help
help:
	@echo "Alvos disponíveis: create_environment, requirements, data, golden-set, pipeline, test, clean"
