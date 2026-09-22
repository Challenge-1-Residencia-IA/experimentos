PROJECT_NAME = experimentos
PYTHON_INTERPRETER = python3

.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> Ambiente criado. Ative com: source .venv/bin/activate"

.PHONY: requirements
requirements:
	pip install -r requirements.txt

.PHONY: test
test:
	pytest tests

.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.DEFAULT_GOAL := help
help:
	@echo "Alvos disponíveis: create_environment, requirements, test, clean"
