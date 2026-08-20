.PHONY: all data train viz test clean help

python := python

help:
	@echo "Comandos disponíveis:"
	@echo "  make data   - Baixa/gera os dados brutos e constrói a base analítica"
	@echo "  make train  - Executa o benchmark dos 5 métodos causais e refutações"
	@echo "  make viz    - Gera todas as figuras, mapas e visualizações"
	@echo "  make test   - Executa os testes automatizados com pytest"
	@echo "  make all    - Executa todo o pipeline end-to-end"

data:
	$(python) -m src.data.feature_engineering

train:
	$(python) -m src.causal.benchmark

viz:
	$(python) -m src.viz.generate_all_plots

test:
	$(python) -m pytest tests/

all: data train viz test

clean:
	rm -rf data/processed/* results/figures/* results/tables/* .pytest_cache __pycache__
