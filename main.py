"""
Ponto de Entrada Principal (CLI) para `industrial-causal-policy`.
-----------------------------------------------------------------
Uso:
    python main.py --step data
    python main.py --step train
    python main.py --step viz
    python main.py --step test
    python main.py --step all
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_data_step():
    logger.info("=== ETAPA 1/4: PROCESSAMENTO DE DADOS E ENGENHARIA DE ATRIBUTOS ===")
    from src.data.feature_engineering import build_analytical_dataset, build_synthetic_control_panel
    build_analytical_dataset()
    build_synthetic_control_panel()


def run_train_step():
    logger.info("=== ETAPA 2/4: ESTIMAÇÃO DO BENCHMARK DE CAUSAL ML E REFUTAÇÃO ===")
    from src.causal.benchmark import run_benchmark
    run_benchmark()


def run_viz_step():
    logger.info("=== ETAPA 3/4: GERAÇÃO DE VISUALIZAÇÕES E MAPAS COROPLÉTICOS ===")
    from src.viz.generate_all_plots import generate_all_figures
    generate_all_figures()


def run_test_step():
    logger.info("=== ETAPA 4/4: SUÍTE DE TESTES AUTOMATIZADOS ===")
    import pytest
    ret_code = pytest.main(["-v", "tests/"])
    if ret_code != 0:
        logger.error("Falha na execução dos testes automatizados!")
        sys.exit(ret_code)


def main():
    parser = argparse.ArgumentParser(description="CLI do Benchmark Industrial Causal Policy")
    parser.add_argument(
        "--step",
        choices=["data", "train", "viz", "test", "all"],
        default="all",
        help="Etapa do pipeline a executar (padrão: all)"
    )
    args = parser.parse_args()

    if args.step == "data":
        run_data_step()
    elif args.step == "train":
        run_train_step()
    elif args.step == "viz":
        run_viz_step()
    elif args.step == "test":
        run_test_step()
    elif args.step == "all":
        run_data_step()
        run_train_step()
        run_viz_step()
        run_test_step()
        logger.info("Pipeline completo executado com sucesso.")


if __name__ == "__main__":
    main()
