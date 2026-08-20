"""
Orquestrador do Benchmark de Inferência Causal.
------------------------------------------------
Executa todos os 5 métodos causais (IPW, Linear DML, Causal Forest DML, DR Learner, X-Learner)
além do Controle Sintético e testes de refutação.
Consolida ATEs, ICs, p-valores e métricas de RMSE em uma tabela de resumo.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from src.causal.causal_forest import estimate_causal_forest
from src.causal.dr_learner import estimate_dr_learner
from src.causal.ipw import estimate_ipw
from src.causal.linear_dml import estimate_linear_dml
from src.causal.refutation import run_refutation_tests
from src.causal.synthetic_control import estimate_synthetic_control
from src.causal.x_learner import estimate_x_learner

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/params.yaml") -> dict:
    """Carrega as configurações centralizadas do arquivo YAML."""
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

COVARIATES = [
    "pib_per_capita", "idhm", "population", "dist_capital_km",
    "senai_presence", "ind_emp_share", "n_establishments",
    "avg_industrial_wage", "urbanization_rate", "tax_revenue_per_capita"
]


def run_benchmark(
    dataset_path: str = "data/processed/analytical_dataset.csv",
    output_table_path: str = "results/tables/benchmark_summary.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executa o benchmark causal completo em 5 métodos + Controle Sintético + Refutações.
    
    Retorna:
        Tupla com (df de resumo do benchmark, df analítico atualizado com CATEs, df de refutação).
    """
    df = pd.read_csv(dataset_path)
    logger.info(f"Iniciando Benchmark de Causal ML em {len(df)} observações...")
    
    cfg = load_config()
    cf_estimators = cfg.get("causal_models", {}).get("causal_forest", {}).get("n_estimators", 1000)

    # 1. IPW
    res_ipw = estimate_ipw(df, COVARIATES, n_bootstrap=100)
    
    # 2. Linear DML
    res_ldml, cates_ldml = estimate_linear_dml(df, COVARIATES)
    
    # 3. Causal Forest DML (estimador principal, n_estimators via params.yaml)
    res_cf, cates_cf, lower_cf, upper_cf, fitted_cf = estimate_causal_forest(
        df, COVARIATES, n_estimators=cf_estimators
    )
    
    # Anexar CATEs ao dataset analítico para análise de heterogeneidade e mapas
    df["cate_cf"] = cates_cf
    df["cate_ci_lower"] = lower_cf
    df["cate_ci_upper"] = upper_cf
    
    # 4. DR Learner
    res_dr, cates_dr = estimate_dr_learner(df, COVARIATES)
    
    # 5. X-Learner
    res_xl, cates_xl = estimate_x_learner(df, COVARIATES)
    
    # Salvar dataframe atualizado com CATEs estimados
    df.to_csv(dataset_path, index=False)
    
    # Consolidar resultados do benchmark
    benchmark_list = [res_ipw, res_ldml, res_cf, res_dr, res_xl]
    df_benchmark = pd.DataFrame(benchmark_list)
    
    # Salvar tabela de resumo do benchmark
    out_file = Path(output_table_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_benchmark.to_csv(out_file, index=False, encoding="utf-8")
    
    # 6. Controle Sintético (Nível macro)
    df_sc_res, sc_weights, sc_summary = estimate_synthetic_control()
    
    # 7. Testes de refutação na estimativa principal da Causal Forest
    cf_ate = res_cf["ate"]
    df_refutation = run_refutation_tests(df, estimated_ate=cf_ate)
    ref_file = out_file.parent / "refutation_summary.csv"
    df_refutation.to_csv(ref_file, index=False, encoding="utf-8")
    
    logger.info("================ RESULTADOS CONSOLIDADOS DO BENCHMARK ================")
    logger.info("\n" + df_benchmark.to_string(index=False))
    logger.info(f"Tabela de resumo salva em {out_file}")
    
    return df_benchmark, df, df_refutation


if __name__ == "__main__":
    run_benchmark()
