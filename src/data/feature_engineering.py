"""
Engenharia de Atributos & Construtor da Base Analítica.
------------------------------------------------------
Consolida os módulos DGP (municípios, emprego, mercado de trabalho),
a atribuição de tratamento e injeta a função de efeito causal heterogêneo
real (Ground Truth CATE) para validação do benchmark.
Também gera o painel estadual para o Controle Sintético.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.dgp_employment import load_caged_data
from src.data.dgp_municipalities import generate_ceara_geojson, load_ibge_covariates
from src.data.dgp_labor_market import load_rais_data
from src.data.treatment_builder import build_treatment_assignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_true_cate(df: pd.DataFrame) -> np.ndarray:
    """
    Função de efeito causal heterogêneo real (Ground Truth CATE) baseada nos atributos municipais.

    tau(X) = 45 + 0.0008 * pib_per_capita + 120 * (idhm - 0.65) - 0.12 * dist_capital_km
             + 35 * senai_presence + 25 * ind_emp_share
    """
    tau = (
        45.0
        + 0.0008 * df["pib_per_capita"]
        + 120.0 * (df["idhm"] - 0.65)
        - 0.12 * df["dist_capital_km"]
        + 35.0 * df["senai_presence"]
        + 25.0 * df["ind_emp_share"]
    )
    return np.maximum(tau, 5.0)


def build_analytical_dataset(
    output_path: str = "data/processed/analytical_dataset.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Constrói a base de dados analítica completa para o benchmark municipal de Causal ML.

    Salva o DataFrame em output_path.
    """
    np.random.seed(seed)

    # 1. Gerar dados de base via DGP
    df_ibge = load_ibge_covariates()
    generate_ceara_geojson(df_ibge)

    mun_codes = df_ibge["mun_code"].tolist()

    df_caged = load_caged_data(mun_codes, seed=seed)
    df_rais = load_rais_data(mun_codes, seed=seed)
    df_treatment = build_treatment_assignment(df_ibge, seed=seed)

    # Agregar saldo do CAGED no nível município x ano
    caged_agg = (
        df_caged.groupby(["mun_code", "year"])
        .agg(
            net_job_gain=("net_job_gain", "sum"),
            total_admissions=("admissions", "sum"),
            total_separations=("separations", "sum"),
            main_cnae_code=("cnae_code", lambda x: x.value_counts().index[0]),
            main_cnae_sector=("cnae_name", lambda x: x.value_counts().index[0])
        )
        .reset_index()
    )

    # Mesclar dados
    df_merged = df_ibge.merge(df_treatment, on="mun_code")
    df_merged = df_merged.merge(df_rais, on=["mun_code", "year"])
    df_merged = df_merged.merge(caged_agg, on=["mun_code", "year"])

    # Calcular CATE verdadeiro
    df_merged["true_cate"] = compute_true_cate(df_merged)

    # Injetar efeito de tratamento no outcome: Y = Y_baseline + T * tau(X) + ruído
    df_merged["raw_net_job_gain"] = df_merged["net_job_gain"]
    treatment_effect = df_merged["fdi_incentive"] * df_merged["true_cate"]
    noise = np.random.normal(0, 15.0, len(df_merged))

    df_merged["net_job_gain"] = (df_merged["raw_net_job_gain"] + treatment_effect + noise).round(1)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out_file, index=False, encoding="utf-8")

    logger.info(
        f"Dataset analitico construido com {len(df_merged)} registros e {df_merged['mun_code'].nunique()} municipios."
    )
    logger.info(f"Salvo em {out_file}")
    return df_merged


def build_synthetic_control_panel(
    output_path: str = "data/processed/synthetic_control_panel.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Constrói a base de painel no nível estadual (2015-2025) comparando o Ceará
    contra um pool de doadores de 9 estados do Nordeste e vizinhos.
    """
    np.random.seed(seed + 10)
    years = list(range(2015, 2026))
    states = ["Ceará", "Piauí", "Maranhão", "Rio Grande do Norte", "Paraíba", "Pernambuco", "Alagoas", "Sergipe", "Bahia", "Pará"]

    records = []
    for state in states:
        is_treated_unit = 1 if state == "Ceará" else 0
        base_ind_emp = 320000 if state == "Ceará" else int(np.random.uniform(110000, 480000))
        base_pib_pc = 21000 if state == "Ceará" else float(np.random.uniform(14000, 28000))

        for year in years:
            trend = 1.0 + (year - 2015) * 0.02
            covid_factor = 0.88 if year == 2020 else 1.0

            # Impacto da política no Ceará a partir de 2021
            policy_impact = 0.0
            if is_treated_unit and year >= 2021:
                policy_impact = (year - 2020) * 8500.0

            ind_emp_total = int(base_ind_emp * trend * covid_factor + policy_impact + np.random.normal(0, 3000))
            pib_pc_year = round(base_pib_pc * trend * covid_factor + np.random.normal(0, 500), 2)

            records.append({
                "state": state,
                "year": year,
                "is_treated_unit": is_treated_unit,
                "treated_period": 1 if (is_treated_unit and year >= 2021) else 0,
                "ind_emp_total": max(ind_emp_total, 10000),
                "pib_per_capita": pib_pc_year,
                "ind_share": round(float(np.random.uniform(0.12, 0.28)), 4)
            })

    df_sc = pd.DataFrame(records)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_sc.to_csv(out_file, index=False, encoding="utf-8")

    logger.info(f"Painel para Controle Sintetico construido com {len(df_sc)} linhas. Salvo em {out_file}")
    return df_sc


if __name__ == "__main__":
    df_analytical = build_analytical_dataset()
    df_sc = build_synthetic_control_panel()
    print(df_analytical.head())
