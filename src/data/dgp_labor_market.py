"""
DGP de Estoque de Emprego e Salários Industriais (baseado na RAIS).
-------------------------------------------------------------------
Gera estoque de emprego formal industrial, número de estabelecimentos ativos
e salário médio mensal industrial por município, calibrado com estatísticas
descritivas da RAIS / MTE.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_rais_data(
    mun_codes: List[int],
    start_year: int = 2020,
    end_year: int = 2025,
    seed: int = 42
) -> pd.DataFrame:
    """
    Gera métricas anuais sintéticas de emprego industrial por município.

    Argumentos:
        mun_codes: Lista de códigos IBGE de 7 dígitos.
        start_year: Ano inicial.
        end_year: Ano final.
        seed: Semente para reprodutibilidade.

    Retorna:
        pd.DataFrame com colunas: [mun_code, year, total_emp_stock, ind_emp_stock,
                                  ind_emp_share, n_establishments, avg_industrial_wage]
    """
    np.random.seed(seed + 1)
    years = list(range(start_year, end_year + 1))
    records = []

    for mun in mun_codes:
        if mun == 2304400:  # Fortaleza
            base_total_emp = 650000
            base_ind_emp = 95000
            base_estabs = 2400
        elif mun in [2307650, 2303709, 2312908, 2307304]:  # Polos industriais
            base_total_emp = 60000
            base_ind_emp = 22000
            base_estabs = 380
        elif mun in [2305209, 2305001, 2300200]:  # Corredor industrial
            base_total_emp = 25000
            base_ind_emp = 12000
            base_estabs = 160
        else:
            base_total_emp = int(np.random.uniform(2000, 15000))
            base_ind_emp = int(base_total_emp * np.random.uniform(0.05, 0.25))
            base_estabs = int(np.clip(base_ind_emp / np.random.uniform(15, 45), 3, 120))

        base_wage = float(np.random.normal(2100.0, 350.0))

        for year in years:
            trend = 1.0 + (year - start_year) * 0.025
            total_emp = int(base_total_emp * trend * np.random.uniform(0.97, 1.03))
            ind_emp = int(base_ind_emp * trend * np.random.uniform(0.95, 1.05))
            ind_share = round(float(ind_emp / max(total_emp, 1)), 4)
            n_estabs = int(base_estabs * trend * np.random.uniform(0.98, 1.02))
            avg_wage = round(float(base_wage * trend * np.random.uniform(0.96, 1.04)), 2)

            records.append({
                "mun_code": mun,
                "year": year,
                "total_emp_stock": total_emp,
                "ind_emp_stock": ind_emp,
                "ind_emp_share": ind_share,
                "n_establishments": n_estabs,
                "avg_industrial_wage": avg_wage
            })

    df = pd.DataFrame(records)
    logger.info(f"DGP RAIS: {len(df)} registros gerados para {len(mun_codes)} municipios.")
    return df


if __name__ == "__main__":
    from src.data.dgp_municipalities import load_ibge_covariates
    muns = load_ibge_covariates()["mun_code"].tolist()
    df_rais = load_rais_data(muns)
    print(df_rais.head())
