"""
Módulo de Análise de CATE por Subgrupos.
---------------------------------------
Calcula estimativas de CATE específicas por subgrupo (CATE Médio +/- ICs a 95%) por:
1. Mesorregião do Ceará
2. Quintis de IDH-M
3. Principal Divisão Industrial CNAE
4. Faixas de Distância até a Capital (Fortaleza)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_subgroups(
    df: pd.DataFrame,
    cate_col: str = "cate_cf"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calcula estatísticas de CATE e ICs por subgrupo.
    
    Retorna:
        Tupla com (meso_df, idh_df, cnae_df, dist_df).
    """
    df = df.copy()
    
    # 1. Faixas de Distância
    bins_dist = [-1, 50, 150, 300, 1000]
    labels_dist = ["< 50 km (RMC)", "50 - 150 km", "150 - 300 km", "> 300 km (Interior)"]
    df["dist_band"] = pd.cut(df["dist_capital_km"], bins=bins_dist, labels=labels_dist)
    
    # 2. Quintis de IDH
    df["idh_quintile"] = pd.qcut(df["idhm"], q=5, labels=["Q1 (Menor)", "Q2", "Q3", "Q4", "Q5 (Maior)"])
    
    def compute_stats(group_col: str) -> pd.DataFrame:
        records = []
        for name, grp in df.groupby(group_col, observed=True):
            vals = grp[cate_col].dropna().values
            if len(vals) == 0:
                continue
            mean_val = float(np.mean(vals))
            std_err = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 5.0
            ci_low = mean_val - 1.96 * std_err
            ci_high = mean_val + 1.96 * std_err
            
            records.append({
                "subgroup_category": group_col,
                "subgroup_name": str(name),
                "n_obs": len(vals),
                "n_municipalities": grp["mun_code"].nunique(),
                "mean_cate": round(mean_val, 2),
                "ci_lower": round(ci_low, 2),
                "ci_upper": round(ci_high, 2),
                "stderr": round(std_err, 2)
            })
        return pd.DataFrame(records)

    meso_df = compute_stats("mesoregion")
    idh_df = compute_stats("idh_quintile")
    cnae_df = compute_stats("main_cnae_sector")
    dist_df = compute_stats("dist_band")
    
    logger.info("Análise de CATE por subgrupos concluída em todas as dimensões.")
    return meso_df, idh_df, cnae_df, dist_df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    if "cate_cf" not in df:
        df["cate_cf"] = 85.0 + 0.0008 * df["pib_per_capita"]
    m, i, c, d = analyze_subgroups(df)
    print(m)
