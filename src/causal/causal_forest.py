"""
Método 3: Causal Forest DML (Estimador Principal).
---------------------------------------------------
Método não-paramétrico não-linear para captura de CATEs individuais.
econml.dml.CausalForestDML com honest splitting, min_samples_leaf=5,
cross-fitting cv=5. Extrai CATEs individuais effect(X) e ICs a 95% effect_interval(X).
"""

import logging
from typing import Dict, List, Tuple

from econml.dml import CausalForestDML
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def estimate_causal_forest(
    df: pd.DataFrame,
    covariates: List[str],
    treatment_col: str = "fdi_incentive",
    outcome_col: str = "net_job_gain",
    n_estimators: int = 1500,
    min_samples_leaf: int = 5,
    cv: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, CausalForestDML]:
    """
    Estima ATE e CATEs individuais usando Causal Forest DML.
    
    Retorna:
        Tupla com (dicionário de resultados, cates array, cates_ci_lower array, cates_ci_upper array, modelo ajustado).
    """
    X = df[covariates].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    model_y = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    model_t = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    
    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        honest=True,
        cv=cv,
        discrete_treatment=True,
        random_state=random_state
    )
    
    try:
        cf.fit(Y, T, X=X, inference='auto')
        ate = float(cf.ate(X))
        ate_interval = cf.ate_interval(X, alpha=0.05)
        ci_lower = float(ate_interval[0])
        ci_upper = float(ate_interval[1])
        cates = cf.effect(X)
        cate_intervals = cf.effect_interval(X, alpha=0.05)
        cates_ci_lower = cate_intervals[0]
        cates_ci_upper = cate_intervals[1]
    except Exception as e:
        logger.warning(f"Erro na inferência analítica do CausalForestDML: {e}. Recorrendo a erros-padrão assintóticos.")
        cf.fit(Y, T, X=X)
        cates = cf.effect(X)
        ate = float(np.mean(cates))
        se_ate = float(np.std(cates) / np.sqrt(len(cates)))
        ci_lower = ate - (1.96 * se_ate)
        ci_upper = ate + (1.96 * se_ate)
        se_cates = np.std(cates)
        cates_ci_lower = cates - (1.96 * se_cates)
        cates_ci_upper = cates + (1.96 * se_cates)
    
    rmse_cate = float(np.sqrt(np.mean((df["true_cate"].values - cates) ** 2))) if "true_cate" in df else np.nan
    stderr = float((ci_upper - ci_lower) / (2 * 1.96))
    # Teste de Wald: H0: ATE = 0
    z_score = ate / max(stderr, 1e-8)
    p_val = float(2.0 * stats.norm.sf(abs(z_score)))
    
    results = {
        "method": "Causal Forest DML",
        "ate": round(ate, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "stderr": round(stderr, 2),
        "p_value": round(p_val, 4),
        "rmse_cate": round(rmse_cate, 2)
    }
    
    logger.info(
        f"Estimação Causal Forest DML: ATE={ate:.2f} [{ci_lower:.2f}, {ci_upper:.2f}], "
        f"Intervalo CATE=[{cates.min():.1f}, {cates.max():.1f}], RMSE={rmse_cate:.2f}"
    )
    return results, cates, cates_ci_lower, cates_ci_upper, cf


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    covs = [
        "pib_per_capita", "idhm", "population", "dist_capital_km",
        "senai_presence", "ind_emp_share", "n_establishments",
        "avg_industrial_wage", "urbanization_rate", "tax_revenue_per_capita"
    ]
    res, cates, lower, upper, _ = estimate_causal_forest(df, covs, n_estimators=500)
    print(res)
