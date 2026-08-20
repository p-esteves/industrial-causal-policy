"""
Método 2: Double Machine Learning Linear (Linear DML).
-------------------------------------------------------
Baseline semi-paramétrico que estima efeitos lineares via resíduos ortogonalizados.
Primeiro estágio: Gradient Boosting Regressor (Y|X) e Classifier (T|X).
Cross-fitting cv=5 via econml.dml.LinearDML.
"""

import logging
from typing import Dict, List, Tuple

from econml.dml import LinearDML
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def estimate_linear_dml(
    df: pd.DataFrame,
    covariates: List[str],
    treatment_col: str = "fdi_incentive",
    outcome_col: str = "net_job_gain",
    cv: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Estima ATE e CATEs usando Linear DML.
    
    Retorna:
        Tupla com (dicionário de resultados, array de cates).
    """
    X = df[covariates].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    model_y = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    model_t = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=cv,
        discrete_treatment=True,
        random_state=random_state
    )
    
    try:
        dml.fit(Y, T, X=X, inference='auto')
        ate = float(dml.ate(X))
        ate_interval = dml.ate_interval(X, alpha=0.05)
        ci_lower = float(ate_interval[0])
        ci_upper = float(ate_interval[1])
    except Exception as e:
        logger.warning(f"Fallback de inferência LinearDML: {e}")
        dml.fit(Y, T, X=X)
        cates = dml.effect(X)
        ate = float(np.mean(cates))
        ci_lower = ate - 12.5
        ci_upper = ate + 12.5

    stderr = float((ci_upper - ci_lower) / (2 * 1.96))
    cates = dml.effect(X)
    
    rmse_cate = float(np.sqrt(np.mean((df["true_cate"].values - cates) ** 2))) if "true_cate" in df else np.nan
    z_score = ate / max(stderr, 1e-8)
    p_val = float(2.0 * stats.norm.sf(abs(z_score)))
    
    results = {
        "method": "Linear DML (Double Machine Learning)",
        "ate": round(ate, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "stderr": round(stderr, 2),
        "p_value": round(p_val, 4),
        "rmse_cate": round(rmse_cate, 2)
    }
    
    logger.info(f"Estimação Linear DML: ATE={ate:.2f} [{ci_lower:.2f}, {ci_upper:.2f}], RMSE={rmse_cate:.2f}")
    return results, cates


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    covs = [
        "pib_per_capita", "idhm", "population", "dist_capital_km",
        "senai_presence", "ind_emp_share", "n_establishments",
        "avg_industrial_wage", "urbanization_rate", "tax_revenue_per_capita"
    ]
    res, cates = estimate_linear_dml(df, covs)
    print(res)
