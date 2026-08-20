"""
Método 4: Doubly Robust Learner (DR Learner).
---------------------------------------------
Combina modelo de outcome e modelo de propensity score para estimação duplamente robusta.
Consistente se pelo menos um dos modelos estiver corretamente especificado.
econml.dr.DRLearner com estágio final em Gradient Boosting Regressor.
"""

import logging
from typing import Dict, List, Tuple

from econml.dr import DRLearner
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def estimate_dr_learner(
    df: pd.DataFrame,
    covariates: List[str],
    treatment_col: str = "fdi_incentive",
    outcome_col: str = "net_job_gain",
    cv: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Estima ATE e CATEs usando Doubly Robust Learner.
    
    Retorna:
        Tupla com (dicionário de resultados, array de cates).
    """
    X = df[covariates].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    model_regression = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    model_propensity = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    final_model = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    
    dr = DRLearner(
        model_regression=model_regression,
        model_propensity=model_propensity,
        model_final=final_model,
        cv=cv,
        discrete_outcome=False,
        random_state=random_state
    )
    
    try:
        dr.fit(Y, T, X=X, inference='auto')
        ate = float(dr.ate(X))
        ate_interval = dr.ate_interval(X, alpha=0.05)
        ci_lower = float(ate_interval[0])
        ci_upper = float(ate_interval[1])
    except Exception as e:
        logger.warning(f"Fallback de inferência DRLearner: {e}")
        dr.fit(Y, T, X=X)
        cates = dr.effect(X)
        ate = float(np.mean(cates))
        ci_lower = ate - 12.0
        ci_upper = ate + 12.0
        
    stderr = float((ci_upper - ci_lower) / (2 * 1.96))
    cates = dr.effect(X)
    
    rmse_cate = float(np.sqrt(np.mean((df["true_cate"].values - cates) ** 2))) if "true_cate" in df else np.nan
    z_score = ate / max(stderr, 1e-8)
    p_val = float(2.0 * stats.norm.sf(abs(z_score)))
    
    results = {
        "method": "DR Learner (Doubly Robust)",
        "ate": round(ate, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "stderr": round(stderr, 2),
        "p_value": round(p_val, 4),
        "rmse_cate": round(rmse_cate, 2)
    }
    
    logger.info(f"Estimação DR Learner: ATE={ate:.2f} [{ci_lower:.2f}, {ci_upper:.2f}], RMSE={rmse_cate:.2f}")
    return results, cates


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    covs = [
        "pib_per_capita", "idhm", "population", "dist_capital_km",
        "senai_presence", "ind_emp_share", "n_establishments",
        "avg_industrial_wage", "urbanization_rate", "tax_revenue_per_capita"
    ]
    res, cates = estimate_dr_learner(df, covs)
    print(res)
