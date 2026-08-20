"""
Método 5: X-Learner (Künzel et al., 2019).
------------------------------------------
Meta-learner especializado para desbalanço severo entre o grupo de tratados e controles.
Imputa resultados contrafactuais usando o pool maior de controle para treinar funções de resposta individuais.
Wrapper para econml.metalearners.XLearner.
"""

import logging
from typing import Dict, List, Tuple

from econml.metalearners import XLearner
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def estimate_x_learner(
    df: pd.DataFrame,
    covariates: List[str],
    treatment_col: str = "fdi_incentive",
    outcome_col: str = "net_job_gain",
    random_state: int = 42
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Estima ATE e CATEs usando X-Learner.
    
    Retorna:
        Tupla com (dicionário de resultados, array de cates).
    """
    X = df[covariates].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    models = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    propensity_model = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    cate_models = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    
    xl = XLearner(
        models=models,
        propensity_model=propensity_model,
        cate_models=cate_models
    )
    
    xl.fit(Y, T, X=X)
    
    cates = xl.effect(X)
    ate = float(np.mean(cates))
    
    # Bootstrap CI para o ATE do X-Learner
    n = len(df)
    boot_ates = []
    np.random.seed(random_state)
    for i in range(50):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            xl_b = XLearner(
                models=GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=random_state + i),
                propensity_model=GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=random_state + i),
                cate_models=GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=random_state + i)
            )
            xl_b.fit(Y[idx], T[idx], X=X[idx])
            boot_ates.append(float(np.mean(xl_b.effect(X[idx]))))
        except Exception:
            continue
            
    if boot_ates:
        ci_lower = float(np.percentile(boot_ates, 2.5))
        ci_upper = float(np.percentile(boot_ates, 97.5))
        stderr = float(np.std(boot_ates))
    else:
        se_cates = float(np.std(cates) / np.sqrt(len(cates)))
        ci_lower = ate - (1.96 * se_cates)
        ci_upper = ate + (1.96 * se_cates)
        stderr = se_cates
    
    rmse_cate = float(np.sqrt(np.mean((df["true_cate"].values - cates) ** 2))) if "true_cate" in df else np.nan
    z_score = ate / max(stderr, 1e-8)
    p_val = float(2.0 * stats.norm.sf(abs(z_score)))
    
    results = {
        "method": "X-Learner (Künzel et al., 2019)",
        "ate": round(ate, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "stderr": round(stderr, 2),
        "p_value": round(p_val, 4),
        "rmse_cate": round(rmse_cate, 2)
    }
    
    logger.info(f"Estimação X-Learner: ATE={ate:.2f} [{ci_lower:.2f}, {ci_upper:.2f}], RMSE={rmse_cate:.2f}")
    return results, cates


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    covs = [
        "pib_per_capita", "idhm", "population", "dist_capital_km",
        "senai_presence", "ind_emp_share", "n_establishments",
        "avg_industrial_wage", "urbanization_rate", "tax_revenue_per_capita"
    ]
    res, cates = estimate_x_learner(df, covs)
    print(res)
