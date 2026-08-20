"""
Estimação de Propensity Score & Diagnóstico de Balanço de Covariáveis.
----------------------------------------------------------------------
Estima propensity scores e(X) = P(T=1|X) via GradientBoostingClassifier,
avalia positividade/overlap e calcula Diferenças Médias Padronizadas (SMD)
antes e depois da ponderação pelo inverso da probabilidade (Métricas do Love Plot).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fit_propensity_score(
    df: pd.DataFrame,
    covariates: List[str],
    treatment_col: str = "fdi_incentive",
    random_state: int = 42
) -> Tuple[np.ndarray, GradientBoostingClassifier]:
    """
    Ajusta modelo de propensity score e(X) = P(T=1|X).
    
    Retorna:
        Tupla com (array de propensity scores, modelo ajustado).
    """
    X = df[covariates].copy()
    y = df[treatment_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=random_state
    )
    clf.fit(X_scaled, y)
    ps = clf.predict_proba(X_scaled)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)
    
    logger.info(f"Propensity scores estimados: média={ps.mean():.4f}, min={ps.min():.4f}, máx={ps.max():.4f}")
    return ps, clf


def compute_smd(
    df: pd.DataFrame,
    covariates: List[str],
    treatment_col: str = "fdi_incentive",
    weights: np.ndarray = None
) -> pd.DataFrame:
    """
    Calcula Diferenças Médias Padronizadas (SMD) para covariáveis antes e depois da ponderação.
    
    SMD = (média_tratado - média_controle) / sqrt((var_tratado + var_controle) / 2)
    
    Retorna:
        pd.DataFrame com colunas: [covariate, smd_unweighted, smd_weighted]
    """
    treated = df[treatment_col] == 1
    control = df[treatment_col] == 0
    
    if weights is None:
        weights = np.ones(len(df))
        
    records = []
    for col in covariates:
        x_tr = df.loc[treated, col]
        x_ctrl = df.loc[control, col]
        w_tr = weights[treated]
        w_ctrl = weights[control]
        
        # Estatísticas não ponderadas
        mean_tr_u = x_tr.mean()
        mean_ctrl_u = x_ctrl.mean()
        var_tr_u = x_tr.var()
        var_ctrl_u = x_ctrl.var()
        smd_u = (mean_tr_u - mean_ctrl_u) / np.sqrt((var_tr_u + var_ctrl_u) / 2.0 + 1e-8)
        
        # Estatísticas ponderadas
        mean_tr_w = np.average(x_tr, weights=w_tr)
        mean_ctrl_w = np.average(x_ctrl, weights=w_ctrl)
        var_tr_w = np.average((x_tr - mean_tr_w) ** 2, weights=w_tr)
        var_ctrl_w = np.average((x_ctrl - mean_ctrl_w) ** 2, weights=w_ctrl)
        smd_w = (mean_tr_w - mean_ctrl_w) / np.sqrt((var_tr_w + var_ctrl_w) / 2.0 + 1e-8)
        
        records.append({
            "covariate": col,
            "smd_unweighted": abs(float(smd_u)),
            "smd_weighted": abs(float(smd_w))
        })
        
    df_smd = pd.DataFrame(records)
    logger.info("Diagnóstico de balanço de covariáveis (SMD) calculado.")
    return df_smd


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    covs = [
        "pib_per_capita", "idhm", "population", "dist_capital_km",
        "senai_presence", "ind_emp_share", "n_establishments",
        "avg_industrial_wage", "urbanization_rate", "tax_revenue_per_capita"
    ]
    ps, _ = fit_propensity_score(df, covs)
    df["ps"] = ps
    weights = np.where(df["fdi_incentive"] == 1, 1 / ps, 1 / (1 - ps))
    smd_df = compute_smd(df, covs, weights=weights)
    print(smd_df)
