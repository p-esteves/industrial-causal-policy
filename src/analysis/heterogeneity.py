"""
Análise SHAP sobre os Determinantes do CATE.
--------------------------------------------
Calcula valores SHAP (SHapley Additive exPlanations) sobre os CATEs da Causal Forest
para identificar os principais determinantes socioeconômicos da heterogeneidade de impacto.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COVARIATES = [
    "pib_per_capita", "idhm", "population", "dist_capital_km",
    "senai_presence", "ind_emp_share", "n_establishments",
    "avg_industrial_wage", "urbanization_rate", "tax_revenue_per_capita"
]

FEATURE_LABELS_PT = {
    "pib_per_capita": "PIB per capita (R$)",
    "idhm": "IDH-M Municipal",
    "population": "População Estimada",
    "dist_capital_km": "Distância a Fortaleza (km)",
    "senai_presence": "Presença de Unidade Técnica",
    "ind_emp_share": "% Emprego Industrial",
    "n_establishments": "Nº Estabelecimentos Ind.",
    "avg_industrial_wage": "Salário Médio Ind. (R$)",
    "urbanization_rate": "Taxa de Urbanização",
    "tax_revenue_per_capita": "Receita Tributária pc"
}


def compute_cate_shap_values(
    df: pd.DataFrame,
    cate_col: str = "cate_cf",
    covariates: List[str] = COVARIATES,
    random_state: int = 42
) -> Tuple[np.ndarray, shap.Explanation, pd.DataFrame]:
    """
    Calcula valores SHAP para um modelo substituto ajustado aos CATEs.
    
    Retorna:
        Tupla com (array de shap_values, objeto shap.Explanation, df de importância de atributos).
    """
    X = df[covariates].copy()
    y_cate = df[cate_col].values
    
    surrogate = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=random_state)
    surrogate.fit(X, y_cate)
    
    explainer = shap.TreeExplainer(surrogate)
    shap_vals = explainer.shap_values(X)
    
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    df_importance = pd.DataFrame({
        "feature": covariates,
        "feature_label": [FEATURE_LABELS_PT.get(c, c) for c in covariates],
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=explainer.expected_value,
        data=X.values,
        feature_names=[FEATURE_LABELS_PT.get(c, c) for c in covariates]
    )
    
    logger.info("Valores SHAP sobre os CATEs calculados com sucesso.")
    return shap_vals, explanation, df_importance


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    if "cate_cf" not in df:
        df["cate_cf"] = 85.0 + 0.0008 * df["pib_per_capita"] + 120.0 * (df["idhm"] - 0.65)
    shap_vals, exp, df_imp = compute_cate_shap_values(df)
    print(df_imp)
