"""
Módulo de Árvore de Decisão de Política (Policy Learning).
---------------------------------------------------------
Ajusta uma árvore de decisão interpretável mapeando covariáveis municipais X
em regras ótimas de alocação de incentivos fiscais FDI para maximizar a criação de empregos.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COVARIATES = [
    "pib_per_capita", "idhm", "population", "dist_capital_km",
    "senai_presence", "ind_emp_share", "n_establishments",
    "avg_industrial_wage", "urbanization_rate", "tax_revenue_per_capita"
]


def fit_policy_tree(
    df: pd.DataFrame,
    cate_col: str = "cate_cf",
    covariates: List[str] = COVARIATES,
    cate_threshold: float = 60.0,
    max_depth: int = 3,
    random_state: int = 42
) -> Tuple[DecisionTreeClassifier, str, pd.DataFrame]:
    """
    Ajusta Policy Tree interpretável recomendando a alocação de incentivos FDI.
    
    Argumentos:
        df: DataFrame analítico com CATEs estimados.
        cate_threshold: CATE mínimo esperado para recomendar o tratamento.
        max_depth: Profundidade da árvore de decisão.
        
    Retorna:
        Tupla com (DecisionTreeClassifier ajustado, texto da árvore, df de recomendações municipais).
    """
    X = df[covariates].copy()
    y_policy = (df[cate_col] >= cate_threshold).astype(int)
    
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5, random_state=random_state)
    tree.fit(X, y_policy)
    
    tree_text = export_text(tree, feature_names=covariates)
    
    df_rec = df[["mun_code", "mun_name", "mesoregion", cate_col]].copy()
    df_rec["recommended_treatment"] = tree.predict(X)
    df_rec["policy_recommendation"] = np.where(
        df_rec["recommended_treatment"] == 1,
        "Conceder Incentivo FDI (Alto Retorno)",
        "Não Conceder (Retorno Baixo/Insuficiente)"
    )
    
    logger.info("Policy Tree ajustada com sucesso.")
    return tree, tree_text, df_rec


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    if "cate_cf" not in df:
        df["cate_cf"] = 85.0 + 0.0008 * df["pib_per_capita"]
    tree, text, recs = fit_policy_tree(df)
    print(text)
    print(recs.head())
