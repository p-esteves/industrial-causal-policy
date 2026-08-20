"""
Módulo de Agrupamento de Respostas de CATE (K-Means).
------------------------------------------------------
Agrupa os municípios do Ceará em k perfis de resposta à política com base nos CATEs estimados
(ex.: Alto Impacto, Médio Impacto, Baixo/Neutro Impacto).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cluster_cate_responses(
    df: pd.DataFrame,
    cate_col: str = "cate_cf",
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrupa estimativas de CATE em perfis de resposta.
    
    Retorna:
        Tupla com (df com atribuição de clusters, df de estatísticas de resumo).
    """
    cates = df[cate_col].values.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(cates)
    
    means = [cates[clusters == i].mean() for i in range(n_clusters)]
    rank_order = np.argsort(means)
    rank_map = {old_id: new_id for new_id, old_id in enumerate(rank_order)}
    
    sorted_clusters = np.array([rank_map[c] for c in clusters])
    
    labels_map = {
        0: "Baixo / Neutro Impacto",
        1: "Médio Impacto",
        2: "Alto Impacto",
        3: "Muito Alto Impacto"
    }
    
    df["cluster_id"] = sorted_clusters
    df["cluster_label"] = df["cluster_id"].map(labels_map)
    
    summary_list = []
    for cid in range(n_clusters):
        sub = df[df["cluster_id"] == cid]
        summary_list.append({
            "cluster_id": cid,
            "cluster_label": labels_map.get(cid, f"Cluster {cid}"),
            "n_municipalities": sub["mun_code"].nunique(),
            "mean_cate": round(float(sub[cate_col].mean()), 2),
            "min_cate": round(float(sub[cate_col].min()), 2),
            "max_cate": round(float(sub[cate_col].max()), 2),
            "mean_idhm": round(float(sub["idhm"].mean()), 3),
            "mean_pib_pc": round(float(sub["pib_per_capita"].mean()), 2),
            "mean_dist_km": round(float(sub["dist_capital_km"].mean()), 1)
        })
        
    df_summary = pd.DataFrame(summary_list)
    logger.info(f"Agrupamento de respostas CATE concluído: {n_clusters} perfis criados.")
    return df, df_summary


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    if "cate_cf" not in df:
        df["cate_cf"] = 85.0 + 0.0008 * df["pib_per_capita"]
    df, summary = cluster_cate_responses(df)
    print(summary)
