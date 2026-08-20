"""
Módulo de Mapas Coropléticos para Municípios do Ceará.
------------------------------------------------------
Gera visualizações de mapas interativos (Folium/Plotly) e estáticos para:
1. Distribuição municipal do CATE (Efeito Causal Heterogêneo)
2. Clusters de Resposta do K-Means (Alto, Médio, Baixo Impacto)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import folium
import numpy as np
import pandas as pd
import plotly.express as px

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_cate_choropleth_map(
    df: pd.DataFrame,
    geojson_path: str = "data/geojson/ceara_municipalities.json",
    cate_col: str = "cate_cf",
    output_html_path: str = "results/figures/cate_choropleth_map.html"
) -> folium.Map:
    """
    Gera mapa coroplético interativo do Folium para os CATEs municipais.
    """
    Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
        
    df_mun = df.groupby("mun_code").agg({
        "mun_name": "first",
        "mesoregion": "first",
        cate_col: "mean"
    }).reset_index()
    df_mun["mun_code_str"] = df_mun["mun_code"].astype(str)
    
    m = folium.Map(
        location=[-5.2, -39.3],
        zoom_start=7,
        tiles="CartoDB dark_matter"
    )
    
    folium.Choropleth(
        geo_data=geojson_data,
        name="CATE (Empregos/Ano)",
        data=df_mun,
        columns=["mun_code_str", cate_col],
        key_on="feature.id",
        fill_color="YlGnBu",
        fill_opacity=0.85,
        line_opacity=0.3,
        legend_name="Efeito Causal Heterogêneo (CATE - Empregos Industriais/Ano)"
    ).add_to(m)
    
    m.save(output_html_path)
    logger.info(f"Mapa coroplético de CATE salvo em {output_html_path}")
    return m


def generate_cluster_choropleth_map(
    df: pd.DataFrame,
    geojson_path: str = "data/geojson/ceara_municipalities.json",
    cluster_col: str = "cluster_label",
    output_html_path: str = "results/figures/cluster_choropleth_map.html"
) -> folium.Map:
    """
    Gera mapa interativo do Folium para os clusters de resposta à política.
    """
    Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
        
    df_mun = df.groupby("mun_code").agg({
        "mun_name": "first",
        "cluster_id": "first",
        cluster_col: "first"
    }).reset_index()
    df_mun["mun_code_str"] = df_mun["mun_code"].astype(str)
    
    m = folium.Map(
        location=[-5.2, -39.3],
        zoom_start=7,
        tiles="CartoDB dark_matter"
    )
    
    folium.Choropleth(
        geo_data=geojson_data,
        name="Clusters de Resposta",
        data=df_mun,
        columns=["mun_code_str", "cluster_id"],
        key_on="feature.id",
        fill_color="PuBuGn",
        fill_opacity=0.85,
        line_opacity=0.3,
        legend_name="Perfil de Resposta à Política (Cluster)"
    ).add_to(m)
    
    m.save(output_html_path)
    logger.info(f"Mapa coroplético de Clusters salvo em {output_html_path}")
    return m


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    if "cate_cf" not in df:
        df["cate_cf"] = 85.0 + 0.0008 * df["pib_per_capita"]
    if "cluster_label" not in df:
        df["cluster_id"] = 1
        df["cluster_label"] = "Médio Impacto"
    generate_cate_choropleth_map(df)
    generate_cluster_choropleth_map(df)
