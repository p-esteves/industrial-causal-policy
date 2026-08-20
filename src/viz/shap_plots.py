"""
Visualizações do SHAP Summary & Importância de Atributos.
---------------------------------------------------------
Gera gráficos com resolução de publicação 300 DPI explicando os determinantes do CATE:
1. Gráfico de Barras de Importância de Atributos (Média do |SHAP|)
2. Gráfico Beeswarm / Summary Plot do SHAP
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from src.viz.forest_plot import setup_dark_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_shap_importance(
    df_importance: pd.DataFrame,
    output_path: str = "results/figures/shap_feature_importance.png"
) -> None:
    """
    Gera gráfico de barras horizontais do valor médio absoluto de SHAP.
    """
    setup_dark_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    df_plot = df_importance.sort_values("mean_abs_shap", ascending=True)
    
    bars = ax.barh(
        df_plot["feature_label"],
        df_plot["mean_abs_shap"],
        color="#0284c7",
        edgecolor="#38bdf8",
        height=0.65
    )
    
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.2f}",
            va="center",
            ha="left",
            color="#f8fafc",
            fontsize=10,
            fontweight="bold"
        )
        
    ax.set_xlabel("Importância Média |SHAP| (Impacto Absoluto na Heterogeneidade do CATE)", fontsize=11, labelpad=10)
    ax.set_title("Principais Determinantes Socioeconômicos da Heterogeneidade do Efeito", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gráfico de importância SHAP salvo em {output_path}")


def plot_shap_beeswarm(
    explanation: shap.Explanation,
    output_path: str = "results/figures/shap_beeswarm_plot.png"
) -> None:
    """
    Gera gráfico beeswarm do SHAP usando a biblioteca shap.
    """
    setup_dark_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(10, 6), dpi=300)
    shap.plots.beeswarm(explanation, show=False, color_bar=True)
    plt.title("Impacto da Variável no CATE (Valores SHAP)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Gráfico beeswarm SHAP salvo em {output_path}")


if __name__ == "__main__":
    df_imp = pd.DataFrame({
        "feature_label": ["PIB per capita", "IDH-M", "Dist. Fortaleza", "Capacitação Técnica", "% Emprego Ind."],
        "mean_abs_shap": [18.4, 14.2, 11.5, 8.7, 6.3]
    })
    plot_shap_importance(df_imp)
