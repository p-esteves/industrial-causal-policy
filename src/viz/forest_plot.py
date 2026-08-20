"""
Módulo de Forest Plots (Qualidade de Publicação 300 DPI).
----------------------------------------------------------
Gera gráficos forest para:
1. Comparação dos métodos de benchmark (ATE +/- IC 95% para IPW, Linear DML, Causal Forest, DR Learner, X-Learner)
2. Heterogeneidade por subgrupos (CATE +/- IC 95% entre mesorregiões, quintis de IDH, divisões CNAE, faixas de distância)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paleta escura coesa
BG_COLOR = "#0f172a"
TEXT_COLOR = "#f8fafc"
PRIMARY_COLOR = "#38bdf8"
ACCENT_COLOR = "#34d399"
GRID_COLOR = "#334155"


def setup_dark_style():
    """Aplica estilo escuro de alta visibilidade para publicação."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "font.family": "sans-serif",
        "font.size": 11
    })


def plot_benchmark_forest(
    df_benchmark: pd.DataFrame,
    output_path: str = "results/figures/forest_plot_methods.png"
) -> None:
    """
    Gera forest plot comparando os 5 métodos de inferência causal.
    """
    setup_dark_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    df_plot = df_benchmark.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(df_plot))
    
    ates = df_plot["ate"].values
    ci_lowers = df_plot["ci_lower"].values
    ci_uppers = df_plot["ci_upper"].values
    methods = df_plot["method"].values
    
    xerr_lower = ates - ci_lowers
    xerr_upper = ci_uppers - ates
    
    ax.errorbar(
        ates, y_pos, xerr=[xerr_lower, xerr_upper],
        fmt="o", color=PRIMARY_COLOR, ecolor=ACCENT_COLOR,
        zorder=2, capsize=5, capthick=2, markersize=9, linewidth=2.5,
        label="ATE Estimado ± IC 95%"
    )
    
    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.7, label="Sem Efeito (Zero)")
    
    for i, (ate, low, high) in enumerate(zip(ates, ci_lowers, ci_uppers)):
        ax.text(ate + 3.0, i + 0.1, f"{ate:.1f} [{low:.1f}, {high:.1f}]", color=TEXT_COLOR, fontsize=10, fontweight="bold")
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11, fontweight="bold")
    ax.set_xlabel("Efeito Tratamento Médio (ATE) — Saldo Líquido de Empregos Industriais/Ano", fontsize=11, labelpad=10)
    ax.set_title("Benchmark de Métodos Causais: Efeito de Incentivos Fiscais no Ceará", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor=GRID_COLOR)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Forest plot do benchmark salvo em {output_path}")


def plot_subgroups_forest(
    subgroups_df: pd.DataFrame,
    output_path: str = "results/figures/forest_plot_subgroups.png"
) -> None:
    """
    Gera forest plot por subgrupos (Mesorregiões, quintis de IDH, divisões CNAE).
    """
    setup_dark_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    
    df_plot = subgroups_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(df_plot))
    
    cates = df_plot["mean_cate"].values
    ci_lowers = df_plot["ci_lower"].values
    ci_uppers = df_plot["ci_upper"].values
    subgroup_names = df_plot["subgroup_name"].values
    
    xerr_lower = cates - ci_lowers
    xerr_upper = ci_uppers - cates
    
    ax.errorbar(
        cates, y_pos, xerr=[xerr_lower, xerr_upper],
        fmt="s", color="#f59e0b", ecolor=PRIMARY_COLOR,
        capsize=4, capthick=1.8, markersize=8, linewidth=2.0,
        label="CATE Médio ± IC 95%"
    )
    
    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.7)
    
    for i, (cate, low, high) in enumerate(zip(cates, ci_lowers, ci_uppers)):
        ax.text(cate + 2.5, i + 0.08, f"{cate:.1f}", color=TEXT_COLOR, fontsize=9.5)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subgroup_names, fontsize=10)
    ax.set_xlabel("Efeito Causal Heterogêneo Médio (CATE) — Empregos Industriais/Ano", fontsize=11, labelpad=10)
    ax.set_title("Heterogeneidade de Impacto por Subgrupos Socioeconômicos", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor=GRID_COLOR)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Forest plot de subgrupos salvo em {output_path}")


if __name__ == "__main__":
    df_b = pd.DataFrame([
        {"method": "IPW", "ate": 82.4, "ci_lower": 62.1, "ci_upper": 102.7},
        {"method": "Linear DML", "ate": 88.1, "ci_lower": 71.4, "ci_upper": 104.8},
        {"method": "Causal Forest DML", "ate": 85.6, "ci_lower": 74.2, "ci_upper": 97.0},
        {"method": "DR Learner", "ate": 84.9, "ci_lower": 72.8, "ci_upper": 97.0},
        {"method": "X-Learner", "ate": 86.3, "ci_lower": 73.9, "ci_upper": 98.7}
    ])
    plot_benchmark_forest(df_b)
