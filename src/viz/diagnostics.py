"""
Módulo de Gráficos de Diagnóstico & Balanço (300 DPI).
------------------------------------------------------
Gera visualizações de diagnóstico para as premissas de identificação causal:
1. Propensity Score Overlap Plot (Distribuição entre Tratados vs Controles)
2. Love Plot (Diferença Média Padronizada SMD pré vs pós ponderação)
3. Gráfico Resumo dos Testes de Refutação DoWhy
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.viz.forest_plot import setup_dark_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_overlap_distribution(
    df: pd.DataFrame,
    ps_col: str = "ps",
    treatment_col: str = "fdi_incentive",
    output_path: str = "results/figures/overlap_propensity_plot.png"
) -> None:
    """
    Gera gráfico de densidade (KDE) para sobreposição do Propensity Score (Overlap Plot).
    """
    setup_dark_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    treated = df[df[treatment_col] == 1][ps_col]
    control = df[df[treatment_col] == 0][ps_col]
    
    sns.kdeplot(treated, ax=ax, color="#38bdf8", fill=True, alpha=0.4, linewidth=2.2, label="Municípios Tratados (Com FDI)")
    sns.kdeplot(control, ax=ax, color="#f59e0b", fill=True, alpha=0.4, linewidth=2.2, label="Municípios Controle (Sem FDI)")
    
    ax.set_xlabel("Propensity Score e(X) = P(Tratado | Covariáveis)", fontsize=11, labelpad=10)
    ax.set_ylabel("Densidade", fontsize=11)
    ax.set_title("Diagnóstico de Sobreposição (Overlap / Positivity Assumption)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gráfico de sobreposição salvo em {output_path}")


def plot_love_plot(
    df_smd: pd.DataFrame,
    output_path: str = "results/figures/love_plot_smd.png"
) -> None:
    """
    Gera Love Plot com Diferenças Médias Padronizadas antes e depois da ponderação.
    """
    setup_dark_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    
    df_plot = df_smd.sort_values("smd_unweighted", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_plot))
    
    ax.scatter(df_plot["smd_unweighted"], y_pos, color="#ef4444", s=70, label="Pré-Ponderação (Bruto)", zorder=3)
    ax.scatter(df_plot["smd_weighted"], y_pos, color="#34d399", s=70, marker="D", label="Pós-Ponderação (IPW)", zorder=3)
    
    for i in range(len(df_plot)):
        ax.plot(
            [df_plot.loc[i, "smd_unweighted"], df_plot.loc[i, "smd_weighted"]],
            [i, i],
            color="#64748b", linestyle=":", linewidth=1.5
        )
        
    ax.axvline(0.1, color="#f59e0b", linestyle="--", linewidth=1.5, label="Limiar de Equilíbrio (SMD = 0.1)")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["covariate"], fontsize=10)
    ax.set_xlabel("Diferença Média Padronizada Absoluta (|SMD|)", fontsize=11, labelpad=10)
    ax.set_title("Love Plot: Balanço de Covariáveis Pré e Pós Ponderação IPW", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Love plot salvo em {output_path}")


def plot_refutation_summary(
    df_refutation: pd.DataFrame,
    output_path: str = "results/figures/refutation_summary.png"
) -> None:
    """
    Gera gráfico resumo dos testes de refutação econométrica.
    """
    setup_dark_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    df_plot = df_refutation.copy()
    y_pos = np.arange(len(df_plot))
    
    orig = df_plot["original_effect"].values
    ref = df_plot["refuted_effect"].values
    tests = df_plot["test_name"].values
    
    width = 0.35
    ax.barh(y_pos - width/2, orig, width, label="Efeito Original (CF DML)", color="#38bdf8")
    ax.barh(y_pos + width/2, ref, width, label="Efeito Refutado", color="#34d399")
    
    ax.axvline(0, color="#ef4444", linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tests, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Efeito Estimado (Empregos Industriais/Ano)", fontsize=11, labelpad=10)
    ax.set_title("Resumo dos Testes de Refutação Econométrica (DoWhy)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gráfico de refutação salvo em {output_path}")


if __name__ == "__main__":
    df_smd = pd.DataFrame({
        "covariate": ["PIB per capita", "IDH-M", "Dist. Fortaleza", "Capacitação Técnica", "Urbanização"],
        "smd_unweighted": [0.45, 0.38, 0.29, 0.52, 0.31],
        "smd_weighted": [0.04, 0.03, 0.05, 0.06, 0.02]
    })
    plot_love_plot(df_smd)
