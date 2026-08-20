"""
Módulo do Gráfico da Trajetória do Controle Sintético (300 DPI).
---------------------------------------------------------------
Plota as trajetórias de emprego industrial do Ceará Real vs Ceará Sintético ao longo do tempo (2015–2025),
destacando o início da política de FDI em 2021 e o efeito causal acumulado.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.viz.forest_plot import setup_dark_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_synthetic_control_trajectory(
    df_sc_res: pd.DataFrame,
    treatment_year: int = 2021,
    output_path: str = "results/figures/synthetic_control_timeline.png"
) -> None:
    """
    Gera gráfico de linha temporal para o Controle Sintético.
    """
    setup_dark_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    years = df_sc_res["year"].values
    real = df_sc_res["real_ceara"].values / 1000.0
    synth = df_sc_res["synthetic_ceara"].values / 1000.0
    gap = df_sc_res["gap_effect"].values
    
    # 1. Gráfico Principal de Trajetória
    ax1.plot(years, real, color="#38bdf8", marker="o", linewidth=2.8, label="Ceará Real (Observado)")
    ax1.plot(years, synth, color="#f59e0b", marker="s", linestyle="--", linewidth=2.5, label="Ceará Sintético (Contrafactual)")
    
    ax1.axvline(treatment_year - 0.5, color="#ef4444", linestyle=":", linewidth=2, label=f"Início da Política FDI ({treatment_year})")
    
    ax1.set_ylabel("Emprego Industrial Total (Milhares)", fontsize=11, labelpad=10)
    ax1.set_title("Avaliação Macroeconômica: Ceará Real vs. Ceará Sintético (2015–2025)", fontsize=13, fontweight="bold", pad=15)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155")
    
    ax1.axvspan(treatment_year - 0.5, years.max() + 0.5, color="#0284c7", alpha=0.08)
    
    # 2. Gráfico do Efeito Causal Líquido (Gap)
    ax2.plot(years, gap, color="#34d399", marker="d", linewidth=2.2)
    ax2.axhline(0, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax2.axvline(treatment_year - 0.5, color="#ef4444", linestyle=":", linewidth=2)
    ax2.axvspan(treatment_year - 0.5, years.max() + 0.5, color="#0284c7", alpha=0.08)
    
    ax2.set_xlabel("Ano", fontsize=11, labelpad=8)
    ax2.set_ylabel("Efeito Causal Líquido (Gap)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gráfico do Controle Sintético salvo em {output_path}")


if __name__ == "__main__":
    df_sc = pd.DataFrame({
        "year": list(range(2015, 2026)),
        "real_ceara": [320000, 325000, 328000, 332000, 335000, 310000, 330000, 345000, 358000, 370000, 382000],
        "synthetic_ceara": [319000, 326000, 327000, 331000, 334000, 311000, 321000, 328000, 334000, 339000, 345000],
        "gap_effect": [1000, -1000, 1000, 1000, 1000, -1000, 9000, 17000, 24000, 31000, 37000]
    })
    plot_synthetic_control_trajectory(df_sc)
