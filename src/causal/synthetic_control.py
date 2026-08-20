"""
Método Complementar: Controle Sintético (Nível Estadual).
---------------------------------------------------------
Constrói um Ceará Sintético a partir de uma combinação ponderada de estados do pool de doadores (2015-2025).
Avalia o impacto macroeconômico agregado dos incentivos fiscais industriais (FDI) sobre o emprego industrial total.
Usa otimização quadrática restrita para os pesos dos doadores (w_j >= 0, soma w_j = 1).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def estimate_synthetic_control(
    panel_path: str = "data/processed/synthetic_control_panel.csv",
    treatment_year: int = 2021,
    treated_state: str = "Ceará"
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Ajusta Controle Sintético para a unidade tratada contra o pool de doadores.
    
    Retorna:
        Tupla com (df de comparação temporal, dicionário de pesos, dicionário de métricas de resumo).
    """
    df_panel = pd.read_csv(panel_path)
    
    pre_period = df_panel["year"] < treatment_year
    post_period = df_panel["year"] >= treatment_year
    
    donor_states = [s for s in df_panel["state"].unique() if s != treated_state]
    
    # Extrair matrizes de resultado no período pré-tratamento
    y_tr_pre = df_panel[(df_panel["state"] == treated_state) & pre_period]["ind_emp_total"].values
    
    y_donors_pre_list = []
    for state in donor_states:
        y_s = df_panel[(df_panel["state"] == state) & pre_period]["ind_emp_total"].values
        y_donors_pre_list.append(y_s)
    Y_donors_pre = np.column_stack(y_donors_pre_list)
    
    # Otimização restrita para minimizar o erro de ajuste pré-tratamento
    n_donors = len(donor_states)
    
    def loss_func(w):
        y_synth_pre = Y_donors_pre @ w
        return np.mean((y_tr_pre - y_synth_pre) ** 2)
        
    bounds = [(0.0, 1.0) for _ in range(n_donors)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_w = np.ones(n_donors) / n_donors
    
    res = minimize(loss_func, init_w, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = res.x
    weights_dict = {state: round(float(w), 4) for state, w in zip(donor_states, weights)}
    
    # Gerar Ceará Sintético para todo o período 2015-2025
    years = sorted(df_panel["year"].unique())
    records = []
    
    for year in years:
        y_real = float(df_panel[(df_panel["state"] == treated_state) & (df_panel["year"] == year)]["ind_emp_total"].iloc[0])
        
        y_donors_year = np.array([
            df_panel[(df_panel["state"] == state) & (df_panel["year"] == year)]["ind_emp_total"].iloc[0]
            for state in donor_states
        ])
        
        y_synth = float(y_donors_year @ weights)
        gap = y_real - y_synth
        
        records.append({
            "year": year,
            "real_ceara": y_real,
            "synthetic_ceara": round(y_synth, 1),
            "gap_effect": round(gap, 1),
            "is_post_treatment": 1 if year >= treatment_year else 0
        })
        
    df_sc_res = pd.DataFrame(records)
    
    pre_rmspe = float(np.sqrt(np.mean(df_sc_res[df_sc_res["year"] < treatment_year]["gap_effect"] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(df_sc_res[df_sc_res["year"] >= treatment_year]["gap_effect"] ** 2)))
    att_macro = float(df_sc_res[df_sc_res["year"] >= treatment_year]["gap_effect"].mean())
    
    summary_metrics = {
        "pre_rmspe": round(pre_rmspe, 1),
        "post_rmspe": round(post_rmspe, 1),
        "rmspe_ratio": round(post_rmspe / max(pre_rmspe, 1.0), 2),
        "att_macro_jobs": round(att_macro, 1)
    }
    
    logger.info(
        f"Controle Sintético ajustado: Pre-RMSPE={pre_rmspe:.1f}, Post-RMSPE={post_rmspe:.1f}, ATT={att_macro:.1f} empregos."
    )
    return df_sc_res, weights_dict, summary_metrics


if __name__ == "__main__":
    df_res, weights, summary = estimate_synthetic_control()
    print(df_res)
    print("Pesos:", weights)
    print("Métricas:", summary)
