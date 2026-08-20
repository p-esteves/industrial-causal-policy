"""
Módulo de Simulação Contrafactual de Políticas (What-If Simulator).
------------------------------------------------------------------
Permite simular o impacto de intervenções hipotéticas no nível municipal (ex.: adição de unidade do SENAI,
aumento de IDH-M, melhoria logística) recalculando o CATE contrafactual e o ganho marginal de empregos.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def simulate_policy_intervention(
    df: pd.DataFrame,
    mun_code: int,
    changes: Dict[str, float],
    cate_col: str = "cate_cf"
) -> Dict[str, float]:
    """
    Simula uma intervenção contrafactual em um município específico.

    Argumentos:
        df: DataFrame analítico com CATEs e covariáveis.
        mun_code: Código IBGE de 7 dígitos do município.
        changes: Dicionário de alterações nas covariáveis (ex.: {"senai_presence": 1, "idhm": 0.70}).
        cate_col: Coluna do CATE base.

    Retorna:
        Dicionário com métricas da simulação:
        [mun_name, base_cate, simulated_cate, marginal_gain, recommended_policy_base, recommended_policy_simulated]
    """
    sub = df[df["mun_code"] == mun_code]
    if len(sub) == 0:
        raise ValueError(f"Município com código {mun_code} não encontrado no dataset.")

    row = sub.iloc[0]
    base_cate = float(row[cate_col])

    # Recalcular CATE contrafactual via variação proporcional ponderada pelas importâncias SHAP
    # tau_sim = tau_base + delta_senai * 35.0 + delta_idhm * 120.0 - delta_dist * 0.12 + ...
    delta_cate = 0.0

    if "senai_presence" in changes:
        delta_senai = changes["senai_presence"] - row["senai_presence"]
        delta_cate += delta_senai * 35.0

    if "idhm" in changes:
        delta_idh = changes["idhm"] - row["idhm"]
        delta_cate += delta_idh * 120.0

    if "dist_capital_km" in changes:
        delta_dist = changes["dist_capital_km"] - row["dist_capital_km"]
        delta_cate += delta_dist * (-0.12)

    if "pib_per_capita" in changes:
        delta_pib = changes["pib_per_capita"] - row["pib_per_capita"]
        delta_cate += delta_pib * 0.0008

    simulated_cate = max(float(base_cate + delta_cate), 0.0)
    marginal_gain = simulated_cate - base_cate

    threshold = 60.0
    rec_base = "Conceder FDI" if base_cate >= threshold else "Não Conceder"
    rec_sim = "Conceder FDI" if simulated_cate >= threshold else "Não Conceder"

    result = {
        "mun_code": mun_code,
        "mun_name": str(row["mun_name"]),
        "mesoregion": str(row["mesoregion"]),
        "base_cate": round(base_cate, 2),
        "simulated_cate": round(simulated_cate, 2),
        "marginal_gain": round(marginal_gain, 2),
        "policy_recommendation_base": rec_base,
        "policy_recommendation_simulated": rec_sim
    }

    logger.info(f"Simulação contrafactual concluída para {row['mun_name']}: {base_cate:.2f} -> {simulated_cate:.2f}")
    return result


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    res = simulate_policy_intervention(df, mun_code=2311306, changes={"senai_presence": 1})
    print(res)
