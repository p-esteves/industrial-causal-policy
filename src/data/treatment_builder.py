"""
Construtor da Variável de Tratamento (Incentivo Fiscal FDI / ICMS).
------------------------------------------------------------------
Constrói a variável dummy binária de tratamento (T) para concessão de incentivos fiscais industriais
via Fundo de Desenvolvimento Industrial (FDI) do Ceará no nível município x ano.
Modela o viés de seleção realista com base nas covariáveis socioeconômicas.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Municípios historicamente beneficiados com alta probabilidade de concessão do FDI
HIGH_PROBABILITY_FDI_MUNS: List[int] = [
    2307650,  # Maracanaú
    2305209,  # Horizonte
    2309607,  # Pacajus
    2303709,  # Caucaia
    2305001,  # Eusébio
    2300200,  # Aquiraz
    2312908,  # Sobral
    2307304,  # Juazeiro do Norte
    2311801,  # Russas
    2304202,  # Crato
    2305506,  # Iguatu
    2311306,  # Quixadá
]


def build_treatment_assignment(
    df_ibge: pd.DataFrame,
    start_year: int = 2020,
    end_year: int = 2025,
    seed: int = 42
) -> pd.DataFrame:
    """
    Gera a atribuição de tratamento para a política de ICMS/FDI com viés de seleção realista.
    
    Argumentos:
        df_ibge: DataFrame de covariáveis do IBGE.
        start_year: Ano inicial.
        end_year: Ano final.
        seed: Semente aleatória.
        
    Retorna:
        pd.DataFrame com colunas: [mun_code, year, fdi_incentive, propensity_score_true]
    """
    np.random.seed(seed)
    years = list(range(start_year, end_year + 1))
    records = []
    
    for _, row in df_ibge.iterrows():
        mun = int(row["mun_code"])
        pib_pc = row["pib_per_capita"]
        idhm = row["idhm"]
        dist_km = row["dist_capital_km"]
        senai = row["senai_presence"]
        
        # Mecanismo de confundimento: logit do propensity score
        # O tratamento é mais provável perto da capital, maior PIB per capita, presença técnica e maior IDH
        logit = (
            -3.2
            + 0.000035 * pib_pc
            + 3.5 * (idhm - 0.6)
            - 0.005 * dist_km
            + 1.2 * senai
        )
        
        if mun in HIGH_PROBABILITY_FDI_MUNS:
            logit += 1.8
            
        ps_true = 1.0 / (1.0 + np.exp(-logit))
        ps_true = float(np.clip(ps_true, 0.05, 0.92))
        
        # Definir ano de adoção para entrada escalonada da política (a partir de 2021)
        treated = 1 if np.random.rand() < ps_true else 0
        adoption_year = 2021 if treated else 9999
        
        for year in years:
            is_treated = 1 if (treated and year >= adoption_year) else 0
            records.append({
                "mun_code": mun,
                "year": year,
                "fdi_incentive": is_treated,
                "propensity_score_true": round(ps_true, 4)
            })
            
    df_treatment = pd.DataFrame(records)
    
    n_treated_muns = df_treatment[df_treatment["year"] == end_year]["fdi_incentive"].sum()
    logger.info(
        f"Atribuição de tratamento gerada: {n_treated_muns} de {len(df_ibge)} municípios tratados até {end_year}."
    )
    return df_treatment


if __name__ == "__main__":
    from src.data.dgp_municipalities import load_ibge_covariates
    df_covs = load_ibge_covariates()
    df_treat = build_treatment_assignment(df_covs)
    print(df_treat.head())
