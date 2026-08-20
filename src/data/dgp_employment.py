"""
DGP de Fluxos de Emprego Industrial (baseado no Novo CAGED).
-------------------------------------------------------------
Gera saldos sintéticos de emprego formal industrial (admissões - desligamentos)
por município do Ceará, setor CNAE 2.0 e ano.

Os parâmetros (escala por polo, sazonalidade, choque de 2020) são calibrados
com base em estatísticas descritivas do Novo CAGED / MTE.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Principais divisões industriais CNAE (2 dígitos)
INDUSTRIAL_CNAES: Dict[int, str] = {
    10: "Fabricação de Produtos Alimentícios",
    13: "Fabricação de Produtos Têxteis",
    14: "Confecção de Artigos do Vestuário",
    15: "Preparação de Couros e Fabricação de Calçados",
    20: "Fabricação de Produtos Químicos",
    22: "Fabricação de Produtos de Material Plástico e Borracha",
    23: "Fabricação de Produtos de Minerais Não-Metálicos",
    25: "Fabricação de Produtos de Metal",
    31: "Fabricação de Móveis"
}


def load_caged_data(
    mun_codes: List[int],
    start_year: int = 2020,
    end_year: int = 2025,
    seed: int = 42
) -> pd.DataFrame:
    """
    Gera painel sintético de saldo líquido de empregos industriais.

    Argumentos:
        mun_codes: Lista de códigos IBGE de 7 dígitos dos municípios.
        start_year: Ano inicial (2020).
        end_year: Ano final (2025).
        seed: Semente aleatória para reprodutibilidade.

    Retorna:
        pd.DataFrame com as colunas: [mun_code, year, cnae_code, cnae_name, admissions, separations, net_job_gain]
    """
    np.random.seed(seed)
    years = list(range(start_year, end_year + 1))
    records = []

    for mun in mun_codes:
        # Fator de escala econômica por município
        scale_factor = 1.0
        if mun == 2304400:  # Fortaleza
            scale_factor = 25.0
        elif mun in [2307650, 2303709, 2312908, 2307304]:  # Polos industriais
            scale_factor = 8.0
        elif mun in [2305209, 2305001, 2300200, 2304202]:  # Corredor industrial
            scale_factor = 4.0

        for year in years:
            for cnae_code, cnae_name in INDUSTRIAL_CNAES.items():
                base_admissions = int(np.random.poisson(lam=12 * scale_factor))
                # Choque de 2020 e recuperação 2021-2022
                if year == 2020:
                    base_separations = int(base_admissions * np.random.uniform(1.05, 1.35))
                elif year in [2021, 2022]:
                    base_separations = int(base_admissions * np.random.uniform(0.70, 0.92))
                else:
                    base_separations = int(base_admissions * np.random.uniform(0.85, 1.05))

                net_gain = base_admissions - base_separations
                records.append({
                    "mun_code": mun,
                    "year": year,
                    "cnae_code": cnae_code,
                    "cnae_name": cnae_name,
                    "admissions": base_admissions,
                    "separations": base_separations,
                    "net_job_gain": net_gain
                })

    df = pd.DataFrame(records)
    logger.info(f"DGP CAGED: {len(df)} registros gerados para {len(mun_codes)} municipios ({start_year}-{end_year}).")
    return df


if __name__ == "__main__":
    from src.data.dgp_municipalities import load_ibge_covariates
    muns = load_ibge_covariates()["mun_code"].tolist()
    df_caged = load_caged_data(muns)
    print(df_caged.head())
