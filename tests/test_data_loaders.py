"""
Testes unitários para os módulos DGP e engenharia de dados.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.data.dgp_municipalities import load_ibge_covariates
from src.data.dgp_employment import load_caged_data
from src.data.dgp_labor_market import load_rais_data
from src.data.treatment_builder import build_treatment_assignment
from src.data.feature_engineering import build_analytical_dataset, build_synthetic_control_panel


def test_ibge_loader():
    df = load_ibge_covariates()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 160
    assert "mun_code" in df.columns
    assert "dist_capital_km" in df.columns
    assert "idhm" in df.columns
    assert df["dist_capital_km"].min() >= 0


def test_caged_loader():
    muns = [2304400, 2307650]
    df = load_caged_data(muns)
    assert isinstance(df, pd.DataFrame)
    assert "net_job_gain" in df.columns
    assert set(df["mun_code"].unique()) == set(muns)


def test_rais_loader():
    muns = [2304400, 2307650]
    df = load_rais_data(muns)
    assert isinstance(df, pd.DataFrame)
    assert "avg_industrial_wage" in df.columns


def test_feature_engineering():
    df_analytical = build_analytical_dataset()
    assert isinstance(df_analytical, pd.DataFrame)
    assert "true_cate" in df_analytical.columns
    assert "fdi_incentive" in df_analytical.columns
    assert "net_job_gain" in df_analytical.columns

    df_sc = build_synthetic_control_panel()
    assert isinstance(df_sc, pd.DataFrame)
    assert "state" in df_sc.columns
