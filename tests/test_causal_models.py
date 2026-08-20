"""
Testes unitários para os estimadores causais e controle sintético.
"""

import pandas as pd
import pytest
from src.causal.ipw import estimate_ipw
from src.causal.linear_dml import estimate_linear_dml
from src.causal.causal_forest import estimate_causal_forest
from src.causal.dr_learner import estimate_dr_learner
from src.causal.x_learner import estimate_x_learner
from src.causal.synthetic_control import estimate_synthetic_control

COVARIATES = [
    "pib_per_capita", "idhm", "population", "dist_capital_km",
    "senai_presence", "ind_emp_share", "n_establishments",
    "avg_industrial_wage", "urbanization_rate", "tax_revenue_per_capita"
]


@pytest.fixture
def sample_dataset():
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    return df


def test_ipw_estimator(sample_dataset):
    res = estimate_ipw(sample_dataset, COVARIATES, n_bootstrap=10)
    assert isinstance(res, dict)
    assert "ate" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert res["ci_lower"] <= res["ate"] <= res["ci_upper"]
    assert 0.0 <= res["p_value"] <= 1.0


def test_linear_dml_estimator(sample_dataset):
    res, cates = estimate_linear_dml(sample_dataset, COVARIATES)
    assert isinstance(res, dict)
    assert len(cates) == len(sample_dataset)
    assert res["ci_lower"] <= res["ate"] <= res["ci_upper"]
    assert 0.0 <= res["p_value"] <= 1.0


def test_causal_forest_estimator(sample_dataset):
    res, cates, lower, upper, model = estimate_causal_forest(sample_dataset, COVARIATES, n_estimators=100)
    assert isinstance(res, dict)
    assert len(cates) == len(sample_dataset)
    assert len(lower) == len(sample_dataset)
    assert res["ci_lower"] <= res["ate"] <= res["ci_upper"]
    assert 0.0 <= res["p_value"] <= 1.0
    # O ATE do Causal Forest deve ser positivo no DGP conhecido (efeito médio real ~85)
    assert res["ate"] > 0.0


def test_dr_learner_estimator(sample_dataset):
    res, cates = estimate_dr_learner(sample_dataset, COVARIATES)
    assert isinstance(res, dict)
    assert len(cates) == len(sample_dataset)
    assert res["ci_lower"] <= res["ate"] <= res["ci_upper"]
    assert 0.0 <= res["p_value"] <= 1.0


def test_x_learner_estimator(sample_dataset):
    res, cates = estimate_x_learner(sample_dataset, COVARIATES)
    assert isinstance(res, dict)
    assert len(cates) == len(sample_dataset)
    assert res["ci_lower"] <= res["ate"] <= res["ci_upper"]
    assert 0.0 <= res["p_value"] <= 1.0


def test_synthetic_control():
    df_sc_res, weights, summary = estimate_synthetic_control()
    assert isinstance(df_sc_res, pd.DataFrame)
    assert isinstance(weights, dict)
    assert "att_macro_jobs" in summary
    assert summary["att_macro_jobs"] > 0
