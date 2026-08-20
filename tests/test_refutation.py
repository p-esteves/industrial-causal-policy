"""
Testes unitários para a suíte de testes de refutação.
"""

import pandas as pd
import pytest
from src.causal.refutation import run_refutation_tests


def test_refutation_suite():
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    df_ref = run_refutation_tests(df, estimated_ate=85.0)
    assert isinstance(df_ref, pd.DataFrame)
    assert len(df_ref) == 4
    assert "test_name" in df_ref.columns
    assert "conclusion" in df_ref.columns
