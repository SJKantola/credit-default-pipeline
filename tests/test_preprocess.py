import pandas as pd
from src.data.preprocess import load_and_clean_data

def test_load_and_clean_data():
    X, y, numeric_cols = load_and_clean_data("data/raw/credit_default.xls")

    # Check shapes
    assert X.shape == (30000, 23), f"Expected (30000, 23) but got {X.shape}"
    assert y.shape == (30000,)

    # Check no undocumented values remain
    assert set(X['education'].unique()).issubset({1, 2, 3, 4})
    assert set(X['marital_status'].unique()).issubset({1, 2, 3})

    # Check sex is binary 0/1
    assert set(X['sex'].unique()) == {0, 1}

    # Check target is binary
    assert set(y.unique()) == {0, 1}

    # Check 'id' is gone
    assert 'id' not in X.columns

    # Check numeric_cols has 23 columns (all features)
    assert len(numeric_cols) == 23