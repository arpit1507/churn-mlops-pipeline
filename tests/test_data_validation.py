from pathlib import Path
import pandas as pd
import pytest

@pytest.fixture
def train_df():
    path = Path("artifacts/data_preparation/train.parquet")
    if not path.exists():
        pytest.skip("train.parquet not generated yet - skipping validation")
    return pd.read_parquet(path)

@pytest.fixture
def test_df():
    path = Path("artifacts/data_preparation/test.parquet")
    if not path.exists():
        pytest.skip("test.parquet not generated yet - skipping validation")
    return pd.read_parquet(path)

def test_data_columns(train_df):
    expected_pca_cols = [f"V{i}" for i in range(1, 29)]
    for col in expected_pca_cols:
        assert col in train_df.columns, f"Missing expected column: {col}"
    assert "Class" in train_df.columns, "Missing target column 'Class'"

def test_no_null_values(train_df, test_df):
    assert train_df.isnull().sum().sum() == 0, "train.parquet contains NaN values"
    assert test_df.isnull().sum().sum() == 0, "test.parquet contains NaN values"

def test_target_is_binary(train_df):
    unique_vals = set(train_df["Class"].unique())
    assert unique_vals.issubset({0, 1}), f"Unexpected classes found: {unique_vals}"