import json
from pathlib import Path
import joblib
import pandas as pd
import pytest

@pytest.fixture
def trained_model():
    path = Path("artifacts/data_training/model.joblib")
    if not path.exists():
        pytest.skip("model.joblib not found - skipping model test")
    return joblib.load(path)

@pytest.fixture
def test_data():
    path = Path("artifacts/data_preparation/test.parquet")
    if not path.exists():
        pytest.skip("test.parquet not found - skipping test")
    df = pd.read_parquet(path)
    X_test = df.drop(columns=["Class"])
    return X_test

def test_model_predicts(trained_model, test_data):
    sample = test_data.head(10)
    preds = trained_model.predict(sample)
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})

def test_metrics_threshold():
    metrics_path = Path("artifacts/data_training/metrics.json")
    if not metrics_path.exists():
        pytest.skip("metrics.json not found")

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Sanity checks: Model must beat random guessing by a wide margin
    assert metrics["roc_auc"] >= 0.85, f"ROC-AUC too low: {metrics['roc_auc']}"
    assert metrics["pr_auc"] >= 0.50, f"PR-AUC too low: {metrics['pr_auc']}"