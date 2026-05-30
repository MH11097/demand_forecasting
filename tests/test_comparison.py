"""Tests for persisted run comparison."""

from src.evaluation.comparison import comparison_table


def test_comparison_prefers_final_test_metrics():
    results = [{
        "model_name": "xgboost",
        "run_id": "run-1",
        "metrics": {},
        "test_metrics": {"nwrmsle": 0.42, "rmse": 1.0},
    }]
    table = comparison_table(results)
    assert table.loc[0, "evaluation_split"] == "test"
    assert table.loc[0, "nwrmsle"] == 0.42
