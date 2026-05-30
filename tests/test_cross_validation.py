"""Integration tests for rolling-origin cross-validation."""

import numpy as np
import pandas as pd

from src.evaluation.cross_validation import walk_forward_cv


class _MeanModel:
    def __init__(self, config):
        self.mean = 0.0

    def train(self, train_df, val_df=None):
        self.mean = float(train_df["unit_sales"].mean())
        return {}

    def predict(self, df):
        return np.full(len(df), self.mean)


def test_walk_forward_cv_builds_fold_local_features():
    rows = []
    for store in (1, 2):
        for item in (10, 20):
            for date in pd.date_range("2017-01-01", periods=12):
                rows.append({
                    "date": date,
                    "store_nbr": store,
                    "item_nbr": item,
                    "series_id": (store - 1) * 2 + (item == 20),
                    "unit_sales": float(store + item // 10),
                    "onpromotion": 0,
                    "perishable": int(item == 10),
                    "is_imputed": 0,
                })
    df = pd.DataFrame(rows).sort_values(["series_id", "date"]).reset_index(drop=True)
    config = {
        "features": {
            "use_time": True, "use_fourier": False, "use_lag": True,
            "lag_windows": [1], "use_rolling": True, "rolling_windows": [3],
            "rolling_stats": ["mean"], "use_group_mean": True,
            "use_payday": False, "use_zero_sales": True, "use_promo": False,
            "use_oil": False, "use_holiday": False, "use_perishable": True,
        },
        "split": {"panel_lookback_days": 30},
        "model": {"skip_scaling": True},
    }
    result = walk_forward_cv(_MeanModel, config, df, n_splits=2, eval_days=2)
    assert result["n_splits"] == 2
    assert result["aggregated"]["nwrmsle_mean"] > 0
