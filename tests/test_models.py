"""Small model-facing feature selection tests."""

import pandas as pd

from src.models.xgboost_model import XGBoostModel


def test_xgboost_respects_disabled_exog_groups():
    df = pd.DataFrame({
        "store_nbr": [1],
        "item_nbr": [10],
        "unit_sales": [2.0],
        "dcoilwtico": [50.0],
        "oil_lag_7": [49.0],
        "onpromotion": [1],
        "promo_count_7": [2],
        "perishable": [1],
        "is_holiday": [1],
        "is_event": [0],
    })
    model = XGBoostModel({"features": {
        "use_oil": False,
        "use_promo": False,
        "use_perishable": False,
        "use_holiday": False,
    }})
    cols = set(model._get_features(df).columns)
    assert cols == {"store_nbr", "item_nbr"}
