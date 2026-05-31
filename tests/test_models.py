"""Small model-facing feature selection tests."""

import numpy as np
import pandas as pd

from src.data.features import fourier_cols
from src.models.prophet_model import ProphetModel
from src.models.sarimax import SARIMAXModel
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


def test_sarimax_extracts_shared_temporal_exog_in_stable_order():
    cols = [*fourier_cols(), "is_holiday", "is_event", "is_payday", "onpromotion"]
    df = pd.DataFrame({col: [i, i + 1] for i, col in enumerate(cols)})
    model = SARIMAXModel({"features": {
        "use_fourier": True, "use_holiday": True, "use_payday": True, "use_promo": True,
    }})
    assert np.array_equal(model._get_exog(df), df[cols].values.astype(float))


def test_prophet_frame_includes_shared_non_fourier_regressors():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-01"]),
        "unit_sales": [2.0],
        "is_holiday": [1],
        "is_event": [0],
        "is_payday": [0],
        "onpromotion": [1],
    })
    model = ProphetModel({"features": {
        "use_fourier": True, "use_holiday": True, "use_payday": True, "use_promo": True,
    }})
    train = model._prepare_prophet_df(df)
    future = model._prepare_prophet_df(df, include_target=False)
    assert train.columns.tolist() == [
        "ds", "y", "is_holiday", "is_event", "is_payday", "onpromotion",
    ]
    assert future.columns.tolist() == [
        "ds", "is_holiday", "is_event", "is_payday", "onpromotion",
    ]
