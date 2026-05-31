"""Tests for the educational shared forecast-exog contract."""

import pandas as pd
import pytest

from src.data.features import fourier_cols
from src.data.forecast_exog import (
    global_model_feature_cols,
    prophet_regressor_cols,
    require_columns,
    temporal_forecast_exog_cols,
)


def _feature_cfg():
    return {
        "use_fourier": True,
        "use_holiday": True,
        "use_payday": True,
        "use_promo": True,
        "use_perishable": True,
        "use_oil": False,
    }


def test_temporal_contract_is_stable_and_explainable():
    assert temporal_forecast_exog_cols(_feature_cfg()) == [
        *fourier_cols(),
        "is_holiday", "is_event", "is_payday", "onpromotion",
    ]


def test_prophet_uses_native_seasonality_instead_of_explicit_fourier():
    assert prophet_regressor_cols(_feature_cfg()) == [
        "is_holiday", "is_event", "is_payday", "onpromotion",
    ]


def test_global_models_use_shared_exog_plus_sales_history_only():
    df = pd.DataFrame(columns=[
        "store_nbr", "item_nbr", "fourier_w_sin_1", "is_holiday", "is_event",
        "is_payday", "onpromotion", "perishable", "unit_sales_lag_1",
        "unit_sales_rolling_mean_7", "series_dow_avg", "zero_sales_last_28",
        "dcoilwtico", "oil_lag_7", "promo_count_7", "month", "is_imputed",
    ])
    cols = global_model_feature_cols(df, _feature_cfg(), include_entity_ids=True)
    assert cols == [
        "store_nbr", "item_nbr", "fourier_w_sin_1", "is_holiday", "is_event",
        "is_payday", "onpromotion", "perishable", "unit_sales_lag_1",
        "unit_sales_rolling_mean_7", "series_dow_avg", "zero_sales_last_28",
    ]


def test_global_models_can_opt_into_oil_for_follow_up_experiments():
    df = pd.DataFrame(columns=["dcoilwtico", "oil_lag_7"])
    cfg = {**_feature_cfg(), "use_oil": True}
    assert global_model_feature_cols(df, cfg) == ["dcoilwtico", "oil_lag_7"]


def test_required_columns_fail_clearly():
    with pytest.raises(ValueError, match="SARIMAX missing required forecast exog columns: is_payday"):
        require_columns(pd.DataFrame({"onpromotion": [0]}), ["onpromotion", "is_payday"], "SARIMAX")
