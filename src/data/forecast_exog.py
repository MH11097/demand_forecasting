"""Shared forecast-time external-information contract for the educational benchmark."""

from collections.abc import Iterable

import pandas as pd

from src.data.features import fourier_cols

_HOLIDAY_COLS = ["is_holiday", "is_event"]
_PAYDAY_COLS = ["is_payday"]
_PROMO_COLS = ["onpromotion"]
_STATIC_GLOBAL_COLS = ["perishable"]
_OPTIONAL_OIL_COLS = ["dcoilwtico", "oil_lag_7"]
_HISTORY_EXACT_COLS = [
    "series_dow_avg", "store_avg", "item_avg", "family_avg", "zero_sales_last_28",
]
_HISTORY_PREFIXES = ("unit_sales_lag_", "unit_sales_rolling_")


def _enabled(feature_cfg: dict | None, key: str) -> bool:
    return (feature_cfg or {}).get(key, True)


def _unique_available(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(col for col in cols if col in df.columns))


def temporal_forecast_exog_cols(feature_cfg: dict | None = None) -> list[str]:
    """Columns that vary by forecast date and are available at forecast origin."""
    cols: list[str] = []
    if _enabled(feature_cfg, "use_fourier"):
        cols += fourier_cols()
    if _enabled(feature_cfg, "use_holiday"):
        cols += _HOLIDAY_COLS
    if _enabled(feature_cfg, "use_payday"):
        cols += _PAYDAY_COLS
    if _enabled(feature_cfg, "use_promo"):
        cols += _PROMO_COLS
    return cols


def prophet_regressor_cols(feature_cfg: dict | None = None) -> list[str]:
    """Shared temporal signals not already represented by Prophet seasonality."""
    return [
        col for col in temporal_forecast_exog_cols(feature_cfg)
        if not col.startswith("fourier_")
    ]


def global_model_feature_cols(
    df: pd.DataFrame,
    feature_cfg: dict | None = None,
    *,
    include_entity_ids: bool = False,
) -> list[str]:
    """Shared exog plus leakage-safe sales history for global models."""
    cols = ["store_nbr", "item_nbr"] if include_entity_ids else []
    cols += temporal_forecast_exog_cols(feature_cfg)
    if _enabled(feature_cfg, "use_perishable"):
        cols += _STATIC_GLOBAL_COLS
    if _enabled(feature_cfg, "use_oil"):
        cols += _OPTIONAL_OIL_COLS
    cols += [col for col in df.columns if col.startswith(_HISTORY_PREFIXES)]
    cols += _HISTORY_EXACT_COLS
    return _unique_available(df, cols)


def require_columns(df: pd.DataFrame, cols: Iterable[str], consumer: str) -> None:
    """Raise a clear error when a model cannot receive its configured exog."""
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{consumer} missing required forecast exog columns: {', '.join(missing)}"
        )
