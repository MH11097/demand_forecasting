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
    "series_dow_avg",
    "store_avg",
    "item_avg",
    "family_avg",
    "zero_sales_last_28",
]
# Phase 1a & 1b: future promo features + conditional promo stats
_PROMO_FEATURES_COLS = [
    "promo_rolling_rate",
    "promo_count_7",
    "promo_count_14",
    "promo_count_30",
    "promo_days_since_last_promo",
    "days_until_next_promo",
    "promo_count_7_future",
    "promo_count_14_future",
    "promo_days_ahead",
    "days_since_last_promo",  # ensure backward compat
]
_CONDITIONAL_PROMO_COLS = [
    "sales_with_promo_mean_7",
    "sales_with_promo_std_7",
    "sales_no_promo_mean_7",
    "sales_no_promo_std_7",
    "sales_with_promo_mean_14",
    "sales_with_promo_std_14",
    "sales_no_promo_mean_14",
    "sales_no_promo_std_14",
    "sales_with_promo_mean_30",
    "sales_with_promo_std_30",
    "sales_no_promo_mean_30",
    "sales_no_promo_std_30",
]
_HISTORY_PREFIXES = (
    "unit_sales_lag_",
    "unit_sales_rolling_",
    # Phase 2: item-level and family-level aggregates
    "item_unit_sales_lag_",
    "item_unit_sales_rolling_",
    "item_sales_with_promo_",
    "item_sales_no_promo_",
    "item_promo_",
    "family_unit_sales_lag_",
    "family_unit_sales_rolling_",
    "family_sales_with_promo_",
    "family_sales_no_promo_",
    "family_promo_",
)


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
        col
        for col in temporal_forecast_exog_cols(feature_cfg)
        if not col.startswith("fourier_")
    ]


def global_model_feature_cols(
    df: pd.DataFrame,
    feature_cfg: dict | None = None,
    *,
    include_entity_ids: bool = False,
) -> list[str]:
    """Shared exog plus leakage-safe sales history for global models.

    Includes Phase 1 & 2 features:
    - Phase 1a: Future promo lookahead (promo_count_*_future, promo_days_ahead)
    - Phase 1b: Conditional promo stats (sales_with/no_promo_mean/std_*)
    - Phase 2a: Item-level aggregates (item_unit_sales_lag_*, item_sales_with/no_promo_*, item_promo_*)
    - Phase 2b: Family-level aggregates (family_unit_sales_lag_*, family_sales_with/no_promo_*, family_promo_*)
    """
    cols = ["store_nbr", "item_nbr"] if include_entity_ids else []
    cols += temporal_forecast_exog_cols(feature_cfg)
    if _enabled(feature_cfg, "use_perishable"):
        cols += _STATIC_GLOBAL_COLS
    if _enabled(feature_cfg, "use_oil"):
        cols += _OPTIONAL_OIL_COLS
    if _enabled(feature_cfg, "use_promo"):
        cols += _unique_available(df, _PROMO_FEATURES_COLS)
        cols += _unique_available(df, _CONDITIONAL_PROMO_COLS)
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
