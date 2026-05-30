"""Walk-forward cross-validation for time series models (Favorita schema).

Dữ liệu đầu vào là cleaned panel trước feature engineering. Mỗi fold dựng fixed
forecast panel, fit target-derived features tại origin và fit scaler riêng trên train.
"""

import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.data.features import add_all_features, apply_log_transform
from src.data.loader import build_forecast_panel
from src.data.preprocessor import _get_numeric_feature_cols
from src.evaluation.metrics import evaluate_all, perishable_weights

_DATE = "date"
_TARGET = "unit_sales"
_GROUP = "series_id"
_METRIC_KEYS = ["nwrmsle", "rmse", "mae", "mape"]


def _fold_weights(test_df: pd.DataFrame) -> np.ndarray | None:
    """Trọng số NWRMSLE theo cờ perishable của test fold (None nếu không có cột)."""
    if "perishable" in test_df.columns:
        return perishable_weights(test_df["perishable"].values)
    return None


def _scale_per_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict):
    """fillna + scale feature liên tục bằng scaler fit trên train (skip nếu tree-based)."""
    train_df = train_df.fillna(0).copy()
    test_df = test_df.fillna(0).copy()
    skip_scaling = config.get("model", {}).get("skip_scaling", False)
    numeric_cols = _get_numeric_feature_cols(train_df)
    if numeric_cols and not skip_scaling:
        scaler = RobustScaler()
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    return train_df, test_df


def _prepare_fold(
    df: pd.DataFrame,
    config: dict,
    train_dates: list,
    test_dates: list,
    eval_days: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fold-local panel and origin-safe features."""
    origin = pd.Timestamp(max(train_dates))
    first_test = pd.Timestamp(min(test_dates))
    last_test = pd.Timestamp(max(test_dates))
    fold_df = build_forecast_panel(
        df,
        train_end=origin,
        forecast_start=first_test,
        forecast_end=last_test,
        lookback_days=config.get("split", {}).get("panel_lookback_days", 90),
    )
    fold_df = add_all_features(
        fold_df,
        feature_cfg=config.get("features", {}),
        train_end=origin,
    )
    if config.get("use_log_sales", False):
        fold_df = apply_log_transform(fold_df)

    train_df = fold_df[fold_df[_DATE].isin(train_dates)].copy()
    test_df = fold_df[fold_df[_DATE].isin(test_dates)].copy()
    if eval_days is not None and eval_days > 0:
        first_n_dates = sorted(test_df[_DATE].unique())[:eval_days]
        test_df = test_df[test_df[_DATE].isin(first_n_dates)]
    return _scale_per_fold(train_df, test_df, config)


def _aggregate(fold_metrics: list[dict]) -> dict:
    """mean/std qua các fold cho từng metric."""
    aggregated = {}
    for key in _METRIC_KEYS:
        values = [m[key] for m in fold_metrics if key in m]
        if values:
            aggregated[f"{key}_mean"] = round(float(np.mean(values)), 6)
            aggregated[f"{key}_std"] = round(float(np.std(values)), 6)
    return aggregated


def walk_forward_cv(
    model_class,
    config: dict,
    df: pd.DataFrame,
    n_splits: int = 5,
    expanding: bool = True,
    eval_days: int = None,
) -> dict:
    """Walk-forward expanding/sliding window CV — retrain model mỗi fold.

    Args:
        model_class: BaseModel subclass khởi tạo lại mỗi fold
        config: Model config dict
        df: Cleaned DataFrame trước feature engineering, sorted theo series_id, date
        n_splits: số fold
        expanding: True = expanding window; False = sliding window
        eval_days: giới hạn N ngày đầu mỗi test fold (None = toàn bộ)
    """
    dates = sorted(df[_DATE].unique())
    total = len(dates)
    step = total // (n_splits + 1)

    fold_metrics = []
    for fold in range(n_splits):
        if expanding:
            train_end_idx = step * (fold + 1)
            train_dates = dates[:train_end_idx]
        else:
            train_start_idx = step * fold
            train_end_idx = step * (fold + 1)
            train_dates = dates[train_start_idx:train_end_idx]

        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + step, total)
        test_dates = dates[test_start_idx:test_end_idx]
        if len(test_dates) == 0:
            continue

        train_df, test_df = _prepare_fold(df, config, train_dates, test_dates, eval_days)

        model = model_class(config)
        start = time.time()
        model.train(train_df)
        train_time = time.time() - start

        predictions = model.predict(test_df)
        y_true = test_df[_TARGET].values
        # use_log_sales: target ở log1p space, predict() đã expm1 -> inverse y_true để cùng scale
        if config.get("use_log_sales", False):
            y_true = np.expm1(y_true.astype(float))

        metrics = evaluate_all(y_true, predictions, _fold_weights(test_df))
        metrics["fold"] = fold
        metrics["training_time_seconds"] = round(train_time, 2)
        fold_metrics.append(metrics)

    return {
        "folds": fold_metrics,
        "aggregated": _aggregate(fold_metrics),
        "n_splits": len(fold_metrics),
    }


def walk_forward_cv_pretrained(
    model,
    config: dict,
    df: pd.DataFrame,
    n_splits: int = 5,
    expanding: bool = True,
    eval_days: int = None,
) -> dict:
    """Reject pretrained CV: a model fit after a historical fold would leak future data."""
    raise ValueError(
        "Pretrained walk-forward CV is not valid because the saved model has seen data "
        "after historical folds. Run CV without --run-dir so each fold retrains."
    )
