"""Walk-forward cross-validation for time series models (Favorita schema).

Dùng cột chuẩn: date, unit_sales, series_id. Mỗi fold fit scaler riêng trên train
(RobustScaler) để tránh leakage. Metric chính: NWRMSLE (trọng số perishable).
"""

import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

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
        df: Full DataFrame đã add features (+ log transform), sorted theo series_id, date
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

        train_df = df[df[_DATE].isin(train_dates)].copy()
        test_df = df[df[_DATE].isin(test_dates)].copy()

        if eval_days is not None and eval_days > 0:
            first_n_dates = sorted(test_df[_DATE].unique())[:eval_days]
            test_df = test_df[test_df[_DATE].isin(first_n_dates)]

        train_df, test_df = _scale_per_fold(train_df, test_df, config)

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
    """Walk-forward CV dùng model đã train sẵn — bỏ retrain, chỉ predict + đánh giá.

    Context rows (seq_len + H - 1) prepend per series_id trước mỗi fold test để
    LSTM đủ lịch sử cho prediction đầu tiên.
    """
    dates = sorted(df[_DATE].unique())
    total = len(dates)
    step = total // (n_splits + 1)

    model_cfg = config.get("model", {})
    ctx_len = model_cfg.get("seq_len", 30) + model_cfg.get("forecast_horizon", 1) - 1

    fold_metrics = []
    for fold in range(n_splits):
        train_end_idx = step * (fold + 1)
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + step, total)
        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]
        if len(test_dates) == 0:
            continue

        train_df = df[df[_DATE].isin(train_dates)].copy()
        test_df = df[df[_DATE].isin(test_dates)].copy()

        if eval_days is not None and eval_days > 0:
            first_n_dates = sorted(test_df[_DATE].unique())[:eval_days]
            test_df = test_df[test_df[_DATE].isin(first_n_dates)]

        train_df, test_df = _scale_per_fold(train_df, test_df, config)

        # Prepend context per series_id -> sequence models cần seq_len ngày lịch sử
        if _GROUP in train_df.columns:
            ctx_df = train_df.groupby(_GROUP, group_keys=False).tail(ctx_len)
        else:
            ctx_df = train_df.tail(ctx_len)

        combined = pd.concat([ctx_df, test_df]).reset_index(drop=True)
        predictions = model.predict(combined)
        predictions = predictions[len(ctx_df):]

        y_true = test_df[_TARGET].values
        if config.get("use_log_sales", False):
            y_true = np.expm1(y_true.astype(float))

        metrics = evaluate_all(y_true, predictions, _fold_weights(test_df))
        metrics["fold"] = fold
        metrics["training_time_seconds"] = 0.0
        fold_metrics.append(metrics)

    return {
        "folds": fold_metrics,
        "aggregated": _aggregate(fold_metrics),
        "n_splits": len(fold_metrics),
    }
