"""Evaluation metrics for time series forecasting.

Metric chính thức của Favorita: NWRMSLE (Normalized Weighted Root Mean Squared
Logarithmic Error). Trọng số = 1.25 cho hàng perishable, 1.0 còn lại; tính trên
log1p(clip(.,0)) -> phạt sai số tương đối, không cho giá trị âm.
"""

import numpy as np


def nwrmsle(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Normalized Weighted Root Mean Squared Logarithmic Error (Favorita metric).

    NWRMSLE = sqrt( Σ w·(log1p(pred) − log1p(true))² / Σ w ), với pred/true clip ≥ 0.
    weights=None -> RMSLE không trọng số (mọi w=1). Truyền weights (1.25 perishable /
    1.0) để khớp đúng công thức competition.
    """
    yt = np.clip(np.asarray(y_true, dtype=float), 0, None)
    yp = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    log_diff_sq = (np.log1p(yp) - np.log1p(yt)) ** 2

    if weights is None:
        w = np.ones_like(yt)
    else:
        w = np.asarray(weights, dtype=float)
    denom = w.sum()
    if denom == 0:
        return 0.0
    return float(np.sqrt(np.sum(w * log_diff_sq) / denom))


def perishable_weights(perishable: np.ndarray) -> np.ndarray:
    """Trọng số NWRMSLE: 1.25 nếu perishable, 1.0 nếu không."""
    p = np.asarray(perishable, dtype=float)
    return np.where(p > 0, 1.25, 1.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Filters zero actual values."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Compute all metrics and return as dict.

    NWRMSLE (competition) để đầu tiên (primary). RMSE (scale gốc), MAE (trực quan),
    MAPE (%) là secondary. weights = trọng số perishable cho NWRMSLE (optional).
    """
    return {
        "nwrmsle": round(nwrmsle(y_true, y_pred, weights), 6),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 6),
    }
