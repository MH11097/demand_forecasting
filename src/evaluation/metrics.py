"""Evaluation metrics for time series forecasting.

Metric chính thức của Favorita: NWRMSLE (Normalized Weighted Root Mean Squared
Logarithmic Error). Trọng số = 1.25 cho hàng perishable, 1.0 còn lại; tính trên
log1p(clip(.,0)) -> phạt sai số tương đối, không cho giá trị âm.
"""

import numpy as np


def _validate_inputs(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reject empty or misaligned evaluation sets instead of reporting fake scores."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.size == 0 or yp.size == 0:
        raise ValueError("Cannot evaluate an empty prediction set")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {yt.shape} != {yp.shape}")
    w = None if weights is None else np.asarray(weights, dtype=float)
    if w is not None and w.shape != yt.shape:
        raise ValueError(f"weights shape mismatch: {w.shape} != {yt.shape}")
    return yt, yp, w


def nwrmsle(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Normalized Weighted Root Mean Squared Logarithmic Error (Favorita metric).

    NWRMSLE = sqrt( Σ w·(log1p(pred) − log1p(true))² / Σ w ), với pred/true clip ≥ 0.
    weights=None -> RMSLE không trọng số (mọi w=1). Truyền weights (1.25 perishable /
    1.0) để khớp đúng công thức competition.
    """
    yt, yp, w = _validate_inputs(y_true, y_pred, weights)
    yt = np.clip(yt, 0, None)
    yp = np.clip(yp, 0, None)
    log_diff_sq = (np.log1p(yp) - np.log1p(yt)) ** 2

    if w is None:
        w = np.ones_like(yt)
    denom = w.sum()
    if denom == 0:
        return 0.0
    return float(np.sqrt(np.sum(w * log_diff_sq) / denom))


def perishable_weights(perishable: np.ndarray) -> np.ndarray:
    """Trọng số NWRMSLE: 1.25 nếu perishable, 1.0 nếu không."""
    p = np.asarray(perishable, dtype=float)
    return np.where(p > 0, 1.25, 1.0)


def weights_from_frame(df) -> np.ndarray | None:
    """Return official Favorita weights when a dataframe carries perishable flags."""
    if "perishable" not in df.columns:
        return None
    return perishable_weights(df["perishable"].values)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred, _ = _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred, _ = _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Filters zero actual values."""
    y_true, y_pred, _ = _validate_inputs(y_true, y_pred)
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
