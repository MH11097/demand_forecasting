"""Grid search và ablation study engine cho Prophet model.

Workflow: tạo tổ hợp tham số (Cartesian product) → train ProphetModel cho mỗi tổ hợp
→ thu thập NWRMSLE/RMSE/MAE/MAPE → trả DataFrame kết quả sắp theo NWRMSLE tăng dần.
"""

import copy
import itertools
import logging
import time
import warnings

import pandas as pd

from src.evaluation.metrics import evaluate_all, weights_from_frame
from src.models.prophet_model import ProphetModel

logger = logging.getLogger(__name__)


def generate_param_grid(param_space: dict[str, list]) -> list[dict]:
    """Tạo Cartesian product từ dict of lists.

    Input:  {"changepoint_prior_scale": [0.01, 0.05], "seasonality_mode": ["additive", "multiplicative"]}
    Output: [{"changepoint_prior_scale": 0.01, "seasonality_mode": "additive"}, ...]
    """
    keys = list(param_space.keys())
    values = list(param_space.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _apply_params_to_config(base_config: dict, params: dict) -> dict:
    """Deep copy config gốc rồi ghi đè tham số model. Giữ nguyên config khác (split, data...)."""
    config = copy.deepcopy(base_config)
    model_cfg = config.setdefault("model", {})
    for key, value in params.items():
        model_cfg[key] = value
    return config


def run_grid_search(
    base_config: dict,
    param_grid: list[dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Chạy ProphetModel cho mỗi tổ hợp tham số, thu thập metrics.

    Args:
        base_config: config gốc (đã load từ YAML)
        param_grid: danh sách dict tham số (từ generate_param_grid)
        train_df: dữ liệu train đã qua preprocess
        val_df: dữ liệu validation đã qua preprocess

    Returns:
        DataFrame gồm cột tham số + metrics, sắp theo NWRMSLE tăng dần
    """
    results = []
    total = len(param_grid)

    for i, params in enumerate(param_grid, 1):
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        logger.info(f"[{i}/{total}] {param_str}")

        config = _apply_params_to_config(base_config, params)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ProphetModel(config)
                start = time.time()
                model.train(train_df, val_df)
                elapsed = time.time() - start

                # Đánh giá trên validation set → metric chính là NWRMSLE (Kaggle)
                predictions = model.predict(val_df)
                y_true = val_df["unit_sales"].values
                metrics = evaluate_all(y_true, predictions, weights_from_frame(val_df))

                row = {**params, **metrics, "time_seconds": round(elapsed, 2)}
                results.append(row)
                logger.info(f"  NWRMSLE={metrics['nwrmsle']:.6f} ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"  FAILED: {e}")
            row = {**params, "nwrmsle": float("inf"), "rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "time_seconds": 0}
            results.append(row)

    df = pd.DataFrame(results).sort_values("nwrmsle").reset_index(drop=True)
    return df


def run_ablation_study(
    base_config: dict,
    regressor_combos: dict[str, list[str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Chạy ablation: cố định hyperparams tốt nhất, thay đổi tổ hợp regressors.

    Mục đích: đo đóng góp biên (marginal contribution) của từng biến ngoại sinh.

    Args:
        base_config: config với best hyperparams đã set sẵn
        regressor_combos: dict tên thí nghiệm → list regressors
            vd: {"Baseline": [], "+Promo": ["Promo"], "Full": ["Promo", "SchoolHoliday"]}
        train_df: dữ liệu train
        val_df: dữ liệu validation

    Returns:
        DataFrame với cột experiment_name, regressors, + metrics
    """
    results = []
    total = len(regressor_combos)

    for i, (exp_name, regressors) in enumerate(regressor_combos.items(), 1):
        logger.info(f"[{i}/{total}] Ablation: {exp_name} → regressors={regressors}")

        config = copy.deepcopy(base_config)
        config.setdefault("model", {})["regressors"] = regressors

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ProphetModel(config)
                start = time.time()
                model.train(train_df, val_df)
                elapsed = time.time() - start

                predictions = model.predict(val_df)
                y_true = val_df["unit_sales"].values
                metrics = evaluate_all(y_true, predictions, weights_from_frame(val_df))

                row = {
                    "experiment": exp_name,
                    "regressors": ", ".join(regressors) if regressors else "(none)",
                    **metrics,
                    "time_seconds": round(elapsed, 2),
                }
                results.append(row)
                logger.info(f"  NWRMSLE={metrics['nwrmsle']:.6f} ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"  FAILED: {e}")
            row = {
                "experiment": exp_name,
                "regressors": ", ".join(regressors) if regressors else "(none)",
                "nwrmsle": float("inf"), "rmse": float("inf"), "mae": float("inf"), "mape": float("inf"),
                "time_seconds": 0,
            }
            results.append(row)

    return pd.DataFrame(results)
