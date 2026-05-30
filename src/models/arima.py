"""ARIMA model — per-series fitting, univariate sales."""

import logging
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class ARIMAModel(BaseModel):
    name = "arima"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.order = tuple(model_cfg.get("order", [1, 1, 1]))
        # trend kiểm soát hằng số/xu hướng: 'n'=không, 'c'=hằng số, 't'=tuyến tính
        self.trend = model_cfg.get("trend", "c")
        # dict keyed by series_id (factorize của cặp store_nbr/item_nbr)
        self.models: dict[int, object] = {}
        # lưu độ dài train mỗi chuỗi → predict() dùng index tuyệt đối thay vì forecast()
        self.train_lengths: dict[int, int] = {}

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        start = time.time()
        series_ids = sorted(train_df["series_id"].unique())
        # ARIMA univariate → fit riêng từng chuỗi series_id
        failed = 0
        for sid in series_ids:
            series_data = train_df[train_df["series_id"] == sid].sort_values("date")
            sales = series_data["unit_sales"].values.astype(float)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = StatsARIMA(
                        sales,
                        order=self.order,
                        trend=self.trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    self.models[sid] = model.fit()
                    self.train_lengths[sid] = len(sales)
            except Exception as e:
                logger.warning(f"ARIMA failed for series_id={sid}: {e}")
                failed += 1

        self._training_time = time.time() - start
        return {
            "training_time": self._training_time,
            "n_samples": len(train_df),
            "series_fitted": len(self.models),
            "series_failed": failed,
        }

    def get_info_criteria(self) -> dict:
        """Trả về AIC/BIC trung bình qua tất cả chuỗi đã fit."""
        if not self.models:
            return {"aic_mean": float("inf"), "bic_mean": float("inf"), "n_fitted": 0}
        aic_vals = [m.aic for m in self.models.values()]
        bic_vals = [m.bic for m in self.models.values()]
        return {"aic_mean": np.mean(aic_vals), "bic_mean": np.mean(bic_vals), "n_fitted": len(aic_vals)}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # reset index: sau split, index gốc không liên tục → gán positional sẽ out of bounds
        df = df.reset_index(drop=True)
        predictions = np.zeros(len(df))
        for sid, group in df.groupby("series_id"):
            idx = group.index
            n_steps = len(group)
            if sid in self.models:
                try:
                    train_len = self.train_lengths.get(sid, 0)
                    # predict(start, end) dùng index tuyệt đối: start = bước đầu tiên sau train
                    forecast = self.models[sid].predict(
                        start=train_len, end=train_len + n_steps - 1
                    )
                    predictions[idx] = np.clip(forecast, 0, None)
                except Exception:
                    predictions[idx] = 0
            else:
                predictions[idx] = 0
        # nếu use_log_sales=True: model train trên log1p(sales) → inverse để trả về sales gốc
        if self.config.get("use_log_sales", False):
            predictions = np.expm1(predictions)
        return np.clip(predictions, 0, None)
