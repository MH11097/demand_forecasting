"""SARIMAX model — per-series với seasonal component m=7 và Fourier exog.

Không có dữ liệu ngoại sinh thực (promotions, holidays) nên dùng
Fourier terms (weekly K=3, yearly K=2) làm exog tổng hợp — chuẩn
theo top Kaggle kernels cho dataset này.
"""

import logging
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsSARIMAX

from src.data.features import fourier_cols
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class SARIMAXModel(BaseModel):
    name = "sarimax"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.order = tuple(model_cfg.get("order", [1, 1, 1]))
        # seasonal_order (P,D,Q,m): m=7 chu kỳ tuần — quan trọng nhất cho dataset này
        self.seasonal_order = tuple(model_cfg.get("seasonal_order", [1, 1, 0, 7]))
        self.trend = model_cfg.get("trend", "n")
        # maxiter: Kalman filter state-space lớn cần nhiều vòng lặp để hội tụ
        self.maxiter = model_cfg.get("maxiter", 200)
        self.models: dict[int, object] = {}
        self.train_lengths: dict[int, int] = {}
        # tên cột Fourier exog (dùng features.fourier_cols() để đồng bộ với data pipeline)
        self._exog_cols = fourier_cols()

    def _get_exog(self, df: pd.DataFrame) -> np.ndarray | None:
        """Trích Fourier exog từ df. Trả về None nếu không có cột nào."""
        available = [c for c in self._exog_cols if c in df.columns]
        if not available:
            return None
        return df[available].values.astype(float)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        start = time.time()
        series_ids = sorted(train_df["series_id"].unique())
        failed = 0
        for sid in series_ids:
            series_data = train_df[train_df["series_id"] == sid].sort_values("date")
            sales = series_data["unit_sales"].values.astype(float)
            exog  = self._get_exog(series_data)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # enforce_stationarity/invertibility=False → nới lỏng ràng buộc, tránh lỗi
                    model = StatsSARIMAX(
                        sales,
                        exog=exog,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        trend=self.trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    self.models[sid] = model.fit(disp=False, maxiter=self.maxiter)
                    self.train_lengths[sid] = len(sales)
            except Exception as e:
                logger.warning(f"SARIMAX failed for series_id={sid}: {e}")
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
        return {"aic_mean": float(np.mean(aic_vals)), "bic_mean": float(np.mean(bic_vals)), "n_fitted": len(aic_vals)}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.reset_index(drop=True)
        predictions = np.zeros(len(df))
        for sid, group in df.groupby("series_id"):
            idx     = group.index
            n_steps = len(group)
            if sid in self.models:
                try:
                    exog = self._get_exog(group)
                    # get_forecast() tương thích hơn khi cần truyền exog
                    forecast_result = self.models[sid].get_forecast(steps=n_steps, exog=exog)
                    predictions[idx] = np.clip(forecast_result.predicted_mean, 0, None)
                except Exception:
                    predictions[idx] = 0
            else:
                predictions[idx] = 0
        # nếu use_log_sales=True: model train trên log1p(sales) → inverse để trả về sales gốc
        if self.config.get("use_log_sales", False):
            predictions = np.expm1(predictions)
        return np.clip(predictions, 0, None)
