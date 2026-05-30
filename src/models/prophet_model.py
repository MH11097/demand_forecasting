"""Prophet model — per-series fitting với US holidays và multiplicative seasonality.

Không có dữ liệu ngoại sinh (promotions, holidays flags) trong dataset →
chỉ dùng US holiday calendar tích hợp sẵn của Prophet. Không thêm regressor.
"""

import logging
import time
import warnings

import numpy as np
import pandas as pd

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class ProphetModel(BaseModel):
    name = "prophet"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        # Trend: changepoint_prior_scale nhỏ → trend mượt, lớn → linh hoạt hơn
        self.changepoint_prior_scale  = model_cfg.get("changepoint_prior_scale", 0.05)
        self.n_changepoints           = model_cfg.get("n_changepoints", 25)
        self.changepoint_range        = model_cfg.get("changepoint_range", 0.8)
        # Dataset có seasonality nhân tính (sales ~ level × seasonal_factor) → multiplicative
        self.seasonality_mode         = model_cfg.get("seasonality_mode", "multiplicative")
        self.seasonality_prior_scale  = model_cfg.get("seasonality_prior_scale", 10.0)
        self.holidays_prior_scale     = model_cfg.get("holidays_prior_scale", 10.0)
        self.models: dict = {}

    def _prepare_prophet_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Đổi tên cột sang format Prophet: ds=date, y=unit_sales."""
        pdf = df[["date", "unit_sales"]].copy()
        pdf = pdf.rename(columns={"date": "ds", "unit_sales": "y"})
        return pdf

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        from prophet import Prophet

        start = time.time()
        series_ids = sorted(train_df["series_id"].unique())
        failed = 0
        for sid in series_ids:
            series_data = train_df[train_df["series_id"] == sid].sort_values("date")
            pdf = self._prepare_prophet_df(series_data)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = Prophet(
                        changepoint_prior_scale=self.changepoint_prior_scale,
                        n_changepoints=self.n_changepoints,
                        changepoint_range=self.changepoint_range,
                        seasonality_mode=self.seasonality_mode,
                        seasonality_prior_scale=self.seasonality_prior_scale,
                        holidays_prior_scale=self.holidays_prior_scale,
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                    )
                    # Dataset Mỹ → dùng US holiday calendar (không có promotions/flags)
                    m.add_country_holidays(country_name="US")
                    m.fit(pdf)
                    self.models[sid] = m
            except Exception as e:
                logger.warning(f"Prophet failed for series_id={sid}: {e}")
                failed += 1

        self._training_time = time.time() - start
        return {
            "training_time": self._training_time,
            "n_samples": len(train_df),
            "series_fitted": len(self.models),
            "series_failed": failed,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.reset_index(drop=True)
        predictions = np.zeros(len(df))
        for sid, group in df.groupby("series_id"):
            idx = group.index
            if sid in self.models:
                try:
                    future = self._prepare_prophet_df(group)
                    forecast = self.models[sid].predict(future)
                    predictions[idx] = np.clip(forecast["yhat"].values, 0, None)
                except Exception:
                    predictions[idx] = 0
            else:
                predictions[idx] = 0
        # nếu use_log_sales=True: model train trên log1p(sales) → inverse để trả về sales gốc
        if self.config.get("use_log_sales", False):
            predictions = np.expm1(predictions)
        return np.clip(predictions, 0, None)
