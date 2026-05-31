"""XGBoost model — global model với store_nbr/item_nbr + exog làm feature.

Global model: train trên tất cả chuỗi trong nhóm store đã lọc, dùng store_nbr/item_nbr
+ exog Favorita (onpromotion, oil, holiday, perishable) làm feature. Feature set lấy
động từ df (tất cả cột numeric ngoài target/ID) + đảm bảo store_nbr/item_nbr được include.
"""

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.forecast_exog import global_model_feature_cols
from src.models.base import BaseModel


def _xgb_device() -> str:
    """Ưu tiên GPU: 'cuda' nếu có, ngược lại 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.n_estimators         = model_cfg.get("n_estimators", 1000)
        self.max_depth            = model_cfg.get("max_depth", 7)
        self.learning_rate        = model_cfg.get("learning_rate", 0.05)
        self.subsample            = model_cfg.get("subsample", 0.8)
        self.colsample_bytree     = model_cfg.get("colsample_bytree", 0.8)
        self.reg_alpha            = model_cfg.get("reg_alpha", 0)
        self.reg_lambda           = model_cfg.get("reg_lambda", 1)
        self.early_stopping_rounds = model_cfg.get("early_stopping_rounds", 50)
        # ưu tiên GPU; có thể ép qua config model.device
        self.device = model_cfg.get("device") or _xgb_device()
        self.model: xgb.XGBRegressor | None = None
        self.feature_cols: list[str] = []

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use entity IDs, shared forecast exog, and leakage-safe sales history."""
        self.feature_cols = global_model_feature_cols(
            df, self.config.get("features", {}), include_entity_ids=True
        )
        return df[self.feature_cols].fillna(0)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        start = time.time()

        X_train = self._get_features(train_df)
        y_train = train_df["unit_sales"].values
        use_es = val_df is not None and len(val_df) > 0

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            # chỉ bật early stopping khi thực sự có eval_set, tránh lỗi XGBoost
            early_stopping_rounds=self.early_stopping_rounds if use_es else None,
            random_state=self.config.get("seed", 42),
            # GPU: tree_method="hist" + device="cuda" (XGBoost ≥2.0). Fallback "cpu" tự động.
            tree_method="hist",
            device=self.device,
            n_jobs=-1,
        )

        fit_params = {}
        if use_es:
            X_val = val_df[self.feature_cols].fillna(0)
            y_val = val_df["unit_sales"].values
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"]  = False

        self.model.fit(X_train, y_train, **fit_params)
        self._training_time = time.time() - start
        return {
            "training_time": self._training_time,
            "n_samples": len(train_df),
            "n_features": len(self.feature_cols),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].fillna(0)
        predictions = self.model.predict(X)
        # nếu use_log_sales=True: model train trên log1p(unit_sales) → inverse về scale gốc
        if self.config.get("use_log_sales", False):
            predictions = np.expm1(predictions)
        return np.clip(predictions, 0, None)
