"""Abstract base class for all forecasting models."""

import io
import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_all


class BaseModel(ABC):
    """Base class that all models must implement."""

    name: str = "base"

    def __init__(self, config: dict):
        self.config = config
        self._training_time: float = 0.0

    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        """Train the model. Returns training info dict."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the given DataFrame."""

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate model on a DataFrame with unit_sales column.

        predict() trả về unit_sales gốc (đã inverse log nếu use_log_sales=True).
        y_true cũng cần inverse để cùng scale với predictions; trọng số NWRMSLE
        lấy từ cờ perishable nếu có.
        """
        import numpy as np
        from src.evaluation.metrics import weights_from_frame
        predictions = self.predict(df)
        y_true = df["unit_sales"].values.astype(float)
        if self.config.get("use_log_sales", False):
            y_true = np.expm1(y_true)
        return evaluate_all(y_true, predictions, weights_from_frame(df))

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from disk. Tự map CUDA → CPU nếu GPU không khả dụng."""
        try:
            import torch
            if not torch.cuda.is_available():
                # Model train trên CUDA nhưng load trên CPU:
                # pickle gọi torch.storage._load_from_bytes bên trong → patch tạm để thêm map_location
                _orig = torch.storage._load_from_bytes
                torch.storage._load_from_bytes = lambda b: torch.load(
                    io.BytesIO(b), map_location="cpu", weights_only=False
                )
                try:
                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                finally:
                    torch.storage._load_from_bytes = _orig
            else:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
            # Chuyển device attribute + network weights sang device hiện tại
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if hasattr(obj, "device"):
                obj.device = device
            if hasattr(obj, "net") and obj.net is not None:
                obj.net.to(device)
            return obj
        except ImportError:
            pass
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_result_template(
        self,
        metrics: dict | None = None,
        param_slug: str = "",
        experiment_name: str = "",
    ) -> dict:
        """Generate standardized result JSON."""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        run_id = f"{param_slug}__{timestamp}" if param_slug else timestamp
        return {
            "model_name": self.name,
            "run_id": run_id,
            "param_slug": param_slug,
            "experiment_name": experiment_name,
            "timestamp": now.isoformat(),
            "config": self.config,
            "metrics": metrics or {},
            "cv_metrics": {},
            "training_time_seconds": self._training_time,
            "predictions_path": "",
            "metadata": {},
        }

    def save_results(self, results: dict, results_dir: str):
        """Save result JSON to results directory."""
        out_dir = Path(results_dir) / self.name / results["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "result.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return str(out_dir)
