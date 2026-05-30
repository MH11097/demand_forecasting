"""Model registry for time series forecasting models."""

from src.models.arima import ARIMAModel
from src.models.lstm import LSTMModel
from src.models.prophet_model import ProphetModel
from src.models.sarimax import SARIMAXModel
from src.models.xgboost_model import XGBoostModel

MODEL_REGISTRY: dict[str, type] = {
    "arima": ARIMAModel,
    "sarimax": SARIMAXModel,
    "prophet": ProphetModel,
    "xgboost": XGBoostModel,
    "lstm": LSTMModel,
}


def get_model_class(name: str) -> type:
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
