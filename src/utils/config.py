"""Configuration loading and merging utilities."""

import copy
from pathlib import Path

import yaml

# Key hyperparams per model for slug generation
_SLUG_KEYS: dict[str, list[str]] = {
    "arima": ["order", "trend"],
    "sarimax": ["order", "seasonal_order", "trend"],
    "prophet": ["changepoint_prior_scale", "seasonality_mode", "seasonality_prior_scale", "holidays_prior_scale", "n_changepoints", "changepoint_range"],
    "xgboost": ["max_depth", "n_estimators", "learning_rate"],
    "lstm": ["hidden_size", "num_layers"],
}

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

_EXOG_FEATURE_FLAGS = {
    "use_oil": "use_oil",
    "use_holidays": "use_holiday",
    "use_onpromotion": "use_promo",
    "use_perishable": "use_perishable",
}


def _apply_exog_feature_flags(config: dict) -> dict:
    """Make exog.* authoritative when a source is disabled.

    The cleaner may retain columns needed for evaluation (notably ``perishable``),
    so model-facing feature flags must also be disabled explicitly.
    """
    exog = config.get("exog", {})
    features = config.setdefault("features", {})
    for exog_key, feature_key in _EXOG_FEATURE_FLAGS.items():
        if not exog.get(exog_key, True):
            features[feature_key] = False
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict. Override values take priority."""
    # config model-specific chỉ chứa key cần ghi đè -> merge sâu để giữ lại các key từ base
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(model_name: str | None = None, overrides: dict | None = None) -> dict:
    """Load base config, merge model-specific config, then apply CLI overrides.

    Args:
        model_name: Model name to load specific config (e.g. "arima" -> configs/arima.yaml)
        overrides: Dict of dotted-key overrides (e.g. {"store_type": "c"})
    """
    # base.yaml chứa config chung (seed, split dates, data paths) -> nền tảng cho mọi model
    base_path = CONFIGS_DIR / "base.yaml"
    with open(base_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # features.yaml chứa cấu hình bật/tắt từng nhóm feature -> merge vào config chung
    features_path = CONFIGS_DIR / "features.yaml"
    if features_path.exists():
        with open(features_path, encoding="utf-8") as f:
            features_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, features_config)

    # model-specific yaml (arima.yaml, xgboost.yaml...) ghi đè hyperparams riêng
    if model_name:
        model_path = CONFIGS_DIR / f"{model_name}.yaml"
        if model_path.exists():
            with open(model_path, encoding="utf-8") as f:
                model_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, model_config)

    # CLI overrides (--set store_type=c) ưu tiên cao nhất -> ghi đè mọi config file
    if overrides:
        for key, value in overrides.items():
            _set_nested(config, key, value)

    return _apply_exog_feature_flags(config)


def _format_slug_value(value) -> str:
    """Format a single value for use in a slug."""
    if isinstance(value, list):
        return "-".join(str(v) for v in value)
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value)


def make_param_slug(config: dict) -> str:
    """Generate a slug string from key hyperparams in the model config.

    Example: ARIMA order=[1,1,1] -> "order_1-1-1"
    Example: XGBoost max_depth=7, n_estimators=1000, lr=0.1 -> "max_depth_7__n_estimators_1000__lr_0p1"
    """
    # mỗi lần chạy dùng hyperparams khác nhau -> tạo slug ngắn gọn làm tên thư mục kết quả, dễ phân biệt
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "")
    keys = _SLUG_KEYS.get(model_name, [])

    parts = []
    for key in keys:
        if key in model_cfg:
            # Use short alias for common long param names
            short = "lr" if key == "learning_rate" else key
            parts.append(f"{short}_{_format_slug_value(model_cfg[key])}")

    return "__".join(parts) if parts else "default"


def _set_nested(d: dict, dotted_key: str, value):
    """Set a value in a nested dict using dotted key notation."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
