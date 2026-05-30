"""Interactive mode helpers cho train script.

Khi user chạy `python scripts/train.py` không có --model:
→ hiện menu chọn model → hỏi tham số → trả (model_name, overrides) để train.

Tách ra file riêng vì MODEL_PARAMS registry khá dài, giữ train.py gọn.
"""

import typer

from src.models import MODEL_REGISTRY

# Mô tả tham số chính của từng model, lấy default từ YAML configs.
# Dùng trong interactive mode để user biết có thể chỉnh gì + giá trị mặc định.
MODEL_PARAMS: dict[str, list[dict]] = {
    "arima": [
        {"key": "model.order", "name": "order (p,d,q)", "default": [1, 1, 1], "type": "list"},
        {"key": "model.max_stores", "name": "max_stores", "default": 50, "type": "int_or_null"},
    ],
    "sarimax": [
        {"key": "model.order", "name": "order (p,d,q)", "default": [1, 1, 1], "type": "list"},
        {"key": "model.seasonal_order", "name": "seasonal_order (P,D,Q,s)", "default": [1, 1, 1, 7], "type": "list"},
        {"key": "model.exog_columns", "name": "exog_columns", "default": ["Promo", "SchoolHoliday"], "type": "list_str"},
        {"key": "model.max_stores", "name": "max_stores", "default": 50, "type": "int_or_null"},
    ],
    "prophet": [
        {"key": "model.changepoint_prior_scale", "name": "changepoint_prior_scale", "default": 0.05, "type": "float"},
        {"key": "model.seasonality_mode", "name": "seasonality_mode", "default": "multiplicative", "type": "str"},
        {"key": "model.regressors", "name": "regressors", "default": ["Promo"], "type": "list_str"},
        {"key": "model.max_stores", "name": "max_stores", "default": 50, "type": "int_or_null"},
    ],
    "xgboost": [
        {"key": "model.n_estimators", "name": "n_estimators", "default": 1000, "type": "int"},
        {"key": "model.max_depth", "name": "max_depth", "default": 7, "type": "int"},
        {"key": "model.learning_rate", "name": "learning_rate", "default": 0.1, "type": "float"},
        {"key": "model.subsample", "name": "subsample", "default": 0.8, "type": "float"},
        {"key": "model.colsample_bytree", "name": "colsample_bytree", "default": 0.8, "type": "float"},
        {"key": "model.early_stopping_rounds", "name": "early_stopping_rounds", "default": 50, "type": "int"},
    ],
    "lstm": [
        {"key": "model.hidden_size", "name": "hidden_size", "default": 128, "type": "int"},
        {"key": "model.num_layers", "name": "num_layers", "default": 2, "type": "int"},
        {"key": "model.seq_len", "name": "seq_len", "default": 30, "type": "int"},
        {"key": "model.batch_size", "name": "batch_size", "default": 256, "type": "int"},
        {"key": "model.epochs", "name": "epochs", "default": 50, "type": "int"},
        {"key": "model.learning_rate", "name": "learning_rate", "default": 0.001, "type": "float"},
        {"key": "model.dropout", "name": "dropout", "default": 0.2, "type": "float"},
    ],
}

# Mô tả ngắn gọn từng model → hiển thị trong menu interactive cho DS dễ chọn
MODEL_DESCRIPTIONS: dict[str, str] = {
    "arima": "Statistical, fit từng store",
    "sarimax": "Statistical + seasonal + exog vars",
    "prophet": "Statistical + holiday + regressors",
    "xgboost": "ML, global model",
    "lstm": "Deep Learning, sequence model",
}


def _parse_input(raw: str, param_type: str, default):
    """Parse user input thành giá trị đúng kiểu. Trả default nếu input rỗng."""
    raw = raw.strip()
    if not raw:
        return default
    if param_type == "int":
        return int(raw)
    if param_type == "float":
        return float(raw)
    if param_type == "str":
        return raw
    if param_type == "int_or_null":
        # cho phép nhập "null" hoặc "None" → bỏ giới hạn max_stores
        return None if raw.lower() in ("null", "none") else int(raw)
    if param_type == "list":
        # "2,1,1" → [2, 1, 1]  hoặc "[2,1,1]" → [2, 1, 1]
        raw = raw.strip("[]")
        parsed = [int(x.strip()) for x in raw.split(",")]
        # default cho biết số phần tử mong đợi, vd order cần đúng 3 phần tử (p,d,q)
        if isinstance(default, list) and len(parsed) != len(default):
            raise ValueError(f"Cần {len(default)} giá trị, nhận được {len(parsed)}")
        return parsed
    if param_type == "list_str":
        # "Promo,SchoolHoliday" → ["Promo", "SchoolHoliday"]
        raw = raw.strip("[]")
        return [x.strip().strip("'\"") for x in raw.split(",")]
    return raw


def _ask_params(model_name: str) -> dict:
    """Hiện danh sách tham số model, cho user nhập giá trị. Trả dict overrides (dotted keys)."""
    params = MODEL_PARAMS.get(model_name, [])
    if not params:
        return {}
    typer.echo(f"\nTham số {model_name.upper()} (Enter = giữ mặc định):")
    overrides = {}
    for p in params:
        while True:
            raw = input(f"  {p['name']} [{p['default']}]: ")
            try:
                value = _parse_input(raw, p["type"], p["default"])
                break
            except (ValueError, TypeError) as e:
                typer.echo(f"  Lỗi: {e}. Nhập lại.")
        # chỉ thêm override khi user thay đổi giá trị → tránh ghi đè không cần thiết
        if value != p["default"]:
            overrides[p["key"]] = value
    return overrides


def interactive_select() -> tuple[str, dict]:
    """Hiện menu chọn model + hỏi tham số. Trả (model_name, overrides dict)."""
    models = list(MODEL_REGISTRY.keys())
    typer.echo("\nChọn model:")
    for i, name in enumerate(models, 1):
        desc = MODEL_DESCRIPTIONS.get(name, "")
        typer.echo(f"  {i}. {name:<12} {desc}")

    # nhận input cho đến khi hợp lệ
    while True:
        raw = input(f"\nNhập số (1-{len(models)}): ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(models):
                break
        except ValueError:
            pass
        typer.echo(f"Vui lòng nhập số từ 1 đến {len(models)}.")

    model_name = models[idx - 1]
    overrides = _ask_params(model_name)
    return model_name, overrides
