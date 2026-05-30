"""Data preprocessing pipeline cho Favorita Grocery Sales Forecasting.

Việc chính: tạo index liên tục cho embedding (store_idx/item_idx, vocab suy ra từ data),
split theo thời gian (KHÔNG shuffle), và scale feature liên tục bằng RobustScaler
(unit_sales lệch phải mạnh + nhiều outlier -> RobustScaler ổn hơn StandardScaler).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Cột KHÔNG scale: target, ID gốc, index embedding, date, và các cờ binary/categorical.
_NO_SCALE = {
    "unit_sales", "store_nbr", "item_nbr", "series_id", "store_idx", "item_idx",
    "id", "cluster", "class", "date", "is_weekend", "onpromotion", "perishable",
    "is_holiday", "holiday_national", "holiday_regional", "holiday_local",
}


def preprocess(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Pipeline: tạo entity index, split theo thời gian, scale feature.

    Args:
        df: DataFrame đã qua feature engineering (có series_id, store_nbr, item_nbr,
            unit_sales, features).
        config: dict với split dates, model.skip_scaling.

    Returns:
        (train_df, val_df, test_df, scaler)
    """
    df = df.copy()
    df = _build_entity_index(df)

    # time series -> split theo thời gian, KHÔNG shuffle (tránh leakage từ tương lai)
    split_cfg = config["split"]
    train_df = df[df["date"] <= split_cfg["train_end"]].copy()

    if "val_start" in split_cfg and "val_end" in split_cfg:
        val_df = df[(df["date"] >= split_cfg["val_start"]) & (df["date"] <= split_cfg["val_end"])].copy()
    else:
        val_df = pd.DataFrame(columns=df.columns)

    test_mask = df["date"] >= split_cfg["test_start"]
    if "test_end" in split_cfg:
        test_mask &= df["date"] <= split_cfg["test_end"]
    test_df = df[test_mask].copy()

    # ARIMA/SARIMAX/Prophet tự xử lý trend+seasonality -> skip scaling để giữ giá trị gốc.
    # XGBoost/LSTM cần chuẩn hoá feature liên tục.
    skip_scaling = config.get("model", {}).get("skip_scaling", False)
    numeric_cols = _get_numeric_feature_cols(train_df)
    scaler = RobustScaler()
    if numeric_cols and not skip_scaling:
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        if len(val_df) > 0:
            val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
        if len(test_df) > 0:
            test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return train_df, val_df, test_df, scaler


def entity_vocab_sizes(df: pd.DataFrame) -> tuple[int, int]:
    """Số store/item duy nhất -> kích thước vocab cho nn.Embedding của LSTM (suy từ data)."""
    return int(df["store_nbr"].nunique()), int(df["item_nbr"].nunique())


def _build_entity_index(df: pd.DataFrame) -> pd.DataFrame:
    """Map store_nbr/item_nbr sang chỉ số liên tục 0..N-1 cho nn.Embedding của LSTM.

    Vocab = số store/item DUY NHẤT trong subset (suy từ data, KHÔNG hardcode 10/50).
    Chỉ là ánh xạ ID nên KHÔNG gây leakage (tập entity cố định, biết trước).
    """
    store_map = {s: i for i, s in enumerate(sorted(df["store_nbr"].unique()))}
    item_map = {it: i for i, it in enumerate(sorted(df["item_nbr"].unique()))}
    df["store_idx"] = df["store_nbr"].map(store_map).astype(int)
    df["item_idx"] = df["item_nbr"].map(item_map).astype(int)
    return df


def _get_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Cột numeric để scale — loại target, ID, index, date và cờ binary/categorical.

    Còn lại (lag/rolling/Fourier/group-mean/oil/promo_rate...) là feature liên tục -> scale.
    """
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in _NO_SCALE]
