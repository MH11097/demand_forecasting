"""Data loading utilities cho Favorita Grocery Sales Forecasting.

Multi-file dataset (train + stores + items + oil + holidays + transactions). Việc
load + join + clean nằm ở cleaner.build_dataset(); module này lo series_id (factorize),
densify (tuỳ chọn) và chọn mẫu chuỗi cho per-series models. Entity = (store_nbr, item_nbr).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import cleaner

TARGET = "unit_sales"


def _add_series_id(df: pd.DataFrame) -> pd.DataFrame:
    """series_id = mã liên tục 0..N-1 cho mỗi cặp (store_nbr, item_nbr).

    Dùng groupby.ngroup (≡ factorize trên cặp) thay cho công thức số học store*K+item:
    ở cardinality Favorita (54 store × ~4100 item) công thức số học dễ đụng độ/ tràn.
    """
    df["series_id"] = df.groupby(["store_nbr", "item_nbr"], sort=True).ngroup()
    return df


def _densify(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """[Legacy] Densify ở tầng loader (sau merge exog) — giữ cho tương thích/tiện ích.

    Zero-fill chuẩn nay nằm ở cleaner.reindex_fill_dates (chạy TRƯỚC merge exog trong
    build_dataset → exog theo ngày gán đúng cho ngày chèn). Hàm này chỉ kích hoạt khi
    config.densify=true và bound theo NGÀY BÁN ĐẦU→CUỐI của TỪNG chuỗi (không toàn cục).
    """
    if not config.get("densify", False):
        return df

    static_cols = [
        c for c in ["store_nbr", "item_nbr", "city", "state", "type", "cluster",
                    "family", "class", "perishable"] if c in df.columns
    ]
    date_cols = [c for c in ["dcoilwtico", "is_holiday", "holiday_national",
                             "holiday_regional", "holiday_local"] if c in df.columns]

    out = []
    for sid, g in df.groupby("series_id", sort=False):
        # bound per-series: chỉ điền trong khoảng sống của chuỗi (tránh zero giả)
        full_dates = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full_dates)
        g.index.name = "date"
        g["series_id"] = sid
        for c in static_cols:                # thuộc tính tĩnh của chuỗi -> ffill/bfill
            g[c] = g[c].ffill().bfill()
        if TARGET in g.columns:
            g[TARGET] = g[TARGET].fillna(0.0)
        if "onpromotion" in g.columns:
            g["onpromotion"] = g["onpromotion"].fillna(0).astype(int)
        for c in date_cols:                  # exog theo ngày -> ffill/bfill theo thời gian
            g[c] = g[c].ffill().bfill()
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def load_raw_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load + join + clean + zero-fill (cleaner.build_dataset) → series_id → sort.

    Zero-fill ngày thiếu đã thực hiện trong build_dataset (clean.zero_fill, mặc định bật).
    Returns:
        (df, df) — trả 2 lần cùng DataFrame để giữ chữ ký tương thích pipeline.
    """
    df = cleaner.build_dataset(config)
    df = _add_series_id(df)
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    return df, df


def load_cleaned_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset đã clean+zero-fill sẵn (cleaner.build_dataset → save).

    Ưu tiên feather (12.5M dòng sau zero-fill → CSV quá nặng); fallback CSV. Feature engineering
    áp dụng SAU khi load (rẻ, phụ thuộc config) — file cache chỉ giữ phần join+zero-fill (nặng).
    """
    cleaned_dir = Path(config["data"]["cleaned_dir"])
    feather_path = cleaned_dir / "train_cleaned.feather"
    csv_path = cleaned_dir / "train_cleaned.csv"
    if feather_path.exists():
        df = pd.read_feather(feather_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        # chưa materialize cache → fallback dựng từ raw (an toàn trên checkout mới)
        print(f"[loader] Cache {feather_path} chưa có → build từ raw (chạy "
              f"`python scripts/clean_data.py` để cache cho lần sau).")
        return load_raw_data(config)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "series_id" not in df.columns:
        df = _add_series_id(df)

    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    return df, df


def select_series(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Chọn mẫu đại diện các chuỗi (store_nbr, item_nbr) cho per-series models.

    Số chuỗi trong 1 nhóm store vẫn lớn để fit ARIMA/SARIMAX/Prophet từng cái ->
    chọn n_series chuỗi trải đều theo tổng unit_sales (volume cao → thấp).

    Config:
        series_sample.n_series: số chuỗi cần chọn (null/0 = dùng tất cả)
        series_sample.strategy: "stratified_volume" | "random"
        seed: cố định để tái lập
    """
    sample_cfg = config.get("series_sample") or {}
    n_series = sample_cfg.get("n_series")
    if not n_series:
        return df  # null/0 -> giữ toàn bộ

    seed = config.get("seed", 42)
    strategy = sample_cfg.get("strategy", "stratified_volume")

    # tổng unit_sales mỗi chuỗi -> đại lượng xếp hạng volume
    totals = df.groupby("series_id")[TARGET].sum().sort_values(ascending=False)
    all_ids = totals.index.to_numpy()
    n_series = min(n_series, len(all_ids))

    if strategy == "random":
        rng = np.random.default_rng(seed)
        chosen = rng.choice(all_ids, size=n_series, replace=False)
    else:
        # stratified_volume: lấy đều dọc bảng xếp hạng volume (cao→thấp)
        idx = np.linspace(0, len(all_ids) - 1, n_series).round().astype(int)
        chosen = all_ids[np.unique(idx)]

    return df[df["series_id"].isin(chosen)].reset_index(drop=True)


def filter_stores(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Backward-compat: lọc nhóm store (delegate cleaner.apply_store_filter)."""
    return cleaner.apply_store_filter(df, config)
