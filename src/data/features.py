"""Feature engineering cho Favorita Grocery Sales Forecasting.

Entity = CẶP (store_nbr, item_nbr) gộp thành series_id. Mọi feature theo thời gian
(lag, rolling) group theo series_id để không rò rỉ giữa các chuỗi. Ngoài feature thời
gian/seasonality, Favorita có EXOG THỰC: onpromotion, oil (dcoilwtico), holidays, và cờ
perishable — thêm vào khi exog.* / features.use_* bật.
"""

import numpy as np
import pandas as pd

GROUP = "series_id"
TARGET = "unit_sales"

# Fourier defaults (theo top Kaggle kernels): weekly P=7 K=3, yearly P=365.25 K=2.
# Dùng làm exogenous nền cho SARIMAX (bổ trợ exog thực).
_FOURIER_SPECS = [("w", 7.0, 3), ("y", 365.25, 2)]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trích feature thời gian từ cột date + cyclical encoding."""
    df = df.copy()
    d = df["date"].dt
    df["year"] = d.year
    df["month"] = d.month
    df["weekofyear"] = d.isocalendar().week.astype(int)
    df["dayofmonth"] = d.day
    df["dayofweek"] = d.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # cyclical: tháng/thứ là tuần hoàn -> sin/cos để model thấy "tháng 12 gần tháng 1"
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_fourier_features(df: pd.DataFrame, specs: list | None = None) -> pd.DataFrame:
    """Fourier terms cho seasonality liên tục (weekly + yearly)."""
    df = df.copy()
    specs = specs or _FOURIER_SPECS
    t = (df["date"] - df["date"].min()).dt.days.to_numpy()
    for name, period, K in specs:
        for k in range(1, K + 1):
            df[f"fourier_{name}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"fourier_{name}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


def fourier_cols(specs: list | None = None) -> list[str]:
    """Danh sách tên cột Fourier (để SARIMAX lấy làm exog nền)."""
    specs = specs or _FOURIER_SPECS
    cols = []
    for name, _period, K in specs:
        for k in range(1, K + 1):
            cols += [f"fourier_{name}_sin_{k}", f"fourier_{name}_cos_{k}"]
    return cols


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Lag unit_sales theo từng chuỗi (store_nbr, item_nbr)."""
    if lags is None:
        # 1,7 ngắn hạn; 14,30 trung hạn; 60,90 dài hạn (khớp horizon 90 ngày)
        lags = [1, 7, 14, 30, 60, 90]
    df = df.copy()
    for lag in lags:
        df[f"{TARGET}_lag_{lag}"] = df.groupby(GROUP)[TARGET].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, windows: list[int] | None = None, stats: list[str] | None = None
) -> pd.DataFrame:
    """Rolling mean/std/median của unit_sales theo từng chuỗi."""
    if windows is None:
        windows = [7, 14, 30]
    if stats is None:
        stats = ["mean", "std", "median"]
    df = df.copy()
    for w in windows:
        grouped = df.groupby(GROUP)[TARGET]
        # shift(1) trước rolling -> tránh leakage (không dùng giá trị hôm nay)
        if "mean" in stats:
            df[f"{TARGET}_rolling_mean_{w}"]   = grouped.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        if "std" in stats:
            df[f"{TARGET}_rolling_std_{w}"]    = grouped.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        if "median" in stats:
            df[f"{TARGET}_rolling_median_{w}"] = grouped.transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
    return df


def add_group_mean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group-mean (target) encodings.

    - series_dow_avg: unit_sales trung bình của (store_nbr,item_nbr) theo thứ trong tuần.
    - store_avg / item_avg: mức bán trung bình theo store_nbr / item_nbr.

    ⚠ Tính trên df truyền vào -> để tránh leakage NÊN gọi với phần train rồi merge sang
    val/test (pipeline preprocess đảm nhận).
    """
    df = df.copy()

    series_dow = (
        df.groupby([GROUP, "dayofweek"])[TARGET].mean().rename("series_dow_avg").reset_index()
    )
    df = df.merge(series_dow, on=[GROUP, "dayofweek"], how="left")

    store_avg = df.groupby("store_nbr")[TARGET].mean().rename("store_avg").reset_index()
    df = df.merge(store_avg, on="store_nbr", how="left")

    item_avg = df.groupby("item_nbr")[TARGET].mean().rename("item_avg").reset_index()
    df = df.merge(item_avg, on="item_nbr", how="left")

    for c in ["series_dow_avg", "store_avg", "item_avg"]:
        df[c] = df[c].fillna(0)
    return df


# ---------------------------------------------------------------------------
# Exogenous Favorita (chỉ áp dụng khi cột nguồn có mặt sau cleaner.build_dataset)
# ---------------------------------------------------------------------------
def add_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """onpromotion (giữ nguyên) + promo_rolling_rate: tỉ lệ KM 14 ngày gần nhất/chuỗi."""
    if "onpromotion" not in df.columns:
        return df
    df = df.copy()
    df["promo_rolling_rate"] = (
        df.groupby(GROUP)["onpromotion"].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    ).fillna(0)
    return df


def add_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    """dcoilwtico (giữ nguyên) + oil_lag_7: giá dầu trễ 1 tuần (ảnh hưởng có độ trễ)."""
    if "dcoilwtico" not in df.columns:
        return df
    df = df.copy()
    df["oil_lag_7"] = df.groupby(GROUP)["dcoilwtico"].shift(7)
    df["oil_lag_7"] = df["oil_lag_7"].fillna(df["dcoilwtico"])
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo cờ holiday tồn tại (đã tạo ở cleaner.merge_holidays); fill 0 nếu thiếu."""
    df = df.copy()
    for c in ["is_holiday", "holiday_national", "holiday_regional", "holiday_local"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(int)
    return df


def add_perishable_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Cờ perishable (từ items.csv) -> feature + trọng số NWRMSLE. Fill 0 nếu thiếu."""
    df = df.copy()
    if "perishable" not in df.columns:
        df["perishable"] = 0
    df["perishable"] = df["perishable"].fillna(0).astype(int)
    return df


def add_all_features(df: pd.DataFrame, feature_cfg: dict | None = None) -> pd.DataFrame:
    """Áp dụng các bước feature engineering, điều khiển bởi feature_cfg (configs/features.yaml).

    Nếu None → bật tất cả với default (nhóm exog chỉ thêm khi cột nguồn có mặt).
    """
    cfg = feature_cfg or {}

    if cfg.get("use_time", True):
        df = add_time_features(df)
    if cfg.get("use_fourier", True):
        df = add_fourier_features(df)
    if cfg.get("use_lag", True):
        df = add_lag_features(df, lags=cfg.get("lag_windows") or None)
    if cfg.get("use_rolling", True):
        df = add_rolling_features(
            df, windows=cfg.get("rolling_windows") or None, stats=cfg.get("rolling_stats") or ["mean", "std", "median"]
        )
    if cfg.get("use_group_mean", True):
        df = add_group_mean_features(df)

    # --- Exog Favorita ---
    if cfg.get("use_promo", True):
        df = add_promotion_features(df)
    if cfg.get("use_oil", True):
        df = add_oil_features(df)
    if cfg.get("use_holiday", True):
        df = add_holiday_features(df)
    if cfg.get("use_perishable", True):
        df = add_perishable_flag(df)

    # lag/rolling tạo NaN ở đầu mỗi chuỗi (chưa đủ quá khứ) -> điền 0 để model không lỗi
    df = df.fillna(0)
    return df


# Cột dẫn xuất từ unit_sales cần log-transform cùng (động theo windows mặc định).
def _sales_derived_cols() -> list[str]:
    cols = [TARGET]
    cols += [f"{TARGET}_lag_{l}" for l in (1, 7, 14, 30, 60, 90)]
    for w in (7, 14, 30):
        cols += [f"{TARGET}_rolling_mean_{w}", f"{TARGET}_rolling_std_{w}", f"{TARGET}_rolling_median_{w}"]
    # group-mean cũng ở cùng scale unit_sales
    cols += ["series_dow_avg", "store_avg", "item_avg"]
    return cols


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """log1p lên unit_sales và toàn bộ feature dẫn xuất từ unit_sales.

    Khớp công thức NWRMSLE (RMSLE): clip(.,0) rồi log1p -> MSE trên log ≈ tối ưu metric.
    KHÔNG đụng tới exog (oil/onpromotion/holiday/perishable) vì khác scale.
    Inverse: expm1(x) = exp(x) - 1.
    """
    df = df.copy()
    for col in _sales_derived_cols():
        if col in df.columns:
            df[col] = df[col].clip(lower=0).transform("log1p")
    return df
