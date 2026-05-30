"""Feature engineering cho Favorita Grocery Sales Forecasting.

Entity = CẶP (store_nbr, item_nbr) gộp thành series_id. Mọi feature theo thời gian
(lag, rolling) group theo series_id để không rò rỉ giữa các chuỗi. Ngoài feature thời
gian/seasonality, Favorita có EXOG THỰC: onpromotion, oil (dcoilwtico), holidays, và cờ
perishable — thêm vào khi exog.* / features.use_* bật.
"""

import re

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


def add_group_mean_features(df: pd.DataFrame, train_end=None) -> pd.DataFrame:
    """Group-mean (target) encodings đa cấp độ.

    - series_dow_avg: unit_sales trung bình của (store_nbr,item_nbr) theo thứ trong tuần.
    - store_avg / item_avg: mức bán trung bình theo store_nbr / item_nbr.
    - family_avg: mức bán trung bình theo nhóm hàng (family, từ items.csv) — bắt mùa vụ
      cấp nhóm, giảm nhiễu cho item bán thưa (theo top solution Favorita).

    ⚠ CHỐNG LEAKAGE: thống kê CHỈ tính trên phần train (`date <= train_end`) rồi merge sang
    MỌI dòng (train/val/test). Nếu `train_end=None` → tính trên toàn df (chỉ dùng khi không
    có split, vd phân tích EDA). series_dow_avg đặc biệt nhạy: nếu tính cả test thì 1 ngày
    test sẽ "thấy" doanh số các ngày cùng thứ khác trong test → gần như lộ đáp án.
    """
    df = df.copy()
    src = df[df["date"] <= pd.Timestamp(train_end)] if train_end is not None else df

    series_dow = (
        src.groupby([GROUP, "dayofweek"])[TARGET].mean().rename("series_dow_avg").reset_index()
    )
    df = df.merge(series_dow, on=[GROUP, "dayofweek"], how="left")

    store_avg = src.groupby("store_nbr")[TARGET].mean().rename("store_avg").reset_index()
    df = df.merge(store_avg, on="store_nbr", how="left")

    item_avg = src.groupby("item_nbr")[TARGET].mean().rename("item_avg").reset_index()
    df = df.merge(item_avg, on="item_nbr", how="left")

    fill_cols = ["series_dow_avg", "store_avg", "item_avg"]
    if "family" in df.columns:
        family_avg = src.groupby("family")[TARGET].mean().rename("family_avg").reset_index()
        df = df.merge(family_avg, on="family", how="left")
        fill_cols.append("family_avg")

    for c in fill_cols:
        df[c] = df[c].fillna(0)
    return df


def add_payday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng ngày lương Ecuador: trả lương khu vực công giữa tháng (15) + cuối tháng.

    Hậu lương → tiêu dùng tăng ~20-30% (đặc biệt cuối tuần ngay sau lương). Theo top
    solution Favorita (btrotta). is_payday: cờ ngày lương; days_since/to_payday: khoảng
    cách (ngày) tới mốc lương gần nhất (xấp xỉ theo chu kỳ 15/cuối-tháng).
    """
    df = df.copy()
    day = df["date"].dt.day.to_numpy()
    ld = df["date"].dt.days_in_month.to_numpy()    # ngày cuối tháng (28-31)

    df["is_payday"] = ((day == 15) | (day == ld)).astype(int)

    # days_to: tới mốc lương kế tiếp (15 hoặc cuối tháng); nếu đã qua cả hai → 15 của tháng sau
    to_15, to_end = (15 - day).astype(float), (ld - day).astype(float)
    cand = np.stack([to_15, to_end], axis=1)
    cand[cand < 0] = np.inf
    days_to = cand.min(axis=1)
    spill = ~np.isfinite(days_to)
    days_to[spill] = (ld - day + 15)[spill]        # vượt cuối tháng → tới 15 tháng sau
    df["days_to_payday"] = days_to.astype(int)

    # days_since: từ mốc lương gần nhất đã qua (15 tháng này hoặc cuối tháng TRƯỚC = day)
    since_15 = (day - 15).astype(float)
    cand_s = np.stack([since_15, day.astype(float)], axis=1)
    cand_s[cand_s < 0] = np.inf
    df["days_since_payday"] = np.where(np.isfinite(cand_s.min(axis=1)),
                                       cand_s.min(axis=1), day).astype(int)
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


def add_promo_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thời điểm khuyến mãi: đếm KM gần đây + khoảng cách tới KM trước/sau (theo chuỗi).

    - promo_count_7/14/30: số ngày KM trong cửa sổ quá khứ (shift(1) → không dùng hôm nay).
    - days_since_last_promo: số ngày kể từ lần KM gần nhất (quá khứ).
    - days_until_next_promo: số ngày tới lần KM kế (TƯƠNG LAI). Hợp lệ vì onpromotion là
      exog BIẾT TRƯỚC ở Favorita (test set có sẵn lịch onpromotion) — không leakage.

    YÊU CẦU: df đã zero-fill (chuỗi liên tục theo ngày) + sort theo [GROUP, date] → khoảng
    cách theo CHỈ SỐ DÒNG ≡ khoảng cách theo NGÀY.
    """
    if "onpromotion" not in df.columns:
        return df
    df = df.sort_values([GROUP, "date"]).reset_index(drop=True)
    g = df.groupby(GROUP)["onpromotion"]
    for w in (7, 14, 30):
        df[f"promo_count_{w}"] = (
            g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).sum())
        ).fillna(0)

    idx = np.arange(len(df), dtype=float)
    is_promo = (df["onpromotion"].to_numpy() > 0)
    promo_pos = pd.Series(np.where(is_promo, idx, np.nan), index=df.index)
    last_pos = promo_pos.groupby(df[GROUP]).ffill()      # vị trí KM gần nhất ≤ hôm nay
    next_pos = promo_pos.groupby(df[GROUP]).bfill()       # vị trí KM gần nhất ≥ hôm nay
    BIG = 999
    df["days_since_last_promo"] = (idx - last_pos).fillna(BIG).clip(upper=BIG).astype(int)
    df["days_until_next_promo"] = (next_pos - idx).fillna(BIG).clip(upper=BIG).astype(int)
    return df


def add_zero_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đếm ngày bán = 0 trong 28 ngày gần nhất (proxy hết hàng/độ biến động cầu).

    Tính trên unit_sales GỐC (trước log-transform). shift(1) → chỉ dùng quá khứ.
    """
    df = df.copy()
    is_zero = (df[TARGET] <= 0).astype(int)
    df["zero_sales_last_28"] = (
        is_zero.groupby(df[GROUP]).transform(lambda x: x.shift(1).rolling(28, min_periods=1).sum())
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


def add_all_features(df: pd.DataFrame, feature_cfg: dict | None = None, train_end=None) -> pd.DataFrame:
    """Áp dụng các bước feature engineering, điều khiển bởi feature_cfg (configs/features.yaml).

    Nếu None → bật tất cả với default (nhóm exog chỉ thêm khi cột nguồn có mặt).
    `train_end`: ngày cuối train → group-mean encodings chỉ tính trên train (chống leakage).
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
        df = add_group_mean_features(df, train_end=train_end)
    if cfg.get("use_payday", True):
        df = add_payday_features(df)
    if cfg.get("use_zero_sales", True):
        df = add_zero_sales_features(df)

    # --- Exog Favorita ---
    if cfg.get("use_promo", True):
        df = add_promotion_features(df)
        df = add_promo_timing_features(df)
    if cfg.get("use_oil", True):
        df = add_oil_features(df)
    if cfg.get("use_holiday", True):
        df = add_holiday_features(df)
    if cfg.get("use_perishable", True):
        df = add_perishable_flag(df)

    # lag/rolling tạo NaN ở đầu mỗi chuỗi (chưa đủ quá khứ) -> điền 0 để model không lỗi
    df = df.fillna(0)
    return df


# Pattern cột dẫn xuất từ unit_sales (cùng scale → log-transform cùng). Suy theo TÊN CỘT
# thực tế (không hardcode windows) để bền vững khi đổi lag_windows/rolling_windows.
_SALES_DERIVED_RE = re.compile(
    rf"^{TARGET}_lag_\d+$|^{TARGET}_rolling_(mean|std|median)_\d+$"
)
_SALES_DERIVED_AVGS = ["series_dow_avg", "store_avg", "item_avg", "family_avg"]


def _sales_derived_cols(df: pd.DataFrame) -> list[str]:
    """Cột ở cùng scale unit_sales: target + lag/rolling + group-mean encodings."""
    cols = [TARGET] if TARGET in df.columns else []
    cols += [c for c in df.columns if _SALES_DERIVED_RE.match(c)]
    cols += [c for c in _SALES_DERIVED_AVGS if c in df.columns]
    return cols


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """log1p lên unit_sales và toàn bộ feature dẫn xuất từ unit_sales.

    Khớp công thức NWRMSLE (RMSLE): clip(.,0) rồi log1p -> MSE trên log ≈ tối ưu metric.
    KHÔNG đụng tới exog (oil/onpromotion/holiday/perishable/payday/promo-count) vì khác scale.
    Inverse: expm1(x) = exp(x) - 1.
    """
    df = df.copy()
    for col in _sales_derived_cols(df):
        df[col] = df[col].clip(lower=0).transform("log1p")
    return df
