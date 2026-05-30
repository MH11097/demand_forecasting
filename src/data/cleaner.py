"""Data cleaning + multi-file join pipeline cho Favorita Grocery Sales Forecasting.

Favorita gồm nhiều file:
    train.csv [id, date, store_nbr, item_nbr, unit_sales, onpromotion]
    stores.csv [store_nbr, city, state, type, cluster]
    items.csv [item_nbr, family, class, perishable]
    transactions.csv [date, store_nbr, transactions]
    oil.csv [date, dcoilwtico]
    holidays_events.csv [date, type, locale, locale_name, description, transferred]

Pipeline: load tất cả → join metadata (stores, items) → merge exog theo date
(oil nội suy, holidays → cờ boolean theo locale) → clip unit_sales âm về 0 →
onpromotion NaN→0 → lọc 1 nhóm store (store_filter) → sort.

Functional style: mỗi hàm nhận/trả DataFrame. build_dataset() là điểm vào tổng.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Các loại sự kiện trong holidays_events.csv được coi là NGÀY NGHỈ thực sự.
# "Work Day" là ngày làm bù (KHÔNG nghỉ) -> loại; "Transfer" là ngày lễ dời tới.
_HOLIDAY_TYPES = {"Holiday", "Additional", "Bridge", "Transfer", "Event"}


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
def _raw_path(raw_dir: str, fname: str) -> Path:
    return Path(raw_dir) / fname


def load_raw(raw_dir: str, config: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Load train.csv + các bảng metadata/exog có khai báo trong config.

    Returns:
        (train_df, tables) — tables là dict {stores, items, oil, holidays, transactions}
        (giá trị None nếu file không khai báo hoặc exog tắt). Giữ tách biệt để
        merge ở các bước sau.
    """
    files = (config or {}).get("data", {}).get("files", {}) if config else {}
    train = pd.read_csv(
        _raw_path(raw_dir, "train.csv"), parse_dates=["date"], low_memory=False
    )

    def _opt(name: str, **kw) -> pd.DataFrame | None:
        fname = files.get(name)
        if not fname:
            return None
        p = _raw_path(raw_dir, fname)
        return pd.read_csv(p, **kw) if p.exists() else None

    tables = {
        "stores": _opt("stores"),
        "items": _opt("items"),
        "oil": _opt("oil", parse_dates=["date"]),
        "holidays": _opt("holidays", parse_dates=["date"]),
        "transactions": _opt("transactions", parse_dates=["date"]),
    }
    return train, tables


# ----------------------------------------------------------------------------
# Joins / merges
# ----------------------------------------------------------------------------
def join_metadata(train: pd.DataFrame, stores: pd.DataFrame | None,
                  items: pd.DataFrame | None) -> pd.DataFrame:
    """Join stores (theo store_nbr) và items (theo item_nbr) vào train."""
    df = train
    if stores is not None:
        df = df.merge(stores, on="store_nbr", how="left")
    if items is not None:
        df = df.merge(items, on="item_nbr", how="left")
    return df


def merge_oil(df: pd.DataFrame, oil: pd.DataFrame | None) -> pd.DataFrame:
    """Merge giá dầu theo date; nội suy ngày thiếu (cuối tuần/lễ không có giá)."""
    if oil is None:
        return df
    oil = oil.sort_values("date").copy()
    # dcoilwtico thiếu nhiều ngày -> nội suy tuyến tính + ffill/bfill 2 đầu
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="linear").ffill().bfill()
    return df.merge(oil, on="date", how="left")


def merge_holidays(df: pd.DataFrame, holidays: pd.DataFrame | None) -> pd.DataFrame:
    """Tạo cờ is_holiday (+ national/regional/local) theo locale.

    - National: áp dụng cho mọi store.
    - Regional: khớp theo state của store (locale_name == state).
    - Local: khớp theo city của store (locale_name == city).
    Hàng transferred=True KHÔNG phải nghỉ ở ngày gốc (đã dời -> type=Transfer lo ngày mới).
    """
    if holidays is None:
        return df
    h = holidays.copy()
    h = h[h["type"].isin(_HOLIDAY_TYPES)]
    h = h[~h["transferred"].fillna(False).astype(bool)]

    nat = set(map(pd.Timestamp, h.loc[h["locale"] == "National", "date"].unique()))
    reg = h.loc[h["locale"] == "Regional", ["date", "locale_name"]].drop_duplicates()
    loc = h.loc[h["locale"] == "Local", ["date", "locale_name"]].drop_duplicates()
    reg_set = set(zip(reg["date"], reg["locale_name"]))
    loc_set = set(zip(loc["date"], loc["locale_name"]))

    df = df.copy()
    df["holiday_national"] = df["date"].isin(nat).astype(int)
    if "state" in df.columns:
        df["holiday_regional"] = [
            int((d, s) in reg_set) for d, s in zip(df["date"], df["state"])
        ]
    else:
        df["holiday_regional"] = 0
    if "city" in df.columns:
        df["holiday_local"] = [
            int((d, c) in loc_set) for d, c in zip(df["date"], df["city"])
        ]
    else:
        df["holiday_local"] = 0
    df["is_holiday"] = (
        df["holiday_national"] | df["holiday_regional"] | df["holiday_local"]
    ).astype(int)
    return df


def merge_transactions(df: pd.DataFrame, transactions: pd.DataFrame | None,
                       config: dict | None = None) -> pd.DataFrame:
    """Merge số giao dịch/ngày/store. ⚠ Chỉ có cho ngày train -> mặc định TẮT (leakage)."""
    use = (config or {}).get("exog", {}).get("use_transactions", False)
    if transactions is None or not use:
        return df
    return df.merge(transactions, on=["date", "store_nbr"], how="left")


# ----------------------------------------------------------------------------
# Clean
# ----------------------------------------------------------------------------
def clip_negative_sales(df: pd.DataFrame) -> pd.DataFrame:
    """unit_sales âm = trả hàng -> clip 0 (khớp công thức NWRMSLE dùng log1p(clip(.,0)))."""
    df = df.copy()
    if "unit_sales" in df.columns:
        df["unit_sales"] = df["unit_sales"].clip(lower=0)
    return df


def fill_onpromotion(df: pd.DataFrame) -> pd.DataFrame:
    """onpromotion có NaN (thiếu thông tin KM) -> coi như không KM (0)."""
    df = df.copy()
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].fillna(False).astype(int)
    return df


def apply_store_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Lọc 1 nhóm store theo store_filter.{by, value} (cluster/type/store_nbr)."""
    sf = config.get("store_filter")
    if not sf or sf.get("value") in (None, [], ""):
        return df
    by, value = sf["by"], sf["value"]
    if by not in df.columns:
        raise KeyError(
            f"store_filter.by='{by}' không có trong cột (cần join stores.csv trước)."
        )
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return df[df[by].isin(values)].reset_index(drop=True)


def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """date=datetime; store_nbr/item_nbr=int; unit_sales=float; onpromotion=int."""
    df = df.copy()
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    for col in ["store_nbr", "item_nbr"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "unit_sales" in df.columns:
        df["unit_sales"] = df["unit_sales"].astype(float)
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].astype(int)
    if "perishable" in df.columns:
        df["perishable"] = df["perishable"].fillna(0).astype(int)
    return df


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort [store_nbr, item_nbr, date] để lag/rolling đúng thứ tự thời gian."""
    if {"store_nbr", "item_nbr", "date"}.issubset(df.columns):
        df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)
    return df


def build_dataset(config: dict) -> pd.DataFrame:
    """Điểm vào: load → join → merge exog → clean → lọc nhóm store → sort."""
    raw_dir = config["data"]["raw_dir"]
    train, tables = load_raw(raw_dir, config)
    df = join_metadata(train, tables["stores"], tables["items"])
    df = apply_store_filter(df, config)          # lọc sớm -> nhẹ các bước sau
    df = merge_oil(df, tables["oil"])
    df = merge_holidays(df, tables["holidays"])
    df = merge_transactions(df, tables["transactions"], config)
    df = clip_negative_sales(df)
    df = fill_onpromotion(df)
    df = ensure_types(df)
    df = sort_data(df)
    return df


# ----------------------------------------------------------------------------
# Validate / save
# ----------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> dict:
    """Kiểm tra chất lượng sau clean, in summary, trả report dict."""
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "null_total": int(df.isnull().sum().sum()),
        "null_counts": df.isnull().sum().to_dict(),
    }
    if "store_nbr" in df.columns:
        report["stores"] = int(df["store_nbr"].nunique())
    if "item_nbr" in df.columns:
        report["items"] = int(df["item_nbr"].nunique())
    if "unit_sales" in df.columns:
        report["sales_min"] = float(df["unit_sales"].min())
        report["sales_max"] = float(df["unit_sales"].max())
        report["sales_mean"] = float(df["unit_sales"].mean())

    print(
        f"Rows: {report['rows']:,} | Cols: {report['columns']} "
        f"| Stores: {report.get('stores', '?')} | Items: {report.get('items', '?')}"
    )
    print(f"Nulls total: {report['null_total']}")
    if "unit_sales" in df.columns:
        print(
            f"unit_sales range: [{report['sales_min']:.1f}, {report['sales_max']:.1f}], "
            f"mean={report['sales_mean']:.2f}"
        )
    return report


def save(df: pd.DataFrame, path: str, fmt: str = "csv") -> None:
    """Lưu DataFrame ra csv hoặc feather."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "feather":
        df.reset_index(drop=True).to_feather(f"{out}.feather")
    else:
        df.to_csv(f"{out}.csv", index=False)
    print(f"Saved {out}.{fmt} ({len(df):,} rows)")
