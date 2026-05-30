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

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Event có thể làm thay đổi cầu nhưng không đồng nghĩa với ngày nghỉ. Giữ cờ riêng để
# model phân biệt World Cup, Black Friday, động đất... với holiday thực sự.
_HOLIDAY_TYPES = {"Holiday", "Additional", "Bridge", "Transfer"}
_EVENT_TYPES = {"Event"}
CACHE_SCHEMA_VERSION = 2


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
    """Merge giá dầu theo date; nội suy CHO MỌI NGÀY LỊCH (không chỉ ngày có trong oil.csv).

    oil.csv chỉ có dòng cho ngày giao dịch (thiếu hẳn cuối tuần/lễ). Nếu chỉ interpolate
    trong các dòng sẵn có rồi left-merge, các ngày lịch vắng mặt khỏi oil.csv sẽ NaN
    (sau zero-fill, ~30% ngày là cuối tuần → NaN hàng loạt). Khắc phục: reindex oil về
    DẢI NGÀY LIÊN TỤC (phủ cả khoảng của df) rồi nội suy → mọi ngày có giá dầu.
    """
    if oil is None:
        return df
    oil = oil.sort_values("date").copy()
    lo = min(oil["date"].min(), df["date"].min())
    hi = max(oil["date"].max(), df["date"].max())
    full = pd.date_range(lo, hi, freq="D")
    oil = oil.set_index("date").reindex(full).rename_axis("date")
    # nội suy tuyến tính + ffill/bfill 2 đầu trên lưới ngày đầy đủ
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="linear").ffill().bfill()
    oil = oil.reset_index()
    return df.merge(oil, on="date", how="left")


def merge_holidays(df: pd.DataFrame, holidays: pd.DataFrame | None) -> pd.DataFrame:
    """Tạo cờ holiday và event riêng biệt theo locale.

    - National: áp dụng cho mọi store.
    - Regional: khớp theo state của store (locale_name == state).
    - Local: khớp theo city của store (locale_name == city).
    Hàng transferred=True KHÔNG phải nghỉ ở ngày gốc (đã dời -> type=Transfer lo ngày mới).
    """
    if holidays is None:
        return df
    df = df.copy()
    h = holidays.copy()
    h = h[~h["transferred"].fillna(False).astype(bool)]

    def _add_locale_flags(frame: pd.DataFrame, events: pd.DataFrame, prefix: str) -> pd.DataFrame:
        nat = set(map(pd.Timestamp, events.loc[events["locale"] == "National", "date"].unique()))
        reg = events.loc[events["locale"] == "Regional", ["date", "locale_name"]].drop_duplicates()
        loc = events.loc[events["locale"] == "Local", ["date", "locale_name"]].drop_duplicates()
        reg_set = set(zip(reg["date"], reg["locale_name"]))
        loc_set = set(zip(loc["date"], loc["locale_name"]))

        frame[f"{prefix}_national"] = frame["date"].isin(nat).astype(int)
        frame[f"{prefix}_regional"] = (
            [int((d, s) in reg_set) for d, s in zip(frame["date"], frame["state"])]
            if "state" in frame.columns else 0
        )
        frame[f"{prefix}_local"] = (
            [int((d, c) in loc_set) for d, c in zip(frame["date"], frame["city"])]
            if "city" in frame.columns else 0
        )
        frame[f"is_{prefix}"] = (
            frame[f"{prefix}_national"]
            | frame[f"{prefix}_regional"]
            | frame[f"{prefix}_local"]
        ).astype(int)
        return frame

    df = _add_locale_flags(df, h[h["type"].isin(_HOLIDAY_TYPES)], "holiday")
    df = _add_locale_flags(df, h[h["type"].isin(_EVENT_TYPES)], "event")
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
    """Clip returns for target metric, but retain EDA-only return indicators."""
    df = df.copy()
    if "unit_sales" in df.columns:
        df["was_return"] = (df["unit_sales"] < 0).astype(int)
        df["returned_units"] = (-df["unit_sales"].clip(upper=0)).astype(float)
        df["unit_sales"] = df["unit_sales"].clip(lower=0)
    return df


def fill_onpromotion(df: pd.DataFrame) -> pd.DataFrame:
    """onpromotion có NaN (thiếu thông tin KM) -> coi như không KM (0)."""
    df = df.copy()
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].fillna(False).astype(int)
    return df


def reindex_fill_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Densify mỗi chuỗi (store_nbr, item_nbr) thành lưới ngày LIÊN TỤC.

    Favorita train.csv KHÔNG ghi ngày bán = 0 (implicit zeros) → ~40% ngày trong khoảng
    sống của chuỗi bị thiếu dòng. Nếu để vậy, groupby.shift(lag) dịch theo DÒNG chứ không
    theo NGÀY → mọi lag/rolling sai lệch. Hàm này reindex mỗi chuỗi về dải ngày liên tục.

    ⚠ Bound theo NGÀY BÁN ĐẦU TIÊN → CUỐI của TỪNG chuỗi (không phải 2013-01-01 toàn cục):
    94.6% chuỗi Favorita ra đời giữa chừng → reindex toàn cục sẽ bịa hàng triệu "zero giả"
    cho giai đoạn sản phẩm chưa tồn tại, làm lệch model + group-mean encodings.

    Gọi TRƯỚC merge_oil/merge_holidays để exog theo ngày (giá dầu, cờ lễ) gán ĐÚNG cho
    ngày chèn thêm (vd ngày 25/12 thiếu sẽ nhận holiday=1, không bị ffill nhầm từ 24/12).

    Cột thêm: is_imputed (1 = dòng chèn zero-fill, 0 = dòng quan sát thật).
    Vectorized: build full grid bằng index.repeat thay vì loop từng chuỗi.
    """
    key = ["store_nbr", "item_nbr"]
    if not set(key + ["date"]).issubset(df.columns):
        return df

    spans = df.groupby(key)["date"].agg(min_date="min", max_date="max")
    spans["ndays"] = (spans["max_date"] - spans["min_date"]).dt.days + 1

    # nở mỗi chuỗi thành ndays dòng, gán date = min_date + offset (0..ndays-1)
    grid = spans.loc[spans.index.repeat(spans["ndays"].to_numpy())].copy()
    offset = grid.groupby(level=key).cumcount().to_numpy()
    grid["date"] = grid["min_date"].to_numpy() + pd.to_timedelta(offset, unit="D")
    grid = grid.reset_index()[key + ["date"]]

    merged = grid.merge(df, on=key + ["date"], how="left", indicator=True)
    merged["is_imputed"] = (merged["_merge"] == "left_only").astype(int)
    merged = merged.drop(columns="_merge")

    merged["unit_sales"] = merged["unit_sales"].fillna(0.0)
    if "onpromotion" in merged.columns:
        merged["onpromotion"] = merged["onpromotion"].fillna(0)

    # thuộc tính tĩnh của chuỗi (store/item) → ffill/bfill trong từng chuỗi
    static = [c for c in ["city", "state", "type", "cluster", "family", "class", "perishable"]
              if c in merged.columns]
    merged = merged.sort_values(key + ["date"]).reset_index(drop=True)
    if static:
        merged[static] = merged.groupby(key)[static].ffill().bfill()
    return merged


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
    # zero-fill ngày thiếu (implicit zeros) TRƯỚC khi merge exog theo ngày & build lag/rolling
    if config.get("clean", {}).get("zero_fill", True):
        df = reindex_fill_dates(df)
    exog = config.get("exog", {})
    if exog.get("use_oil", True):
        df = merge_oil(df, tables["oil"])
    if exog.get("use_holidays", True):
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


def cache_signature(config: dict) -> dict:
    """Config subset that changes the expensive cleaned cache."""
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "store_filter": config.get("store_filter"),
        "clean": config.get("clean", {}),
        "exog": {
            key: config.get("exog", {}).get(key, default)
            for key, default in (
                ("use_oil", True),
                ("use_holidays", True),
                ("use_transactions", False),
            )
        },
        "files": config.get("data", {}).get("files", {}),
    }


def save(df: pd.DataFrame, path: str, fmt: str = "csv", config: dict | None = None) -> None:
    """Lưu DataFrame ra csv hoặc feather (feather nén zstd để chia sẻ qua git < 100MB)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "feather":
        # zstd: 12.46M dòng 291MB (lz4 mặc định) -> ~72MB; read_feather tự giải nén
        df.reset_index(drop=True).to_feather(
            f"{out}.feather", compression="zstd", compression_level=19
        )
    else:
        df.to_csv(f"{out}.csv", index=False)
    if config is not None:
        with open(f"{out}.manifest.json", "w", encoding="utf-8") as f:
            json.dump({"cache_signature": cache_signature(config)}, f, indent=2, sort_keys=True)
    print(f"Saved {out}.{fmt} ({len(df):,} rows)")
