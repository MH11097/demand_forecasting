"""Tests for the Favorita data layer: cleaner + loader + features + preprocessor.

Dùng fixture tổng hợp nhỏ hình dạng Favorita (multi-file) — không cần tải dữ liệu thật.
"""

import numpy as np
import pandas as pd

from src.data import cleaner, features, loader, preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _train():
    dates = pd.date_range("2017-01-01", periods=4, freq="D")
    rows = []
    rid = 0
    for store in (1, 2):
        for item in (10, 20):
            for d in dates:
                rows.append({"id": rid, "date": d, "store_nbr": store,
                             "item_nbr": item, "unit_sales": 5.0, "onpromotion": 0})
                rid += 1
    df = pd.DataFrame(rows)
    df.loc[0, "unit_sales"] = -3.0          # trả hàng -> phải clip 0
    df.loc[1, "onpromotion"] = np.nan       # thiếu KM -> phải fill 0
    return df


def _stores():
    return pd.DataFrame({
        "store_nbr": [1, 2],
        "city": ["Quito", "Guayaquil"],
        "state": ["Pichincha", "Guayas"],
        "type": ["A", "B"],
        "cluster": [1, 2],
    })


def _items():
    return pd.DataFrame({
        "item_nbr": [10, 20],
        "family": ["GROCERY", "BEVERAGES"],
        "class": [1001, 2002],
        "perishable": [1, 0],
    })


def _oil():
    # có 1 ngày NaN -> interpolate phải lấp
    return pd.DataFrame({
        "date": pd.date_range("2017-01-01", periods=4, freq="D"),
        "dcoilwtico": [50.0, np.nan, 52.0, 53.0],
    })


def _holidays():
    return pd.DataFrame({
        "date": pd.to_datetime(["2017-01-02", "2017-01-03", "2017-01-04"]),
        "type": ["Holiday", "Holiday", "Holiday"],
        "locale": ["National", "Regional", "National"],
        "locale_name": ["Ecuador", "Pichincha", "Ecuador"],
        "description": ["a", "b", "c"],
        "transferred": [False, False, True],   # 2017-01-04 transferred -> KHÔNG nghỉ
    })


# ---------------------------------------------------------------------------
# cleaner
# ---------------------------------------------------------------------------
def test_join_metadata():
    df = cleaner.join_metadata(_train(), _stores(), _items())
    for col in ("cluster", "type", "city", "family", "perishable"):
        assert col in df.columns


def test_clip_negative_sales():
    df = cleaner.clip_negative_sales(_train())
    assert df["unit_sales"].min() >= 0


def test_fill_onpromotion():
    df = cleaner.fill_onpromotion(_train())
    assert df["onpromotion"].isnull().sum() == 0
    assert df["onpromotion"].dtype.kind in "iu"


def test_apply_store_filter_by_cluster():
    df = cleaner.join_metadata(_train(), _stores(), _items())
    config = {"store_filter": {"by": "cluster", "value": [1]}}
    out = cleaner.apply_store_filter(df, config)
    assert set(out["store_nbr"].unique()) == {1}


def test_merge_oil_interpolates():
    df = cleaner.merge_oil(_train(), _oil())
    assert df["dcoilwtico"].isnull().sum() == 0
    # ngày NaN (2017-01-02) được nội suy ~51
    val = df.loc[df["date"] == "2017-01-02", "dcoilwtico"].iloc[0]
    assert 50.0 < val < 52.0


def test_merge_holidays_locale_and_transferred():
    df = cleaner.join_metadata(_train(), _stores(), _items())
    df = cleaner.merge_holidays(df, _holidays())
    # 2017-01-02 national -> mọi store nghỉ
    assert df.loc[df["date"] == "2017-01-02", "is_holiday"].all()
    # 2017-01-03 regional Pichincha -> chỉ store 1 (Pichincha)
    reg = df[df["date"] == "2017-01-03"]
    assert reg.loc[reg["store_nbr"] == 1, "is_holiday"].all()
    assert not reg.loc[reg["store_nbr"] == 2, "is_holiday"].any()
    # 2017-01-04 transferred -> KHÔNG nghỉ
    assert not df.loc[df["date"] == "2017-01-04", "is_holiday"].any()


def test_build_dataset(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _train().to_csv(raw / "train.csv", index=False)
    _stores().to_csv(raw / "stores.csv", index=False)
    _items().to_csv(raw / "items.csv", index=False)
    _oil().to_csv(raw / "oil.csv", index=False)
    _holidays().to_csv(raw / "holidays_events.csv", index=False)

    config = {
        "data": {"raw_dir": str(raw), "files": {
            "stores": "stores.csv", "items": "items.csv",
            "oil": "oil.csv", "holidays": "holidays_events.csv"}},
        "store_filter": {"by": "cluster", "value": [1]},
        "exog": {"use_transactions": False},
    }
    df = cleaner.build_dataset(config)
    assert set(df["store_nbr"].unique()) == {1}     # đã lọc cluster 1
    assert df["unit_sales"].min() >= 0              # đã clip
    assert "perishable" in df.columns and "is_holiday" in df.columns


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------
def test_add_series_id_factorize():
    df = loader._add_series_id(_train().copy())
    # 2 store × 2 item = 4 chuỗi -> series_id 0..3
    assert df["series_id"].nunique() == 4
    assert df["series_id"].min() == 0 and df["series_id"].max() == 3


def test_select_series_samples():
    df = loader._add_series_id(_train().copy())
    out = loader.select_series(df, {"series_sample": {"n_series": 2}, "seed": 1})
    assert out["series_id"].nunique() == 2


def test_densify_fills_missing_dates():
    df = loader._add_series_id(_train().copy())
    df = df[df["date"] != "2017-01-02"]            # bỏ 1 ngày khỏi mọi chuỗi
    dens = loader._densify(df, {"densify": True})
    # mỗi chuỗi đủ 4 ngày trở lại, unit_sales ngày thêm = 0
    assert (dens.groupby("series_id")["date"].nunique() == 4).all()
    assert dens.loc[dens["date"] == "2017-01-02", "unit_sales"].eq(0).all()


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _featured():
    df = cleaner.join_metadata(_train(), _stores(), _items())
    df = cleaner.merge_oil(df, _oil())
    df = cleaner.merge_holidays(df, _holidays())
    df = cleaner.clip_negative_sales(cleaner.fill_onpromotion(df))
    df = loader._add_series_id(df)
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def test_add_lag_features_naming():
    df = features.add_lag_features(_featured())
    assert "unit_sales_lag_1" in df.columns
    assert "sales_lag_1" not in df.columns       # tên cũ phải biến mất


def test_apply_log_transform_clips_and_logs():
    df = _featured()
    df["unit_sales"] = [-2.0] + [5.0] * (len(df) - 1)
    out = features.apply_log_transform(df)
    assert out["unit_sales"].iloc[0] == 0.0       # clip(-2,0)->0 -> log1p(0)=0
    assert np.isclose(out["unit_sales"].iloc[1], np.log1p(5.0))


def test_exog_feature_builders():
    df = features.add_all_features(_featured())
    for col in ("promo_rolling_rate", "oil_lag_7", "is_holiday", "perishable"):
        assert col in df.columns


# ---------------------------------------------------------------------------
# preprocessor
# ---------------------------------------------------------------------------
def test_entity_index_and_vocab():
    df = preprocessor._build_entity_index(_featured())
    assert df["store_idx"].min() == 0 and df["store_idx"].max() == 1   # 2 store
    assert df["item_idx"].max() == 1                                   # 2 item
    n_store, n_item = preprocessor.entity_vocab_sizes(df)
    assert (n_store, n_item) == (2, 2)


def test_preprocess_split_and_scaling():
    df = features.add_all_features(_featured())
    config = {"split": {"train_end": "2017-01-02", "test_start": "2017-01-03",
                        "test_end": "2017-01-04"}, "model": {}}
    train_df, val_df, test_df, scaler = preprocessor.preprocess(df, config)
    assert len(train_df) > 0 and len(test_df) > 0
    # cờ binary/id KHÔNG bị scale (giữ giá trị gốc 0/1)
    assert set(train_df["onpromotion"].unique()).issubset({0, 1})
    assert set(train_df["perishable"].unique()).issubset({0, 1})
