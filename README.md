# Favorita Grocery Sales Forecasting — Model Comparison

Comparative framework for time series forecasting models on the **Corporación Favorita Grocery Sales Forecasting** dataset (Kaggle: `favorita-grocery-sales-forecasting`).

## Models

| Model | Type | Fitting Strategy | Scope |
|-------|------|------------------|-------|
| ARIMA | Statistical | Per-series | Representative sample (~40 store-item) |
| SARIMAX | Statistical | Per-series + Fourier exog | Representative sample (~40 store-item) |
| Prophet | Statistical | Per-series | Representative sample (~40 store-item) |
| XGBoost | ML | Global (store/item + exog features) | All series in cluster |
| LSTM | Deep Learning | Global (store/item embeddings) | All series in cluster |

> **Scope = 1 store cluster.** Bộ Favorita đầy đủ (54 stores × ~4100 items, ~125M dòng) quá lớn để fit. Project lọc **1 nhóm store** (một `cluster`, cấu hình qua `store_filter` trong `configs/base.yaml`) cho mọi model. Mọi model dùng CÙNG subset + CÙNG temporal split (train_end 2017-05-15, test 2017-05-16 → 08-15).
>
> **Per-series sample:** ARIMA/SARIMAX/Prophet fit 1 model/chuỗi. Số chuỗi trong 1 cluster vẫn lớn → chọn mẫu ~40 chuỗi trải đều theo volume (`series_sample` trong `configs/base.yaml`). Global models (XGBoost, LSTM) dùng toàn bộ chuỗi trong cluster.
>
> **SARIMAX dùng Fourier exog xác định (deterministic):** weekly K=3 + yearly K=2. Oil/onpromotion tương lai không biết trước → dùng làm exog gây rò rỉ (leakage), nên SARIMAX chỉ dùng Fourier terms tính được cho mọi ngày. GPU ưu tiên cho XGBoost/LSTM.

## Dataset

[Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) — multi-file, daily 2013-01-01 → 2017-08-15:

| File | Cột chính |
|------|-----------|
| `train.csv` | id, date, store_nbr, item_nbr, unit_sales, onpromotion |
| `stores.csv` | store_nbr, city, state, type, cluster |
| `items.csv` | item_nbr, family, class, perishable |
| `transactions.csv` | date, store_nbr, transactions |
| `oil.csv` | date, dcoilwtico (giá dầu WTI) |
| `holidays_events.csv` | date, type, locale, locale_name, description, transferred |

Scale: 54 stores × ~4100 items. Metric chính thức: **NWRMSLE** (Normalized Weighted RMSLE — trọng số 1.25 cho hàng perishable, 1.0 còn lại; tính trên log1p giá trị clip ≥ 0).

Đặc điểm: `unit_sales` có thể **âm** (trả hàng) → clip về 0; `onpromotion` có NaN → False. Exog **thực**: onpromotion, oil price, holidays (national/regional/local), perishable flag. `test.csv` của Kaggle (16 ngày) **không có nhãn** → đánh giá dùng ~90 ngày cuối train (train ≤ 2017-05-15, test 2017-05-16 → 08-15).

### Download

```bash
kaggle competitions download -c favorita-grocery-sales-forecasting -p data/raw/
# giải nén tất cả file (.csv.7z / .zip tùy phiên bản):
7z x 'data/raw/*.7z' -odata/raw/   # hoặc unzip
# cần: data/raw/{train,stores,items,transactions,oil,holidays_events}.csv
```

## Setup

```bash
uv venv /root/.venvs/demand_forecasting --python 3.11
uv sync
source /root/.venvs/demand_forecasting/bin/activate
```

## Usage

Schema (entity, target, time) khai báo trong `configs/base.yaml` → `schema:` block. Entity là cặp `(store_nbr, item_nbr)` gộp thành `series_id` (= `pd.factorize`). Nhóm store dự báo chọn qua `store_filter`.

```bash
# Train 1 model với config mặc định
python scripts/train.py --model arima
python scripts/train.py --model sarimax
python scripts/train.py --model prophet
python scripts/train.py --model xgboost
python scripts/train.py --model lstm

# Override param bất kỳ qua --set
python scripts/train.py --model lstm --set model.hidden_size=128 --set model.seq_len=90

# Chọn số chuỗi cho per-series models
python scripts/train.py --model arima --set series_sample.n_series=50

# Đổi nhóm store dự báo
python scripts/train.py --model xgboost --set store_filter.value=[2]

# Cross-validation walk-forward
python scripts/evaluate.py --model xgboost --cv expanding --n-splits 3 --eval-days 90

# So sánh nhiều model (sorted by NWRMSLE)
python scripts/compare_cv.py --models arima,sarimax,prophet,xgboost,lstm
```

## Project Structure

```
demand_forecasting/
├── configs/      # YAML: base (schema/split/store_filter/exog) + per-model + features
├── data/raw/     # train.csv, stores.csv, items.csv, oil.csv, holidays_events.csv (gitignored)
├── src/
│   ├── data/     # loader, cleaner (multi-file join + exog), features, preprocessor
│   ├── models/   # BaseModel + ARIMA/SARIMAX/Prophet/XGBoost/LSTM
│   ├── analysis/ # ACF/PACF, stationarity, residual diagnostics (Box-Jenkins)
│   ├── evaluation/ # NWRMSLE + metrics, walk-forward CV, comparison
│   └── utils/
├── notebooks/    # EDA, comparison templates
├── scripts/      # train, evaluate, compare CLIs
└── plans/reports/ # research notes (EDA + feature engineering)
```

## Evaluation

- **Primary metric:** NWRMSLE (competition metric — perishable-weighted RMSLE)
- **Secondary:** RMSE, MAE, MAPE
- **Cross-validation:** walk-forward expanding window
