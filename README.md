# Favorita Grocery Sales Forecasting — So sánh mô hình dự báo

Khung so sánh nhiều mô hình dự báo chuỗi thời gian trên bộ **Corporación Favorita Grocery Sales Forecasting** (Kaggle: `favorita-grocery-sales-forecasting`). Tất cả mô hình dùng **cùng dữ liệu, cùng feature, cùng split** để so sánh công bằng theo metric chính thức **NWRMSLE**.

---

| Việc | Lệnh |
|------|------|
| **Train + đánh giá test 1 model** | `python scripts/train.py train --model xgboost` |
| Train **tương tác** (menu chọn model + tham số) | `python scripts/train.py train` |
| **Cross-validation** (đánh giá kỹ 1 model) | `python scripts/evaluate.py --model xgboost --cv expanding --n-splits 3 --eval-days 90` |
| **So sánh nhiều model** (sắp theo NWRMSLE) | `python scripts/compare_cv.py --models arima,sarimax,prophet,xgboost,lstm` |
| Tạo **cache dữ liệu** (chạy 1 lần, train sau sẽ nhanh) | `python scripts/clean_data.py` |
| **EDA** trực quan | mở `notebooks/01_eda.ipynb` |

---

## ❓ "Train" và "Test" chạy file nào?

- **Train + Test gộp 1 lệnh → `scripts/train.py`.** Một lần chạy sẽ: load dữ liệu → feature engineering → train trên tập train (≤ 2017-05-15) → **tự động đánh giá trên tập test** (2017-05-16 → 08-15) → in `Test metrics` (NWRMSLE...) → lưu model + biểu đồ + `result.json` vào `results/<model>/...`.
- **Không có file `test.py` riêng.** "Test" = ~90 ngày cuối của train (Kaggle test thật 16 ngày không có nhãn). `train.py` lo cả train lẫn test.
- **Đánh giá chuyên sâu (cross-validation walk-forward)** → `scripts/evaluate.py` (1 model) hoặc `scripts/compare_cv.py` (nhiều model, ra bảng so sánh).

Ví dụ vòng đời đầy đủ:
```bash
python scripts/clean_data.py                                   # 1) tạo cache (1 lần)
python scripts/train.py train --model xgboost                  # 2) train + test 1 model
python scripts/train.py train --model all                      # 3) train + test tất cả
python scripts/compare_cv.py --models xgboost,lstm,prophet     # 4) so sánh qua CV
```

Override tham số bất kỳ qua `--set key=value` (dotted key):
```bash
python scripts/train.py train --model lstm --set model.hidden_size=128 --set model.seq_len=90
python scripts/train.py train --model arima --set series_sample.n_series=50
python scripts/train.py train --model xgboost --set store_filter.value=[2]   # đổi nhóm store
python scripts/train.py train --model xgboost --set forecast_strategy=recursive
```

---

## Cài đặt

```bash
uv venv /root/.venvs/demand_forecasting --python 3.11
uv sync
source /root/.venvs/demand_forecasting/bin/activate
```

---

# 🧪 PHẦN A — CẤU HÌNH THỰC NGHIỆM

## A.1. Mô hình

| Model | Loại | Cách fit | Phạm vi |
|-------|------|----------|---------|
| ARIMA | Thống kê | Per-series, không exog (baseline) | Mẫu ~40 chuỗi đại diện |
| SARIMAX | Thống kê | Per-series + shared temporal exog | Mẫu ~40 chuỗi đại diện |
| Prophet | Thống kê | Per-series + native seasonality + regressors | Mẫu ~40 chuỗi đại diện |
| XGBoost | ML | Global + shared exog + sales history | Toàn bộ chuỗi trong subset DAIRY |
| LSTM | Deep Learning | Global + shared exog + sales history | Toàn bộ chuỗi trong subset DAIRY |

- **Phạm vi = DAIRY tại 1 cluster store.** Bộ Favorita đầy đủ (54 store × ~4100 item, ~125M dòng) quá lớn → lọc `cluster=[1]` bằng `store_filter`, rồi giữ toàn bộ SKU có `family=[DAIRY]` bằng `item_filter`.
- **Kết quả chỉ đại diện cho subset DAIRY tại cluster 1**, không đại diện cho toàn bộ danh mục Favorita.
- **Per-series (ARIMA/SARIMAX/Prophet)** fit 1 model/chuỗi → chọn mẫu ~40 chuỗi (`series_sample`). **Global (XGBoost/LSTM)** dùng toàn bộ chuỗi trong subset DAIRY (`series_sample.n_series=null`).
- **ARIMA** cố ý không dùng exog để làm baseline univariate dễ giải thích.

### Shared external-information benchmark

Các model hỗ trợ exog nhận cùng **nguồn thông tin bên ngoài khả dụng tại forecast
origin**. Không ép mọi model dùng cùng cách biểu diễn: Prophet có seasonality nội bộ,
trong khi SARIMAX nhận Fourier tường minh.

| Nguồn thông tin | Biểu diễn | Model dùng |
|-----------------|-----------|-----------|
| Mùa vụ lịch | Fourier tuần K=3 + năm K=2, hoặc native seasonality tương đương | SARIMAX, Prophet, XGBoost, LSTM |
| Ngày đặc biệt | `is_holiday`, `is_event` | SARIMAX, Prophet, XGBoost, LSTM |
| Chu kỳ chi tiêu | `is_payday` | SARIMAX, Prophet, XGBoost, LSTM |
| Lịch khuyến mãi | `onpromotion` | SARIMAX, Prophet, XGBoost, LSTM |
| Đặc tính sản phẩm | `perishable` | XGBoost, LSTM |

- **Fourier không thay thế exog thực.** Fourier mô tả quy luật lặp theo lịch; holiday
  và promotion mô tả tác động riêng của từng ngày.
- **`perishable` chỉ đưa vào model global.** Trong SARIMAX/Prophet per-series, mỗi model
  chỉ fit một `(store,item)` nên `perishable` là hằng số, không giải thích biến động theo ngày.
- **Oil và transactions không dùng trong benchmark chính.** Giá trị tương lai không tự
  nhiên biết trước tại forecast origin. Oil vẫn giữ trong cache để làm thí nghiệm phụ.

## A.2. Temporal split, validate & chiến lược dự báo (CHỐT — mọi người dùng chung)

### Chia tập theo NGÀY (không shuffle) — `configs/base.yaml → split`

| Tập | Khoảng ngày | Mặc định | Ghi chú |
|-----|-------------|----------|---------|
| **Train** | `≤ 2017-05-15` | ✅ bật | huấn luyện |
| **Validate** | `val_start → val_end` | ❌ **CHƯA bật** | xem dưới |
| **Test** | `2017-05-16 → 2017-08-15` (~92 ngày) | ✅ bật | đánh giá cuối |

- **1 file `data/cleaned/train_cleaned.feather` chứa đủ dữ liệu cho cả 3 tập** (toàn dải ngày). `preprocessor.preprocess()` cắt theo ngày lúc chạy — KHÔNG có file riêng cho từng tập.
- **Validate mặc định RỖNG** (base.yaml không khai báo `val_start/val_end`). Muốn có tập validate → thêm vào `split`:
  ```yaml
  split:
    train_end: "2017-04-15"
    val_start: "2017-04-16"
    val_end:   "2017-05-15"
    test_start: "2017-05-16"
    test_end:   "2017-08-15"
  ```
  (XGBoost/LSTM vẫn tự cắt val từ đuôi train cho early-stopping kể cả khi val_df rỗng.)
- ⚠ File cache là **DAIRY tại cluster 1** theo `store_filter` + `item_filter` hiện tại. Đổi một trong hai filter → **build lại cache** (`python scripts/clean_data.py`).
- Không leakage: split theo thời gian; group-mean encodings chỉ tính trên train (`train_end`); lag/rolling dùng `shift(1)`.

### Dự báo bao nhiêu ngày?

- **Đánh giá trên ~92 ngày test** (2017-05-16 → 2017-08-15) — đây là vùng có nhãn để chấm NWRMSLE.
- `forecast_horizon: 90` trong config = horizon mục tiêu. (Kaggle test THẬT chỉ 16 ngày và **không có nhãn** → dự án cắt ~92 ngày cuối train làm test có nhãn.)
- Theo từng loại model:
  - **Per-series (ARIMA/SARIMAX/Prophet):** forecast liên tục cả span test của từng chuỗi.
  - **Global XGBoost:** predict từng ngày test trực tiếp từ feature ngày đó.
  - **LSTM:** cửa sổ vào `seq_len=90` ngày → xuất `forecast_horizon=90` ngày.

### Direct / Multioutput / Recursive — `forecast_strategy`

| Chiến lược | Cách làm | Khi nào |
|-----------|----------|---------|
| **`multioutput`** (MẶC ĐỊNH) | 1 model xuất **H giá trị cùng lúc** | ổn định, dùng cho LSTM đa bước |
| `direct` | 1 model dự đoán đúng điểm **T+H** | dự báo 1 mốc xa |
| `recursive` | train **T+1** rồi lăn bánh H lần (dùng dự đoán làm input) | đơn giản, dễ tích lũy sai số |

Đổi qua `--set forecast_strategy=recursive`. Với sequence model, `model.seq_len` = số ngày lịch sử đầu vào; train.py tự ghép **context window = `seq_len + H − 1`** ngày vào đầu val/test để ngày đầu có đủ lịch sử.

### Các "knob" chốt chung (đừng đổi tùy tiện)

| Knob | Giá trị | Ý nghĩa |
|------|---------|---------|
| `use_log_sales` | `true` | log1p(clip) target — khớp NWRMSLE |
| `loss_fn` | `mse` | MSE trên log ≈ tối ưu NWRMSLE |
| `metrics.primary` | `nwrmsle` | metric competition |
| `store_filter` | `cluster=[1]` | nhóm store dùng chung |
| `item_filter` | `family=[DAIRY]` | ngành hàng duy nhất của benchmark giáo dục |
| `series_sample.n_series` | `40` (per-series) / `null` (global) | số chuỗi fit |
| `clean.zero_fill` | `true` | bù ngày bán 0 |
| `data.use_cleaned` | `true` | dùng cache feather |
| `features.use_oil` | `false` | loại oil khỏi benchmark chính; cache vẫn giữ |

## A.3. Cấu hình (configs/)

| File | Nội dung |
|------|----------|
| `base.yaml` | data paths, schema, `store_filter`, `item_filter`, `clean.zero_fill`, exog toggles, split dates, `forecast_horizon/strategy`, `use_cleaned`, `seed` |
| `features.yaml` | bật/tắt từng nhóm feature, danh sách lag/rolling windows |
| `{model}.yaml` | hyperparameter riêng từng model (arima/sarimax/prophet/xgboost/lstm) |

## A.4. Hyperparameter & tuning

Mỗi model có file `configs/<model>.yaml` chứa knob riêng — chỉnh trực tiếp hoặc override qua `--set model.<key>=<value>`:

| Model | Knob chính |
|-------|-----------|
| ARIMA | `model.order` `[p,d,q]`, `model.trend` |
| SARIMAX | `model.order`, `model.seasonal_order` `[P,D,Q,s=7]`, shared temporal exog |
| Prophet | `changepoint_prior_scale`, `seasonality_mode`, `seasonality_prior_scale`, shared regressors |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha/lambda`, `device` |
| LSTM | `hidden_size`, `num_layers`, `dropout`, `seq_len`, `forecast_horizon`, `epochs`, `patience` |

**Grid search** qua tổ hợp hyperparameter:
```bash
python scripts/train.py grid-search --model xgboost --grid '{"model.max_depth":[5,7,9],"model.n_estimators":[300,500]}'
```
Các script tuning chuyên sâu khác: `scripts/tune_arima_grid.py`, `tune_sarimax_*.py`, `tune-prophet-*.py`, `grid_search_xgboost.py`, `randomized_search_xgboost.py`.

## A.5. Đánh giá

- **Metric chính — NWRMSLE** (Normalized Weighted RMSLE, metric competition):

  `NWRMSLE = sqrt( Σ wᵢ·(log1p(predᵢ) − log1p(trueᵢ))² / Σ wᵢ )`, clip pred/true ≥ 0; trọng số `w = 1.25` cho hàng *perishable*, `1.0` còn lại. **Càng nhỏ càng tốt.**
- **Phụ:** RMSE, MAE, MAPE và RMSPE. Hai metric phần trăm chỉ tính trên các dòng
  `actual != 0`; NWRMSLE vẫn là metric chọn model vì xử lý được zero-sales.
- **Cross-validation:** walk-forward expanding/sliding (`evaluate.py`, `compare_cv.py`).

**Kết quả lưu ở đâu** — mỗi lần `train.py` chạy → `results/<model>/<param_slug>/` (slug gồm hyperparam + horizon + log + strategy + timestamp):
```
result.json                 # metrics (val + test), metadata, training_info, loss_summary
model.pkl                   # model đã train
loss_history.json           # train/val loss theo epoch (LSTM)
test_predictions.png, test_residuals.png, test_predictions_zoomed.png
val_predictions.png,  val_residuals.png,  val_predictions_zoomed.png
```

---

# 🗂 PHẦN B — XỬ LÝ DỮ LIỆU

## B.1. Dataset

[Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) — đa file, daily 2013-01-01 → 2017-08-15:

| File | Cột chính |
|------|-----------|
| `train.csv` | id, date, store_nbr, item_nbr, unit_sales, onpromotion |
| `stores.csv` | store_nbr, city, state, type, cluster |
| `items.csv` | item_nbr, family, class, perishable |
| `transactions.csv` | date, store_nbr, transactions |
| `oil.csv` | date, dcoilwtico (giá dầu WTI) |
| `holidays_events.csv` | date, type, locale, locale_name, description, transferred |

**Đặc điểm & xử lý:**
- Benchmark giáo dục giữ toàn bộ SKU thuộc `family=DAIRY` tại `cluster=1`; kết quả không đại diện cho toàn bộ Favorita.
- `unit_sales` có thể **âm** (trả hàng) → clip về 0 và giữ cờ return cho EDA; không bỏ sản phẩm từng có trả hàng. `onpromotion` có NaN → 0.
- ⭐ **Ngày bán = 0 bị bỏ qua** (implicit zeros): trên subset DAIRY, **24,1% ngày** trong khoảng sống của chuỗi không có dòng. Pipeline **zero-fill** mỗi chuỗi về lưới ngày liên tục (bound từ ngày bán đầu→cuối) để lag/rolling tính theo NGÀY, không theo dòng.
- Metric chính thức: **NWRMSLE** (xem A.5).

### Tải dữ liệu
```bash
kaggle competitions download -c favorita-grocery-sales-forecasting -p data/raw/
7z x 'data/raw/*.7z' -odata/raw/    # hoặc unzip
# cần: data/raw/{train,stores,items,transactions,oil,holidays_events}.csv
```

## B.2. Quy trình dữ liệu (pipeline)

```
data/raw/*.csv
   │  cleaner.build_dataset()  →  data/cleaned/train_cleaned.feather (cache)
   ▼
[1] load + join (stores, items)
[2] lọc nhóm store (store_filter)
[3] lọc family DAIRY (item_filter) trước khi nở dữ liệu
[4] ⭐ ZERO-FILL: reindex mỗi (store,item) về lưới ngày liên tục, +cờ is_imputed
[5] merge nguồn exog vào cache: oil (nội suy mọi ngày), holidays (cờ national/regional/local), onpromotion
[6] clip âm→0 (giữ cờ return để EDA), fill onpromotion→0
   │  loader.prepare_holdout_panel()              ← fixed panel theo forecast origin
   │  features.add_all_features(train_end=...)    ← lag/rolling/group-mean chỉ dùng lịch sử train
   ▼
[7] feature engineering (xem dưới) → log1p(clip) target
   │  preprocessor.preprocess()
   ▼
[8] series_id + store_idx/item_idx (embedding) → split theo thời gian (không shuffle) → RobustScaler
   ▼
   model input
```

**Các nhóm feature** (bật/tắt trong `configs/features.yaml`):
- **Time + cyclical**: year, month, dayofweek, is_weekend, sin/cos.
- **Fourier**: weekly K=3, yearly K=2 (mùa vụ lịch liên tục; Prophet dùng native seasonality tương đương).
- **Lag**: `unit_sales_lag_{1,7,14,28,30,60,90,365}` theo từng chuỗi.
- **Rolling**: mean/std/median cửa sổ 7/14/30 (shift(1) chống leakage).
- **Group-mean**: store_avg, item_avg, family_avg, series_dow_avg; fit train-only,
  leave-one-out trên train và freeze khi forecast.
- **Payday** (lương Ecuador 15 & cuối tháng): is_payday, days_since/to_payday.
- **Promo**: promo_rolling_rate, promo_count_7/14/30, days_since_last/until_next_promo.
- **Zero-sales**: zero_sales_last_28 (ngày ghi nhận bán 0; không suy diễn stockout).
- **Exog Favorita**: holiday/event flags riêng, onpromotion, perishable (cờ + trọng số
  NWRMSLE). dcoilwtico + oil_lag_7 vẫn tạo được khi bật `features.use_oil=true`.

> Feature engineering có thể tạo nhiều cột phục vụ EDA hoặc thí nghiệm phụ. Benchmark
> chính chỉ đưa shared external information và feature lịch sử chống leakage vào model.

> Cache `data/cleaned/train_cleaned.feather` giữ phần clean+zero-fill (nặng); feature áp dụng lúc load. Cache mới có manifest cấu hình; cache legacy không có manifest sẽ bị từ chối. Rebuild một lần bằng `python scripts/clean_data.py`.

---

## Cấu trúc dự án

```
demand_forecasting/
├── configs/        # base + features + per-model YAML
├── data/raw/       # *.csv gốc (gitignored)
├── data/cleaned/   # cache feather (gitignored, tạo bởi clean_data.py)
├── src/
│   ├── data/       # cleaner (join + ZERO-FILL + exog), features, preprocessor, loader
│   ├── models/     # BaseModel + ARIMA/SARIMAX/Prophet/XGBoost/LSTM
│   ├── analysis/   # ACF/PACF, stationarity, residual diagnostics
│   └── evaluation/ # NWRMSLE + metrics, walk-forward CV, comparison
├── scripts/        # clean_data, train, evaluate, compare_cv, measure_sparsity, tune_*
├── notebooks/      # 01_eda (EDA trực quan), 02_comparison
├── docs/           # phương pháp (chuong-2), codebase-summary, changelog, roadmap
└── plans/reports/  # báo cáo research + status
```

---

## Yêu cầu, tài nguyên & test

- **Môi trường:** Python 3.11, `uv`; venv tại `/root/.venvs/demand_forecasting`. Reproducible với `seed: 42`.
- **GPU/CPU:** XGBoost & LSTM ưu tiên CUDA nếu có (cấu hình PyTorch cu118). Không có GPU / GPU yếu → `--set model.device=cpu`.
- **Tài nguyên:** `train.csv` ~125M dòng. `clean_data.py` load toàn bộ → lọc cluster 1 + DAIRY trước zero-fill → cache còn **831.649 dòng** toàn dải ngày, trong đó **774.048 dòng train**; file Feather nén khoảng **5,1 MB**. Train sau đó dùng cache → nhanh.
- **Chẩn đoán độ thưa:** `python scripts/measure_sparsity.py` (đo tỉ lệ implicit-zero + ước lượng RAM sau reindex).
- **Chạy test:** `python -m pytest tests/ -q`.

---

## Trạng thái & bước tiếp

✅ **Đã xong (data layer, sẵn sàng baseline tiểu luận):** zero-fill bound first-sale,
fixed pseudo-holdout panel, origin-safe target features, shared forecast-exog contract,
weighted NWRMSLE, cache validation, EDA trực quan tiếng Việt.

⏭ **Bước tiếp (model layer):**
1. Chọn `(p,d,q)(P,D,Q)₇` thật cho ARIMA/SARIMAX qua ACF/PACF (notebook mục 9) — hiện đang placeholder `(1,1,1)`.
2. Train XGBoost/LSTM global full feature; tune hyperparameter.
3. So sánh cuối qua `compare_cv.py`.

⚠ **Giới hạn đã biết (cùng nắm trước):**
- **Validate chưa bật mặc định** — `base.yaml` chỉ có train+test; thêm `val_start/val_end` nếu cần (xem A.2).
- **ARIMA/SARIMAX order là placeholder `(1,1,1)`** → cần tune qua ACF/PACF, metric hiện chưa phản ánh năng lực thật.
- **XGBoost qua `train.py` có thể treo ở bước predict trên GPU yếu** → dùng `--set model.device=cpu`.
- **Pseudo-holdout cold start:** chỉ gồm series đã xuất hiện gần forecast origin; Kaggle test thật có panel chính thức nên cần xử lý cold start riêng nếu làm submission.
- **Oil:** tắt trong benchmark chính; có thể bật cho thí nghiệm phụ nếu dùng forecast hoặc scenario.
- **`docs/` và `plans/` mặc định bị gitignore** — chỉ force-add tài liệu cần chia sẻ.
