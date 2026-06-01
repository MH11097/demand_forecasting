# Plan: Improve XGBoost Model Performance từ 1.104 NWRMSLE

## TL;DR
Kết quả XGBoost (1.104 NWRMSLE) thấp hơn LGB (0.506) do thiếu **promotion modeling**, **future promo lookahead**, và **multi-level aggregation**. Kế hoạch là thêm lần lượt: (1) future promo features + conditional promo stats, (2) multi-level aggregation, (3) multi-horizon training architecture, (4) sample weighting + decay features. Ưu tiên từ bước 1, vì LGB chứng minh promo là key driver.

---

## Steps (Phased Approach)

### Phase 1: Add Promotion Features (High Impact, Low Risk)
1. Add `future_promo_lookahead` features
   - Extract onpromotion từ test set (T+1 to T+14) 
   - Tạo: `promo_count_7_future`, `promo_count_14_future`, `promo_days_ahead`
   - Tương tự LGB: `promo_3_aft`, `promo_7_aft`, `promo_14_aft`
   - **Note**: Đây là exog (không leakage) vì test set onpromotion đã biết

2. Add conditional promo statistics
   - Tạo `promo_masked` mask (rolling_rate > 0)
   - Tính `sales_with_promo_mean_7/14/30` (mean sales chỉ ngày có promo)
   - Tính `sales_no_promo_mean_7/14/30` (mean sales chỉ ngày không promo)
   - Tương tự cho rolling_std, rolling_median

3. Update `src/data/features.py`
   - Add `generate_future_promo_features()` function
   - Add `generate_conditional_promo_stats()` function
   - Thêm vào `feature_engineering()` pipeline
   - **Cảnh báo lỗi categorical**: Các feature này là int/float, không phải str

4. Test & Verify
   - Run training trên subset (10 series) để kiểm tra dtype
   - Kiểm tra feature count, NaN không
   - So sánh NWRMSLE trước/sau

### Phase 2: Add Multi-Level Aggregations (Medium Impact)
1. Cấu trúc dữ liệu để tính item-level & family-level aggregates
   - Item-level: tổng bán hàng từ tất cả store (trừ target store để tránh leakage)
   - Family-level: tổng bán hàng từ tất cả item trong family
   - Store-family-level: (store, family) aggregates

2. Tính aggregate features
   - `item_promo_rolling_rate_7`, `item_sales_mean_7`, v.v.
   - `family_promo_rolling_rate_7`, `family_sales_mean_7`, v.v.
   - Tương tự conditional promo stats ở level này

3. Thêm vào feature engineering
   - Cách LGB: reindex aggregates về series level
   - Avoid leakage: fit aggregates trên train set, freeze cho val/test

4. Test
   - Feature count sẽ tăng lên ~100+ (từ 39 hiện tại)
   - Kiểm tra overfitting bằng CV/validation set

### Phase 3: Implement Multi-Horizon Training (Medium-High Impact, High Risk)
**Điều kiện**: Chỉ làm nếu Phase 1+2 không đạt kết quả, vì thay đổi architecture lớn

1. Thay đổi training strategy
   - Thay vì 1 mô hình multioutput, train 90 mô hình riêng (1 mỗi ngày)
   - Mỗi mô hình dự báo `target_day=T+i` thay vì tất cả T+1:T+90

2. Data preparation
   - Pivot data: mỗi hàng = (store, item, date), 1 cột target (sales at T+i)
   - Tạo 90 dataset_i, mỗi cái có target khác nhau

3. Refactor `scripts/train.py`
   - Loop thay vì 1 mô hình: 90 iterations
   - Tương tự LGB: đắt hơn nhưng có thể fit riêng learning rate/early stop per day

4. Verify
   - Training time sẽ gấp ~90x (từ 35 giây → ~50 phút)
   - Kết quả dự báo sẽ smooth hơn (không bị dominated by near-term patterns)

### Phase 4: Add Decay Weighting & Sample Weighting (Low-Medium Impact)
1. Decay weighting trên lags & rolling features
   - Thay vì lag_t bình thường, dùng lag_t * (0.9^t)
   - Áp dụng trên rolling features: rolling_mean * decay_weights

2. Sample weighting by perishability
   - Lấy `perishable` flag từ items metadata
   - Weight = `perishable * 0.25 + 1` (tương tự LGB)
   - Fit XGBoost với `sample_weight=weight`

3. Update training code
   - Modify `src/data/features.py`: add decay_weighting()
   - Modify `scripts/train.py`: add sample_weight logic
   - Modify `src/models/xgboost_model.py`: accept weight parameter

---

## Relevant Files
- `src/models/xgboost_model.py` — Training logic, need to add weight support
- `src/data/features.py` — Feature generation, add promo + aggregation features
- `src/data/preprocessor.py` — Data pipeline, verify zero-fill + leakage-safe aggregates
- `scripts/train.py` — Main training loop, potentially refactor for multi-horizon
- `configs/features.yaml` — Feature config, update with new features
- `notebooks/1st_place_lgb_model_public_0_506_private_0_511.py` — Reference for multi-level agg & conditional promo

---

## Verification

### Phase 1 (Future Promo Features)
- ✓ No dtype errors (int/float only)
- ✓ Features created (promo_count_7_future, etc.)
- ✓ NWRMSLE < 1.104 (expect 0.95-1.00)

### Phase 2 (Multi-Level Aggregations)
- ✓ Item-level aggregates computed (no leakage)
- ✓ Family-level aggregates computed (no leakage)
- ✓ Feature count = 100+, no NaNs
- ✓ NWRMSLE < 0.95 (expect 0.80-0.90)

### Phase 3 (Multi-Horizon Training)
- ✓ 90 models train successfully
- ✓ Early stopping per model works
- ✓ NWRMSLE < 0.80 (expect 0.60-0.70 target)

### Phase 4 (Decay & Weighting)
- ✓ Decay weights applied correctly
- ✓ Sample weights used in training
- ✓ NWRMSLE < 0.70 (expect 0.50-0.55, near LGB)

### Regression Tests
- CV/test NWRMSLE improves each phase
- No overfitting (validate on held-out series)
- Training time acceptable (~5-30 min per run)

---

## Decisions & Scope

### Include
- Future promo lookahead (exog, no leakage)
- Conditional promo statistics (with_promo vs without_promo)
- Multi-level aggregations (item, family level)
- Decay weighting (0.9^t factors)
- Sample weighting by perishability
- Possible multi-horizon training (if needed)

### Exclude (Out of Scope)
- Change to fully different algorithm (e.g., Prophet, LSTM)
- Hyperparameter tuning (can do after features fixed)
- External data (API calls, web scraping)
- Different loss function (stick with MSE for XGBoost native)

---

## Further Clarifications

1. **Future Promo Lookahead Safety**: Test set onpromotion is provided, so using it IS exogenous. LGB uses it → you should too. No leakage risk.

2. **Multi-Horizon Worth It?**: LGB's 16 separate models helped, but could be diminishing returns given 90-day horizon. Do Phase 1-2 first; Phase 3 only if stuck below 0.85.

3. **Dtype Error Handling**: When adding new features, ensure:
   - Never add string columns (except if explicitly categorical)
   - Always cast to int/float or category dtype
   - Test on small dataset first (n_series=10)

4. **Validation Strategy**: Use walk-forward CV or time-series split, not random split, to avoid leakage.
