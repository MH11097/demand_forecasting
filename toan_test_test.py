import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from datetime import date, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==============================================================================
# PHẦN 1: CÁC HÀM ĐÁNH GIÁ (METRICS)
# ==============================================================================
def calculate_nwrmsle(y_true, y_pred, weights):
    """Tính chỉ số NWRMSLE chuẩn của cuộc thi Kaggle Favorita"""
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    squared_log_error = (log_pred - log_true) ** 2
    nwrmsle = np.sqrt(np.average(squared_log_error, weights=weights))
    return nwrmsle


def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """Tính MAPE (có xử lý tránh lỗi chia cho 0)"""
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape


def evaluate_model(y_true, y_pred, weights):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    nwrmsle = calculate_nwrmsle(y_true, y_pred, weights)

    print("\n" + "=" * 40)
    print(" 📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ TẬP TEST")
    print("=" * 40)
    print(f" NWRMSLE (Kaggle) : {nwrmsle:.4f}")
    print(f" RMSE             : {rmse:.4f}")
    print(f" MAE              : {mae:.4f}")
    print(f" MAPE             : {mape:.2f} %")
    print("=" * 40)
    return nwrmsle


# ==============================================================================
# PHẦN 2: CHUẨN BỊ DỮ LIỆU TEST (GIỮ NGUYÊN LOGIC NHƯ LÚC TRAIN)
# ==============================================================================
print("1. Đang tải dữ liệu từ file feather...")
df_all = pd.read_feather(r"data\cleaned\train_cleaned.feather")

if not pd.api.types.is_datetime64_any_dtype(df_all["date"]):
    df_all["date"] = pd.to_datetime(df_all["date"])

df_2017 = df_all.loc[df_all["date"] >= pd.Timestamp(2017, 1, 1)].copy()

if "perishable" in df_2017.columns:
    items = df_2017[["item_nbr", "perishable"]].drop_duplicates().set_index("item_nbr")
else:
    items = pd.DataFrame(index=df_2017["item_nbr"].unique())
    items["perishable"] = 1

max_date = df_2017["date"].max().date()
t_test = max_date - timedelta(days=15)  # Lấy 16 ngày cuối cùng làm Test

print("2. Đang tạo Feature Engineering cho tập Test...")
promo_2017 = (
    df_2017.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]]
    .unstack(level=-1)
    .fillna(False)
)
promo_2017.columns = promo_2017.columns.get_level_values(1)

df_2017_sales = (
    df_2017.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]]
    .unstack(level=-1)
    .fillna(0)
)
df_2017_sales.columns = df_2017_sales.columns.get_level_values(1)

items = items.reindex(df_2017_sales.index.get_level_values(1)).fillna(1)


def get_timespan(df, dt, minus, periods, freq="D"):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]


# Chỉ tạo dữ liệu riêng cho tập Test
X = pd.DataFrame(
    {
        "day_1_2017": get_timespan(df_2017_sales, t_test, 1, 1).values.ravel(),
        "mean_3_2017": get_timespan(df_2017_sales, t_test, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017_sales, t_test, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017_sales, t_test, 14, 14).mean(axis=1).values,
        "mean_30_2017": get_timespan(df_2017_sales, t_test, 30, 30).mean(axis=1).values,
        "mean_60_2017": get_timespan(df_2017_sales, t_test, 60, 60).mean(axis=1).values,
        "mean_140_2017": get_timespan(df_2017_sales, t_test, 140, 140)
        .mean(axis=1)
        .values,
        "promo_14_2017": get_timespan(promo_2017, t_test, 14, 14)
        .sum(axis=1)
        .values.astype(float),
        "promo_60_2017": get_timespan(promo_2017, t_test, 60, 60)
        .sum(axis=1)
        .values.astype(float),
        "promo_140_2017": get_timespan(promo_2017, t_test, 140, 140)
        .sum(axis=1)
        .values.astype(float),
    }
)

for i in range(7):
    X["mean_4_dow{}_2017".format(i)] = (
        get_timespan(df_2017_sales, t_test, 28 - i, 4, freq="7D").mean(axis=1).values
    )
    X["mean_20_dow{}_2017".format(i)] = (
        get_timespan(df_2017_sales, t_test, 140 - i, 20, freq="7D").mean(axis=1).values
    )

for i in range(16):
    if (t_test + timedelta(days=i)) in promo_2017.columns:
        X["promo_{}".format(i)] = promo_2017[t_test + timedelta(days=i)].values.astype(
            np.uint8
        )
    else:
        X["promo_{}".format(i)] = 0

X_test = X
y_test_log = df_2017_sales[pd.date_range(t_test, periods=16)].values

# ==============================================================================
# PHẦN 3: ĐỌC MÔ HÌNH VÀ DỰ BÁO
# ==============================================================================
print("3. Đang nạp mô hình từ file xgboost_16days_models.pkl...")
with open("xgboost_16days_models.pkl", "rb") as f:
    models = pickle.load(f)

print("4. Đang chạy dự báo trên tập Test...")
dtest = xgb.DMatrix(X_test)
test_pred_log = []

# Dự báo lần lượt cho 16 ngày bằng 16 mô hình
for i in range(16):
    test_pred_log.append(models[i].predict(dtest))

# Chuyển đổi list (16 x num_items) thành ma trận (num_items x 16)
test_pred_log = np.array(test_pred_log).transpose()

# ==============================================================================
# PHẦN 4: CHUYỂN ĐỔI VỀ SCALE GỐC VÀ ĐÁNH GIÁ NWRMSLE
# ==============================================================================
# Do dữ liệu train trước đó là dạng log(x+1), ta phải dùng exp(x)-1 để trả về số lượng thật
y_test_original = np.expm1(y_test_log)
test_pred_original = np.expm1(test_pred_log)

# Lấy mảng trọng số cho các items (dễ hỏng = 1.25, thường = 1.0)
item_weights = items["perishable"].values * 0.25 + 1.0

# Nhân bản trọng số 16 lần để khớp với shape của Y (num_items, 16)
weights_matrix = np.tile(item_weights.reshape(-1, 1), (1, 16))

# Làm phẳng (flatten) tất cả thành mảng 1D để đưa vào hàm tính toán
print("5. Đang tính toán các chỉ số...")
evaluate_model(
    y_true=y_test_original.flatten(),
    y_pred=test_pred_original.flatten(),
    weights=weights_matrix.flatten(),
)
