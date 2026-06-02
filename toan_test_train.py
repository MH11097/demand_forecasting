"""
Mô hình XGBoost dự báo unit_sales trong 16 ngày.
Tích hợp: Tự động chia Train/Val/Test từ file duy nhất, tránh Data Leakage, lưu mô hình.
"""

from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle

print("1. Đang tải dữ liệu từ file feather...")
# Đọc toàn bộ dữ liệu
df_all = pd.read_feather(r"data\cleaned\train_cleaned.feather")

# Đảm bảo định dạng thời gian
if not pd.api.types.is_datetime64_any_dtype(df_all["date"]):
    df_all["date"] = pd.to_datetime(df_all["date"])

# Lọc lấy dữ liệu từ năm 2017 để train cho nhẹ và sát xu hướng
df_2017 = df_all.loc[df_all["date"] >= pd.Timestamp(2017, 1, 1)].copy()

# Giả định: Vì bạn lọc family = DAIRY, hầu hết đều là hàng dễ hỏng (perishable = 1)
# Ta tạo bảng items trực tiếp từ dữ liệu để không cần file items.csv nữa
if "perishable" in df_2017.columns:
    items = df_2017[["item_nbr", "perishable"]].drop_duplicates().set_index("item_nbr")
else:
    items = pd.DataFrame(index=df_2017["item_nbr"].unique())
    items["perishable"] = 1  # Mặc định DAIRY là 1

print("2. Tự động tính toán mốc thời gian chia Train/Val/Test...")
# Tính toán ngày kết thúc của dữ liệu
max_date = df_2017["date"].max().date()
print(f"   -> Ngày cuối cùng trong dữ liệu: {max_date}")

# Tập Test lấy 16 ngày cuối cùng
t_test = max_date - timedelta(days=15)
# Tập Val lùi lại 21 ngày so với Test (giống tỷ lệ kịch bản gốc Kaggle)
t_val = t_test - timedelta(days=21)
# Tập Train lấy lùi lại 6 tuần trước Val
t_train_start = t_val - timedelta(days=7 * 5)

print(f"   -> Mốc bắt đầu Train: {t_train_start}")
print(f"   -> Mốc bắt đầu Validation: {t_val}")
print(f"   -> Mốc bắt đầu Test: {t_test}")

print("3. Đang tiền xử lý features (Pivot tables)...")
# Tiền xử lý Promotion
promo_2017 = (
    df_2017.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]]
    .unstack(level=-1)
    .fillna(False)
)
promo_2017.columns = promo_2017.columns.get_level_values(1)

# Tiền xử lý Unit Sales
df_2017_sales = (
    df_2017.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]]
    .unstack(level=-1)
    .fillna(0)
)
df_2017_sales.columns = df_2017_sales.columns.get_level_values(1)

items = items.reindex(df_2017_sales.index.get_level_values(1)).fillna(1)


# Hàm lấy khung thời gian
def get_timespan(df, dt, minus, periods, freq="D"):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]


# Hàm tạo Dataset
def prepare_dataset(t_date):
    X = pd.DataFrame(
        {
            "day_1_2017": get_timespan(df_2017_sales, t_date, 1, 1).values.ravel(),
            "mean_3_2017": get_timespan(df_2017_sales, t_date, 3, 3)
            .mean(axis=1)
            .values,
            "mean_7_2017": get_timespan(df_2017_sales, t_date, 7, 7)
            .mean(axis=1)
            .values,
            "mean_14_2017": get_timespan(df_2017_sales, t_date, 14, 14)
            .mean(axis=1)
            .values,
            "mean_30_2017": get_timespan(df_2017_sales, t_date, 30, 30)
            .mean(axis=1)
            .values,
            "mean_60_2017": get_timespan(df_2017_sales, t_date, 60, 60)
            .mean(axis=1)
            .values,
            "mean_140_2017": get_timespan(df_2017_sales, t_date, 140, 140)
            .mean(axis=1)
            .values,
            "promo_14_2017": get_timespan(promo_2017, t_date, 14, 14)
            .sum(axis=1)
            .values.astype(float),
            "promo_60_2017": get_timespan(promo_2017, t_date, 60, 60)
            .sum(axis=1)
            .values.astype(float),
            "promo_140_2017": get_timespan(promo_2017, t_date, 140, 140)
            .sum(axis=1)
            .values.astype(float),
        }
    )

    for i in range(7):
        X["mean_4_dow{}_2017".format(i)] = (
            get_timespan(df_2017_sales, t_date, 28 - i, 4, freq="7D")
            .mean(axis=1)
            .values
        )
        X["mean_20_dow{}_2017".format(i)] = (
            get_timespan(df_2017_sales, t_date, 140 - i, 20, freq="7D")
            .mean(axis=1)
            .values
        )

    for i in range(16):
        # Lấy thông tin khuyến mãi cho 16 ngày tương lai
        if (t_date + timedelta(days=i)) in promo_2017.columns:
            X["promo_{}".format(i)] = promo_2017[
                t_date + timedelta(days=i)
            ].values.astype(np.uint8)
        else:
            X["promo_{}".format(i)] = 0  # Điền 0 nếu tương lai chưa có data promo

    # Lấy nhãn thực tế (Y) cho 16 ngày tiếp theo để Train/Test
    # Vì Test lần này ta có đáp án thật, ta sẽ trả về cả Y cho tập Test
    try:
        y = df_2017_sales[pd.date_range(t_date, periods=16)].values
    except KeyError:
        # Nếu không đủ 16 ngày thật (chỉ rủi ro ở tương lai xa), trả về Y rỗng
        y = None

    return X, y


print("4. Đang xây dựng Dữ liệu Train, Val, Test...")
# Tạo tập Train
X_l, y_l = [], []
for i in range(6):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(t_train_start + delta)
    X_l.append(X_tmp)
    y_l.append(y_tmp)

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l

# Tạo tập Val và Test
X_val, y_val = prepare_dataset(t_val)
X_test, y_test = prepare_dataset(t_test)

print("5. Bắt đầu huấn luyện 16 mô hình XGBoost...")
param = {
    "objective": "reg:squarederror",
    "eta": 0.1,
    "max_depth": 3,
    "eval_metric": "rmse",
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "seed": 137,
}
num_rounds = 157
plst = list(param.items())

val_pred = []
test_pred = []
models = []

dtest = xgb.DMatrix(X_test)

for i in range(16):
    print(f"--- Đang huấn luyện dự báo cho ngày thứ {i+1}/16 ---")
    # Nhân bản mảng trọng số lên 6 lần cho khớp với X_train (bằng np.tile)
    weight_train = np.tile(items["perishable"].values, 6) * 0.25 + 1
    weight_val = items["perishable"].values * 0.25 + 1

    dtrain = xgb.DMatrix(X_train, label=y_train[:, i], weight=weight_train)
    dval = xgb.DMatrix(X_val, label=y_val[:, i], weight=weight_val)

    watchlist = [(dtrain, "train"), (dval, "val")]

    # Huấn luyện mô hình cho ngày thứ i
    model = xgb.train(
        plst,
        dtrain,
        num_rounds,
        watchlist,
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    # Dự báo
    val_pred.append(model.predict(dval))
    test_pred.append(model.predict(dtest))
    models.append(model)

print("\n6. Đánh giá sai số MSE...")
print("-> Validation MSE:", mean_squared_error(y_val, np.array(val_pred).transpose()))
if y_test is not None:
    print(
        "-> Test MSE (Thực tế):",
        mean_squared_error(y_test, np.array(test_pred).transpose()),
    )

print("\n7. Đang lưu mô hình vào file 'xgboost_16days_models.pkl'...")
with open("xgboost_16days_models.pkl", "wb") as f:
    pickle.dump(models, f)
print("Hoàn tất mọi quá trình!")
