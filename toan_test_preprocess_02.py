import pandas as pd
import time
import numpy as np

input_csv = r"dev\train_2017.csv"
output_feather = r"dev\train_2017.feather"

print("1. Đang đọc file CSV (Có thể mất khoảng vài chục giây)...")
start_time = time.time()

# Đọc file CSV
df = pd.read_csv(
    "dev/train_2017.csv",
    usecols=[1, 2, 3, 4, 5],
    dtype={"onpromotion": bool},
    converters={"unit_sales": lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    # skiprows=range(1, 66458909),  # 2016-01-01
)

print("2. Đang tối ưu hóa kiểu dữ liệu...")
# Ép kiểu cột 'date' về datetime ngay lúc này.
# Feather sẽ lưu cứng định dạng này, giúp file nhẹ hơn và đọc nhanh hơn sau này.
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# (Tùy chọn) Ép kiểu các cột số nguyên để giảm dung lượng RAM
if "store_nbr" in df.columns:
    df["store_nbr"] = df["store_nbr"].astype("int16")
if "item_nbr" in df.columns:
    df["item_nbr"] = df["item_nbr"].astype("int32")

print("3. Đang lưu sang định dạng Feather...")
# Lưu toàn bộ dataframe thành file feather
df.to_feather(output_feather)

end_time = time.time()
print(f"Hoàn tất! Đã tạo file {output_feather} trong {end_time - start_time:.2f} giây.")
