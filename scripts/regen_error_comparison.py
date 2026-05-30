"""Regenerate results/comparison/error_comparison.png với số liệu CV cập nhật.

Chỉ vẽ lại bar chart, KHÔNG đụng comparison_table.csv/md hay cv_results.json.
Tái sử dụng plot_error_comparison từ scripts/compare_30day.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_30day import plot_error_comparison

# Mean values (stds không dùng trong bar chart). MAPE ở dạng phân số (×100 để hiển
# thị %); NWRMSLE giữ nguyên scale. TODO: thay placeholder bằng số CV Favorita thực tế.
ROWS = [
    {"Model": "ARIMA",   "_mae": 1243.15, "_rmse": 1675.23, "_mape": 0.1891, "_nwrmsle": 0.2500},
    {"Model": "SARIMAX", "_mae":  849.00, "_rmse": 1211.00, "_mape": 0.1172, "_nwrmsle": 0.1592},
    {"Model": "Prophet", "_mae":  860.44, "_rmse": 1232.19, "_mape": 0.1227, "_nwrmsle": 0.1613},
    {"Model": "XGBoost", "_mae":  700.51, "_rmse":  963.87, "_mape": 0.1074, "_nwrmsle": 0.1556},
    {"Model": "LSTM",    "_mae": 1430.50, "_rmse": 2014.80, "_mape": 0.2128, "_nwrmsle": 0.3405},
]


if __name__ == "__main__":
    df = pd.DataFrame(ROWS)
    plot_error_comparison(df)
