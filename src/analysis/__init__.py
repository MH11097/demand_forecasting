"""Module phân tích chuỗi thời gian theo Box-Jenkins.

Cung cấp: kiểm định tính dừng (ADF/KPSS), biểu đồ ACF/PACF, chẩn đoán phần dư.
"""

from src.analysis.acf_pacf import plot_acf_pacf, suggest_pq_range
from src.analysis.residual_diagnostics import diagnose_residuals
from src.analysis.stationarity import test_stationarity

__all__ = [
    "test_stationarity",
    "plot_acf_pacf",
    "suggest_pq_range",
    "diagnose_residuals",
]
