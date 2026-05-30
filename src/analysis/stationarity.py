"""Kiểm định tính dừng (stationarity) cho chuỗi thời gian.

Dùng ADF + KPSS kết hợp để xác định d (bậc sai phân):
- ADF: H0 = chuỗi có unit root (không dừng). p < 0.05 → bác bỏ → dừng.
- KPSS: H0 = chuỗi dừng quanh trend. p < 0.05 → bác bỏ → không dừng.
- Cả hai đồng thuận → kết luận chắc chắn hơn, tránh lỗi do 1 test riêng lẻ.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def _run_adf(series: np.ndarray) -> dict:
    """Chạy ADF test, trả dict kết quả gọn."""
    result = adfuller(series, autolag="AIC")
    return {
        "adf_stat": round(result[0], 4),
        "adf_pvalue": round(result[1], 4),
        "adf_lags": result[2],
        # p < 0.05 → bác bỏ H0 (unit root) → chuỗi dừng
        "adf_stationary": result[1] < 0.05,
    }


def _run_kpss(series: np.ndarray) -> dict:
    """Chạy KPSS test (regression='c' cho level stationarity)."""
    # nlags="auto" → tự chọn số lag dựa trên độ dài chuỗi
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, lags, crit = kpss(series, regression="c", nlags="auto")
    return {
        "kpss_stat": round(stat, 4),
        "kpss_pvalue": round(pvalue, 4),
        "kpss_lags": lags,
        # p > 0.05 → không bác bỏ H0 (chuỗi dừng) → kết luận dừng
        "kpss_stationary": pvalue > 0.05,
    }


def _determine_d(adf_stationary: bool, kpss_stationary: bool) -> int:
    """Kết hợp ADF + KPSS để gợi ý d.

    Bảng quyết định:
    | ADF dừng | KPSS dừng | Kết luận        | d |
    |----------|-----------|-----------------|---|
    | Có       | Có        | Dừng            | 0 |
    | Có       | Không     | Dừng quanh trend| 0 |
    | Không    | Có        | Có unit root    | 1 |
    | Không    | Không     | Không dừng      | 1 |
    """
    if adf_stationary:
        return 0
    return 1


def test_stationarity(
    series: np.ndarray,
    store_id: int | None = None,
    max_d: int = 2,
) -> dict:
    """Kiểm định tính dừng cho 1 chuỗi, thử sai phân đến khi dừng hoặc đạt max_d.

    Trả về dict gồm: kết quả test gốc, kết quả sau sai phân, d gợi ý.
    """
    results = {"store_id": store_id, "levels": []}

    current = series.copy()
    suggested_d = 0

    for d in range(max_d + 1):
        if d > 0:
            # sai phân bậc d: loại bỏ trend → kiểm tra lại tính dừng
            current = np.diff(current)

        # chuỗi quá ngắn sau sai phân → dừng
        if len(current) < 20:
            break

        adf = _run_adf(current)
        kpss_res = _run_kpss(current)
        level_result = {
            "d": d,
            **adf,
            **kpss_res,
            "conclusion": "stationary" if adf["adf_stationary"] and kpss_res["kpss_stationary"] else "non-stationary",
        }
        results["levels"].append(level_result)

        # dừng lại khi cả 2 test đồng thuận chuỗi dừng
        if adf["adf_stationary"] and kpss_res["kpss_stationary"]:
            suggested_d = d
            break
        suggested_d = d + 1

    # giới hạn d không vượt max_d
    results["suggested_d"] = min(suggested_d, max_d)
    return results


def stationarity_summary(results_list: list[dict]) -> pd.DataFrame:
    """Tổng hợp kết quả kiểm định từ nhiều store thành DataFrame.

    Mỗi dòng = 1 store, cột gồm: store_id, suggested_d, ADF/KPSS stats ở level gốc (d=0).
    """
    rows = []
    for res in results_list:
        # lấy kết quả ở chuỗi gốc (d=0) để so sánh ngang giữa các store
        level0 = res["levels"][0] if res["levels"] else {}
        rows.append({
            "store_id": res["store_id"],
            "suggested_d": res["suggested_d"],
            "adf_stat": level0.get("adf_stat"),
            "adf_pvalue": level0.get("adf_pvalue"),
            "adf_stationary": level0.get("adf_stationary"),
            "kpss_stat": level0.get("kpss_stat"),
            "kpss_pvalue": level0.get("kpss_pvalue"),
            "kpss_stationary": level0.get("kpss_stationary"),
            "conclusion_d0": level0.get("conclusion"),
        })
    return pd.DataFrame(rows)
