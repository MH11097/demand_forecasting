"""Chẩn đoán phần dư (residual diagnostics) cho ARIMA/SARIMAX.

Mục đích: kiểm tra phần dư có phải white noise không.
- Nếu white noise → model đã capture hết signal, adequate.
- Nếu còn pattern → model bỏ sót thông tin, cần điều chỉnh p/q/d.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


def ljung_box_test(residuals: np.ndarray, lags: list[int] | None = None) -> dict:
    """Chạy Ljung-Box test trên phần dư.

    H0: phần dư là white noise (không có autocorrelation).
    p > 0.05 → không bác bỏ → phần dư OK → model adequate.
    p < 0.05 → bác bỏ → phần dư có pattern → model chưa tốt.
    """
    if lags is None:
        # test ở nhiều lag → phát hiện cả short-term và long-term correlation
        lags = [5, 10, 15, 20]

    # lọc lags hợp lệ: phải nhỏ hơn độ dài chuỗi
    valid_lags = [lag for lag in lags if lag < len(residuals)]
    if not valid_lags:
        return {"error": "Chuỗi phần dư quá ngắn để test"}

    result = acorr_ljungbox(residuals, lags=valid_lags, return_df=True)
    rows = []
    for lag in valid_lags:
        row = result.loc[lag]
        rows.append({
            "lag": lag,
            "lb_stat": round(row["lb_stat"], 4),
            "lb_pvalue": round(row["lb_pvalue"], 4),
            # p > 0.05 → white noise → OK
            "is_white_noise": row["lb_pvalue"] > 0.05,
        })
    return {
        "tests": rows,
        # tất cả lag đều white noise → model adequate
        "overall_adequate": all(r["is_white_noise"] for r in rows),
    }


def plot_residual_acf(
    residuals: np.ndarray,
    nlags: int = 30,
    title: str = "Residual ACF",
    save_path: str | None = None,
) -> plt.Figure:
    """Vẽ ACF của phần dư — kiểm tra trực quan white noise.

    Phần dư tốt: tất cả lag nằm trong confidence band (vùng xanh).
    Có lag ngoài band → model chưa capture hết autocorrelation.
    """
    nlags = min(nlags, len(residuals) // 2 - 1)
    acf_vals, ci = acf(residuals, nlags=nlags, alpha=0.05)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(nlags + 1), acf_vals, width=0.3, color="steelblue", alpha=0.8)
    lower = ci[:, 0] - acf_vals
    upper = ci[:, 1] - acf_vals
    ax.fill_between(range(nlags + 1), lower, upper, alpha=0.15, color="blue")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def diagnose_residuals(
    residuals: np.ndarray,
    store_id: int | None = None,
    save_dir: str | None = None,
) -> dict:
    """Chạy toàn bộ chẩn đoán phần dư cho 1 store.

    Gộp Ljung-Box + ACF plot → trả dict kết quả + lưu biểu đồ nếu có save_dir.
    """
    lb_result = ljung_box_test(residuals)

    save_path = None
    if save_dir:
        suffix = f"_store{store_id}" if store_id else ""
        save_path = str(Path(save_dir) / f"residual_acf{suffix}.png")

    plot_residual_acf(
        residuals,
        title=f"Residual ACF - Store {store_id}" if store_id else "Residual ACF",
        save_path=save_path,
    )

    return {
        "store_id": store_id,
        "ljung_box": lb_result,
        "n_residuals": len(residuals),
        "mean": round(float(np.mean(residuals)), 4),
        "std": round(float(np.std(residuals)), 4),
    }
