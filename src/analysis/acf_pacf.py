"""Vẽ ACF/PACF và gợi ý vùng p, q cho ARIMA/SARIMAX.

Quy tắc đọc biểu đồ (Box-Jenkins):
- ACF tắt nhanh sau lag q, PACF tắt dần → MA(q) dominant → tập trung chọn q.
- PACF tắt nhanh sau lag p, ACF tắt dần → AR(p) dominant → tập trung chọn p.
- Cả hai tắt dần → ARMA(p,q) hỗn hợp → cần grid search.
- Lag mùa vụ (7, 14, 21...) có spike → cần seasonal component (SARIMAX).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf


def plot_acf_pacf(
    series: np.ndarray,
    nlags: int = 40,
    title: str = "ACF/PACF",
    save_path: str | None = None,
    alpha: float = 0.05,
) -> plt.Figure:
    """Vẽ ACF và PACF cạnh nhau, kèm confidence interval.

    series đã được sai phân d lần trước khi truyền vào → biểu đồ phản ánh
    chuỗi dừng, giúp đọc pattern AR/MA chính xác hơn.
    """
    nlags = min(nlags, len(series) // 2 - 1)

    acf_vals, acf_ci = acf(series, nlags=nlags, alpha=alpha)
    pacf_vals, pacf_ci = pacf(series, nlags=nlags, alpha=alpha, method="ywm")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- ACF ---
    axes[0].bar(range(nlags + 1), acf_vals, width=0.3, color="steelblue", alpha=0.8)
    # confidence band: vùng ngoài band → lag có ý nghĩa thống kê
    lower = acf_ci[:, 0] - acf_vals
    upper = acf_ci[:, 1] - acf_vals
    axes[0].fill_between(range(nlags + 1), lower, upper, alpha=0.15, color="blue")
    axes[0].axhline(y=0, color="black", linewidth=0.5)
    axes[0].set_title(f"{title} - ACF")
    axes[0].set_xlabel("Lag")

    # --- PACF ---
    axes[1].bar(range(nlags + 1), pacf_vals, width=0.3, color="darkorange", alpha=0.8)
    lower_p = pacf_ci[:, 0] - pacf_vals
    upper_p = pacf_ci[:, 1] - pacf_vals
    axes[1].fill_between(range(nlags + 1), lower_p, upper_p, alpha=0.15, color="orange")
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_title(f"{title} - PACF")
    axes[1].set_xlabel("Lag")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def suggest_pq_range(
    series: np.ndarray,
    nlags: int = 40,
    alpha: float = 0.05,
) -> dict:
    """Phân tích ACF/PACF để gợi ý vùng p, q.

    Heuristic:
    1. Đếm số lag liên tiếp vượt confidence interval (significant lags).
    2. significant PACF lags → gợi ý max p. significant ACF lags → gợi ý max q.
    3. Kiểm tra spike tại lag 7, 14, 21 → gợi ý seasonal period.

    Kết quả chỉ là gợi ý ban đầu — user đọc biểu đồ để quyết định cuối cùng.
    """
    nlags = min(nlags, len(series) // 2 - 1)

    acf_vals, acf_ci = acf(series, nlags=nlags, alpha=alpha)
    pacf_vals, pacf_ci = pacf(series, nlags=nlags, alpha=alpha, method="ywm")

    # lag vượt confidence interval → có ý nghĩa thống kê
    acf_sig = _significant_lags(acf_vals, acf_ci)
    pacf_sig = _significant_lags(pacf_vals, pacf_ci)

    # tìm lag significant cao nhất trong 8 lag đầu → vùng tham số đáng khảo sát
    # dùng max significant lag thay vì first cutoff vì ARMA hỗn hợp thường
    # có lag rải rác (không liên tiếp), first cutoff sẽ bỏ sót
    acf_max_sig = _max_significant_lag(acf_sig, max_lag=8)
    pacf_max_sig = _max_significant_lag(pacf_sig, max_lag=8)

    # gợi ý vùng search: 0 → max_sig + 1 (buffer), tối thiểu 3 để grid search có ý nghĩa
    suggested_q_max = max(min(acf_max_sig + 1, 6), 3)
    suggested_p_max = max(min(pacf_max_sig + 1, 6), 3)

    # phát hiện seasonal spike tại bội số của 7 (calendar week) và 6 (trading week — data lọc ngày đóng cửa)
    seasonal_lags_7 = _detect_seasonal_spikes(acf_vals, acf_ci, period=7, nlags=nlags)
    seasonal_lags_6 = _detect_seasonal_spikes(acf_vals, acf_ci, period=6, nlags=nlags)

    return {
        "acf_significant_lags": acf_sig,
        "pacf_significant_lags": pacf_sig,
        "acf_max_sig_lag": acf_max_sig,
        "pacf_max_sig_lag": pacf_max_sig,
        "suggested_p_range": list(range(suggested_p_max + 1)),
        "suggested_q_range": list(range(suggested_q_max + 1)),
        "seasonal_spikes_at_7": seasonal_lags_7,
        "seasonal_spikes_at_6": seasonal_lags_6,
        "has_weekly_pattern": len(seasonal_lags_7) > 0 or len(seasonal_lags_6) > 0,
    }


def _significant_lags(vals: np.ndarray, ci: np.ndarray) -> list[int]:
    """Trả về list lag có giá trị significant (CI không chứa 0, bỏ lag 0).

    statsmodels trả CI bao quanh giá trị ACF/PACF → kiểm tra CI có chứa 0 không,
    nếu cả lower lẫn upper cùng dấu → lag đó significant.
    """
    sig = []
    for i in range(1, len(vals)):
        # CI không chứa 0 → significant (cả 2 bound cùng dương hoặc cùng âm)
        if ci[i, 0] > 0 or ci[i, 1] < 0:
            sig.append(i)
    return sig


def _max_significant_lag(sig_lags: list[int], max_lag: int = 8) -> int:
    """Tìm lag significant cao nhất trong phạm vi max_lag.

    Dùng max thay vì first-cutoff vì chuỗi ARMA hỗn hợp thường có lag
    rải rác (1, 3, 7...) → first-cutoff ở lag 2 sẽ bỏ sót lag 3 và 7.
    Trả 0 nếu không có lag nào significant.
    """
    filtered = [lag for lag in sig_lags if lag <= max_lag]
    return max(filtered) if filtered else 0


def _detect_seasonal_spikes(
    acf_vals: np.ndarray,
    acf_ci: np.ndarray,
    period: int = 7,
    nlags: int = 40,
) -> list[int]:
    """Kiểm tra spike tại bội số của period (7, 14, 21...).

    Spike tại lag mùa vụ → chuỗi có seasonal pattern → nên dùng SARIMAX thay ARIMA.
    """
    spikes = []
    for k in range(1, nlags // period + 1):
        lag = k * period
        if lag < len(acf_vals):
            # CI không chứa 0 → seasonal spike significant
            if acf_ci[lag, 0] > 0 or acf_ci[lag, 1] < 0:
                spikes.append(lag)
    return spikes
