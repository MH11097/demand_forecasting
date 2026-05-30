"""Visualization cho tuning results — tạo biểu đồ phục vụ viết tiểu luận.

Reuse pattern từ src/utils/visualization.py: fig/ax, dpi=150, plt.close sau khi lưu.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_param_sensitivity(
    results_df: pd.DataFrame,
    param_name: str,
    metric: str = "nwrmsle",
    save_path: str | None = None,
) -> plt.Figure:
    """Line plot: 1 tham số vs NWRMSLE (mean ± std qua các tổ hợp khác).

    Mỗi giá trị của param_name → tính trung bình NWRMSLE trên tất cả tổ hợp còn lại
    → thấy rõ tham số nào ảnh hưởng nhiều, vùng giá trị nào tốt.
    """
    if param_name not in results_df.columns:
        return None

    # Group theo giá trị param → mean/std metric qua tất cả tổ hợp khác
    grouped = results_df.groupby(param_name)[metric].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals = range(len(grouped))
    ax.errorbar(x_vals, grouped["mean"], yerr=grouped["std"], marker="o", capsize=4, linewidth=2, markersize=6)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in grouped[param_name]], rotation=45, ha="right")
    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Sensitivity: {param_name} vs {metric.upper()}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_tuning_heatmap(
    results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "nwrmsle",
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap 2D tương tác giữa 2 tham số → thấy vùng tham số tối ưu.

    Tính mean metric cho mỗi cặp (param_x, param_y) qua các tổ hợp tham số còn lại.
    """
    if param_x not in results_df.columns or param_y not in results_df.columns:
        return None

    # Pivot: hàng = param_y, cột = param_x, giá trị = mean metric
    pivot = results_df.groupby([param_y, param_x])[metric].mean().reset_index()
    pivot_table = pivot.pivot(index=param_y, columns=param_x, values=metric)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlOrRd_r", ax=ax, linewidths=0.5)
    ax.set_title(f"{metric.upper()}: {param_x} vs {param_y}")
    ax.set_xlabel(param_x.replace("_", " ").title())
    ax.set_ylabel(param_y.replace("_", " ").title())
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_top_k_comparison(
    results_df: pd.DataFrame,
    k: int = 10,
    metric: str = "nwrmsle",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart top-K tổ hợp tốt nhất → so sánh trực quan cấu hình tối ưu."""
    # Lấy top-K theo metric thấp nhất
    top_df = results_df.nsmallest(k, metric).reset_index(drop=True)

    # Tạo nhãn ngắn gọn từ tham số (bỏ cột metric/time)
    param_cols = [c for c in top_df.columns if c not in ["nwrmsle", "rmse", "mae", "mape", "time_seconds"]]
    labels = []
    for _, row in top_df.iterrows():
        parts = [f"{c}={row[c]}" for c in param_cols]
        labels.append("\n".join(parts))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(k), top_df[metric], color=sns.color_palette("viridis", k))
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, fontsize=7, ha="center")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Top {k} cấu hình tốt nhất ({metric.upper()})")

    # Ghi giá trị metric trên mỗi cột
    for bar, val in zip(bars, top_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_ablation_results(
    ablation_df: pd.DataFrame,
    metric: str = "nwrmsle",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart so sánh NWRMSLE theo từng tổ hợp regressors.

    Highlight đóng góp biên: baseline vs từng biến đơn lẻ vs full model.
    Thứ tự giữ nguyên theo thí nghiệm (không sort) để thấy tiến trình thêm biến.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(ablation_df)
    colors = sns.color_palette("Set2", n)
    bars = ax.bar(range(n), ablation_df[metric], color=colors)
    ax.set_xticks(range(n))
    ax.set_xticklabels(ablation_df["experiment"], rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Ablation Study: đóng góp từng regressor ({metric.upper()})")

    # Ghi giá trị + đường baseline tham chiếu
    baseline_val = ablation_df[metric].iloc[0] if len(ablation_df) > 0 else 0
    ax.axhline(y=baseline_val, color="gray", linestyle="--", alpha=0.5, label=f"Baseline: {baseline_val:.4f}")

    for bar, val in zip(bars, ablation_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def generate_tuning_report(results_df: pd.DataFrame, top_k: int = 10, model_name: str = "Prophet") -> str:
    """Tạo bảng Markdown tổng hợp kết quả tuning → copy vào tiểu luận.

    Gồm: summary thống kê + bảng top-K chi tiết. Param model_name để reuse cho nhiều model.
    """
    lines = [f"# Kết quả Tuning {model_name}", ""]

    # Thống kê tổng quan
    lines.append("## Tổng quan")
    lines.append(f"- Tổng số tổ hợp: **{len(results_df)}**")
    best = results_df.iloc[0] if len(results_df) > 0 else None
    if best is not None:
        lines.append(f"- NWRMSLE tốt nhất: **{best['nwrmsle']:.6f}**")
        lines.append(f"- NWRMSLE trung bình: **{results_df['nwrmsle'].mean():.6f}**")
        lines.append(f"- NWRMSLE kém nhất: **{results_df['nwrmsle'].max():.6f}**")
    lines.append("")

    # Top-K chi tiết
    top_df = results_df.head(top_k)
    lines.append(f"## Top {top_k} cấu hình tốt nhất")
    lines.append("")

    # Header
    cols = list(top_df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in top_df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    return "\n".join(lines)
