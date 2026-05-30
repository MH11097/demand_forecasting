"""Visualization utilities for time series forecasting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    dates: pd.Series | None = None,
    save_path: str | None = None,
):
    """Plot actual vs predicted values."""
    # vẽ actual vs predicted trên cùng trục -> trực quan thấy model dự đoán sát hay lệch
    fig, ax = plt.subplots(figsize=(14, 5))
    x = dates if dates is not None else range(len(y_true))
    ax.plot(x, y_true, label="Actual", alpha=0.8)
    ax.plot(x, y_pred, label="Predicted", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date" if dates is not None else "Index")
    ax.set_ylabel("Unit Sales")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_predictions_zoomed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted (Zoomed)",
    dates: pd.Series | None = None,
    window_size: int = 60,
    n_windows: int = 4,
    save_path: str | None = None,
):
    """Vẽ nhiều cửa sổ ngẫu nhiên (độ dài window_size) để zoom vào chi tiết.

    Do val set có hàng trăm ngàn điểm, chart toàn bộ quá dày để đọc.
    Hàm này chọn n_windows đoạn rải đều trong cột dư liệu và vẽ riêng từng đoạn.
    """
    n = len(y_true)
    if n <= window_size:
        # quá ít điểm -> fallback vẽ toàn bộ
        return plot_predictions(y_true, y_pred, title=title, dates=dates, save_path=save_path)

    # chọn các start index rải đều thoáng (không cụm vào đầu hoặc cuối)
    rng = np.random.default_rng(seed=42)
    margin = window_size
    safe_range = n - 2 * margin
    if safe_range < n_windows:
        starts = np.linspace(0, n - window_size - 1, n_windows, dtype=int)
    else:
        starts = sorted(rng.choice(np.arange(margin, n - window_size - margin), size=n_windows, replace=False))

    ncols = 2
    nrows = (n_windows + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()
    x_arr = np.array(dates) if dates is not None else np.arange(n)

    for i, start in enumerate(starts):
        end = start + window_size
        ax = axes[i]
        ax.plot(x_arr[start:end], y_true[start:end], label="Actual", linewidth=1.5)
        ax.plot(x_arr[start:end], y_pred[start:end], label="Predicted", linewidth=1.5, linestyle="--")
        ax.set_title(f"Window {i + 1}  (idx {start}–{end - 1})", fontsize=9)
        ax.set_ylabel("Unit Sales")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if dates is not None:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    # ẩn trục dư (nếu n_windows lẻ)
    for j in range(len(starts), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_multi_horizon(
    y_true: np.ndarray,
    y_pred_all: np.ndarray,
    title: str = "Forecast Fan",
    dates: pd.Series | None = None,
    horizons_to_show: list[int] | None = None,
    n_origins: int = 5,
    store_ids: np.ndarray | None = None,
    open_flags: np.ndarray | None = None,
    save_path: str | None = None,
):
    """Visualize forecast fan: từ N điểm gốc (origins), vẽ dự báo T+1..T+H.

    Args:
        y_true:          Sales thực shape (n,)
        y_pred_all:      output của predict_all_horizons(), shape (n, H)
        horizons_to_show: list các bước muốn highlight, mặc định [1, H//4, H//2, H]
        n_origins:       số điểm gốc rải đều dùng để vẽ fan
        dates:           index ngày tháng (optional)
        store_ids:       Store ID per row — dùng để tránh chọn origin vắt qua ranh giới store
        open_flags:      Open flag (0/1) per row — ưu tiên origin ngày mở cửa, forecast window có bán hàng
    """
    n, H = y_pred_all.shape
    if horizons_to_show is None:
        horizons_to_show = sorted({1, H // 4, H // 2, H})
        horizons_to_show = [h for h in horizons_to_show if 1 <= h <= H]

    # ── Chọn origins thông minh ──────────────────────────────────────────────
    margin = H + 10
    safe = list(range(margin, n - H - 5))

    if store_ids is not None:
        # Loại bỏ origins mà cửa sổ forecast vắt qua ranh giới store
        safe = [i for i in safe if store_ids[i] == store_ids[min(i + H, n - 1)]]

    if open_flags is not None:
        # Ưu tiên origins: ngày mở cửa + forecast window có ít nhất H//3 ngày open
        scored = [
            (i, int(open_flags[i] == 1) * 2 + int(np.sum(open_flags[i + 1 : i + H + 1]) >= H // 3))
            for i in safe
        ]
        best_safe = [i for i, s in scored if s == 3]   # origin open + enough open future
        if len(best_safe) >= n_origins:
            safe = best_safe
        else:
            # fallback: chỉ cần origin open
            safe = [i for i, s in scored if s >= 2] or safe

    if len(safe) < n_origins:
        origins = list(range(0, n, max(1, n // n_origins)))[:n_origins]
    else:
        step = max(1, len(safe) // n_origins)
        origins = [safe[i * step] for i in range(n_origins)]

    x_arr = np.array(dates) if dates is not None else np.arange(n)
    cmap = plt.cm.get_cmap("autumn_r", len(horizons_to_show))

    fig, axes = plt.subplots(n_origins, 1, figsize=(14, 3.5 * n_origins), sharex=False)
    if n_origins == 1:
        axes = [axes]

    for ax_i, orig in enumerate(origins):
        ax = axes[ax_i]

        # ── Vẽ actual line (context + forecast window) ───────────────────────
        ctx_start = max(0, orig - H)
        # giới hạn trong cùng store nếu có store_ids
        if store_ids is not None:
            store_end = orig + H
            while store_end > orig and store_end < n and store_ids[store_end] != store_ids[orig]:
                store_end -= 1
            ctx_end = min(n, store_end + 1)
        else:
            ctx_end = min(n, orig + H + 1)

        ax.plot(x_arr[ctx_start:ctx_end], y_true[ctx_start:ctx_end],
                color="steelblue", linewidth=1.8, label="Actual", zorder=3)

        # Open=0 days → đánh dấu bằng x màu xám trên actual line
        if open_flags is not None:
            closed_mask = np.where((open_flags[ctx_start:ctx_end] == 0))[0]
            if len(closed_mask) > 0:
                ax.scatter(x_arr[ctx_start + closed_mask], y_true[ctx_start + closed_mask],
                           marker="x", color="gray", s=20, zorder=4,
                           label="Closed (Open=0)" if ax_i == 0 else "")

        # ── Đánh dấu origin ──────────────────────────────────────────────────
        ax.axvline(x=x_arr[orig], color="black", linestyle=":", linewidth=1.2, alpha=0.7)
        origin_open = open_flags[orig] == 1 if open_flags is not None else True
        ax.annotate(
            f"origin" + (" [open]" if origin_open else " [closed]"),
            xy=(x_arr[orig], y_true[orig]),
            xytext=(0, 10), textcoords="offset points", fontsize=7, ha="center",
            color="black" if origin_open else "gray",
        )

        # ── Forecast fan: arrows + actual dot at each horizon ─────────────────
        for ci, h in enumerate(horizons_to_show):
            target_idx = orig + h
            if target_idx >= n or target_idx >= ctx_end:
                continue
            y_fc   = y_pred_all[orig, h - 1]
            y_act  = y_true[target_idx]
            x_fc   = x_arr[target_idx]
            color  = cmap(ci)

            # Arrow forecast → predicted point
            ax.annotate("", xy=(x_fc, y_fc), xytext=(x_arr[orig], y_true[orig]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))

            # Predicted dot
            ax.scatter([x_fc], [y_fc], color=color, zorder=6, s=50,
                       label=f"pred h={h}" if ax_i == 0 else "")

            # Actual dot at target (diamond marker) — compare vs prediction
            ax.scatter([x_fc], [y_act], color=color, zorder=7, s=60,
                       marker="D", edgecolors="black", linewidths=0.6,
                       label=f"actual h={h}" if ax_i == 0 else "")

            # Error annotation at h=H (last horizon)
            if h == horizons_to_show[-1] and y_act > 0:
                pct_err = abs(y_fc - y_act) / y_act * 100
                ax.annotate(f"{pct_err:.0f}%\nerr",
                            xy=(x_fc, max(y_fc, y_act)),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=6.5, color=color)

        ax.set_ylabel("Unit Sales")
        ax.grid(True, alpha=0.3)
        if dates is not None:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

        store_label = f"  Store {store_ids[orig]}" if store_ids is not None else ""
        date_label  = f"  ({x_arr[orig]})" if dates is not None else ""
        ax.set_title(f"Origin idx={orig}{store_label}{date_label}", fontsize=9)

        if ax_i == 0:
            ax.legend(fontsize=7.5, ncol=3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str = "nwrmsle",
    x_col: str = "model_name",
    save_path: str | None = None,
):
    """Bar chart comparing models/experiments on a given metric."""
    # biểu đồ cột so sánh metric giữa các model -> sort tăng dần, model tốt nhất ở bên trái
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_df = results_df.sort_values(metric)
    sns.barplot(data=sorted_df, x=x_col, y=metric, ax=ax, palette="viridis")
    ax.set_title(f"Comparison - {metric.upper()}")
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(metric.upper())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals",
    save_path: str | None = None,
):
    """Plot residual distribution and scatter."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # scatter: residual vs predicted -> nếu có pattern (phễu, cong) thì model chưa bắt hết signal
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=5)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{title} - Scatter")

    # histogram: phân phối residual -> lý tưởng là chuông đối xứng quanh 0
    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title} - Distribution")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    title: str = "Training Loss",
    save_path: str | None = None,
):
    """Vẽ đường loss theo từng epoch (train + val nếu có).
    
    Args:
        train_losses: loss từng epoch khi train
        val_losses:   loss từng epoch trên validation (None nếu không có)
        title:        tiêu đề biểu đồ
        save_path:    nếu có, lưu file PNG vào đường dẫn này
    """
    # bỏ epoch đầu tiên (thường rất cao, làm méo scale) -> hiển thị từ epoch 2 trở đi
    skip = 1
    train_plot = train_losses[skip:]
    val_plot = val_losses[skip:] if val_losses else None
    epochs = range(skip + 1, skip + 1 + len(train_plot))

    # nếu có val_losses -> vẽ thêm panel gap (train-val gap theo epoch)
    if val_plot and len(val_plot) == len(train_plot):
        fig, (ax, ax_gap) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax_gap = None

    ax.plot(epochs, train_plot, label="Train Loss", linewidth=2, marker="o", markersize=3)
    if val_plot:
        ax.plot(epochs, val_plot, label="Val Loss", linewidth=2, marker="s", markersize=3, linestyle="--")
        # vùng tô giữa train và val -> trực quan thấy gap
        ax.fill_between(epochs, train_plot, val_plot, alpha=0.12, color="orange", label="Gap")
        # đánh dấu epoch có val loss tốt nhất (tính trên toàn bộ val_losses gốc)
        best_epoch = int(np.argmin(val_losses)) + 1
        best_loss = min(val_losses)
        ax.axvline(x=best_epoch, color="green", linestyle=":", alpha=0.7)
        ax.annotate(
            f"Best val\nepoch {best_epoch}\n{best_loss:.4f}",
            xy=(best_epoch, best_loss),
            xytext=(best_epoch + max(1, len(train_losses) // 15), best_loss),
            fontsize=8,
            color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=1.2),
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch" if ax_gap is None else "")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # panel gap: val_loss - train_loss theo epoch -> gap dương = overfitting, gap âm = underfitting
    if ax_gap is not None and val_plot and len(val_plot) == len(train_plot):
        gap = [v - t for v, t in zip(val_plot, train_plot)]
        colors = ["tomato" if g > 0 else "steelblue" for g in gap]
        ax_gap.bar(epochs, gap, color=colors, alpha=0.7, width=0.6)
        ax_gap.axhline(y=0, color="black", linewidth=0.8)
        ax_gap.set_title("Val - Train gap  (đỏ = overfit, xanh = underfit)", fontsize=9)
        ax_gap.set_xlabel("Epoch")
        ax_gap.set_ylabel("Gap")
        ax_gap.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig