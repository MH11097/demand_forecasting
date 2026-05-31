"""So sánh 5 models trên 30-day forecast: bảng metrics + biểu đồ Actual vs Forecast + Error comparison.

Output lưu tại results/comparison/:
  - comparison_table.csv / .md   — bảng MAE, RMSE, MAPE, RMSPE (CV 5-fold mean ± std)
  - error_comparison.png         — bar chart so sánh các metrics
  - actual_vs_forecast.png       — overlay actual + 5 model predictions (30 ngày, trung bình qua stores)
  - error_by_model.png           — boxplot/bar chart phân phối error theo model
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Thêm project root vào path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features import add_all_features, apply_log_transform
from src.data.loader import filter_stores, load_raw_data, prepare_holdout_panel
from src.data.preprocessor import preprocess
from src.evaluation.metrics import evaluate_all, weights_from_frame
from src.models.base import BaseModel
from src.utils.config import load_config

# ── Config ───────────────────────────────────────────────────────────────────
MODELS = ["arima", "sarimax", "prophet", "xgboost", "lstm"]
DISPLAY_NAMES = {
    "arima": "ARIMA",
    "sarimax": "SARIMAX",
    "prophet": "Prophet",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}
# Thứ tự hiển thị: statistical → ML → DL
MODEL_ORDER = ["ARIMA", "SARIMAX", "Prophet", "XGBoost", "LSTM"]

OUTPUT_DIR = Path("results/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 30-day test window chung cho tất cả models (trong khoảng test split Favorita 2017-05-16→08-15)
TEST_START = "2017-07-02"
TEST_END = "2017-07-31"

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
PALETTE = {
    "ARIMA": "#e74c3c",
    "SARIMAX": "#e67e22",
    "Prophet": "#2ecc71",
    "XGBoost": "#3498db",
    "LSTM": "#9b59b6",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BẢNG SO SÁNH TỪ CV RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
def build_cv_table() -> pd.DataFrame:
    """Đọc cv_results.json của 5 models → tạo bảng so sánh metrics."""
    rows = []
    for model_name in MODELS:
        cv_path = Path(f"results/{model_name}/cv/cv_results.json")
        if not cv_path.exists():
            print(f"  [SKIP] {model_name}: không tìm thấy {cv_path}")
            continue
        with open(cv_path) as f:
            cv = json.load(f)
        agg = cv["aggregated"]
        row = {
            "Model": DISPLAY_NAMES[model_name],
            "MAE": f"{agg['mae_mean']:.1f} ± {agg['mae_std']:.1f}",
            "RMSE": f"{agg['rmse_mean']:.1f} ± {agg['rmse_std']:.1f}",
            "MAPE (%)": f"{agg['mape_mean'] * 100:.2f} ± {agg['mape_std'] * 100:.2f}",
            "NWRMSLE": f"{agg['nwrmsle_mean']:.4f} ± {agg['nwrmsle_std']:.4f}",
            # Giá trị số để sort
            "_mae": agg["mae_mean"],
            "_rmse": agg["rmse_mean"],
            "_mape": agg["mape_mean"],
            "_nwrmsle": agg["nwrmsle_mean"],
        }
        if "rmspe_mean" in agg:
            row["RMSPE (%)"] = f"{agg['rmspe_mean'] * 100:.2f} ± {agg['rmspe_std'] * 100:.2f}"
            row["_rmspe"] = agg["rmspe_mean"]
        rows.append(row)
    df = pd.DataFrame(rows)
    # Sort theo NWRMSLE (primary metric)
    df = df.sort_values("_nwrmsle").reset_index(drop=True)
    return df


def save_table(df: pd.DataFrame):
    """Lưu bảng so sánh dạng CSV và Markdown."""
    display_cols = ["Model", "MAE", "RMSE", "MAPE (%)"]
    if "RMSPE (%)" in df.columns:
        display_cols.append("RMSPE (%)")
    display_cols.append("NWRMSLE")
    # CSV
    df[display_cols].to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
    # Markdown
    md = df[display_cols].to_markdown(index=False)
    (OUTPUT_DIR / "comparison_table.md").write_text(
        "# Model Comparison — 5-Fold Expanding CV\n\n"
        "Metrics: mean ± std across 5 folds. Sorted by NWRMSLE (primary metric).\n\n"
        f"{md}\n"
    )
    print(f"  → Saved comparison_table.csv/.md")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ERROR COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
def plot_error_comparison(df: pd.DataFrame):
    """Bar chart so sánh MAE, RMSE, MAPE, RMSPE, NWRMSLE của 5 models."""
    metrics = [
        ("_mae", "MAE", "Mean Absolute Error"),
        ("_rmse", "RMSE", "Root Mean Squared Error"),
        ("_mape", "MAPE (%)", "Mean Absolute Percentage Error"),
        ("_rmspe", "RMSPE (%)", "Root Mean Squared Percentage Error"),
        ("_nwrmsle", "NWRMSLE", "Normalized Weighted RMSLE"),
    ]
    metrics = [
        metric for metric in metrics
        if metric[0] in df.columns and df[metric[0]].notna().all()
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
    axes = np.atleast_1d(axes)

    for ax, (col, label, title) in zip(axes, metrics):
        plot_df = df.set_index("Model").loc[MODEL_ORDER].reset_index()
        values = plot_df[col].values
        if "pe" in col:
            values = values * 100  # percentage metrics
        colors = [PALETTE[m] for m in plot_df["Model"]]

        bars = ax.bar(plot_df["Model"], values, color=colors, edgecolor="white", linewidth=0.8)
        # Giá trị trên mỗi cột
        for bar, v in zip(bars, values):
            fmt = f"{v:.1f}%" if "pe" in col else f"{v:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(label)
        ax.set_ylim(0, max(values) * 1.18)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Error Comparison — 5-Fold Cross-Validation", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved error_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOAD MODELS + GENERATE 30-DAY PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def _find_latest_run(model_name: str) -> Path | None:
    """Tìm run directory mới nhất (theo mtime — an toàn hơn so với sort theo tên
    khi các run có prefix khác nhau, ví dụ order_3-... vs order_4-...)."""
    base = Path("results") / model_name
    if not base.exists():
        return None
    # Bỏ qua cv/ và tuning/ và grid_search*
    runs = [d for d in base.iterdir()
            if d.is_dir() and d.name not in ("cv", "tuning") and not d.name.startswith("grid_search")]
    if not runs:
        return None
    return max(runs, key=lambda d: d.stat().st_mtime)


def load_and_predict(model_name: str) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray | None] | None:
    """Load model đã train, predict trên 30-day test window.

    Returns:
        (y_true, y_pred, dates, weights) hoặc None nếu load/predict fail
    """
    run_dir = _find_latest_run(model_name)
    if run_dir is None:
        print(f"  [SKIP] {model_name}: không tìm thấy run directory")
        return None

    model_path = run_dir / "model.pkl"
    if not model_path.exists():
        print(f"  [SKIP] {model_name}: không có model.pkl tại {run_dir}")
        return None

    try:
        # Load config cho model này
        config = load_config(model_name, {})

        # Override split dates → 30-day test window chung
        config["split"] = {
            "train_end": (pd.Timestamp(TEST_START) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "test_start": TEST_START,
            "test_end": TEST_END,
            "panel_lookback_days": 90,
        }

        # Load + prepare data
        df, _ = load_raw_data(config)
        df = filter_stores(df, config)
        df = prepare_holdout_panel(df, config)
        df = add_all_features(df, feature_cfg=config.get("features", {}),
                              train_end=config["split"]["train_end"])
        if config.get("use_log_sales", False):
            df = apply_log_transform(df)

        train_df, _, test_df, _ = preprocess(df, config)

        if len(test_df) == 0:
            print(f"  [SKIP] {model_name}: test_df rỗng cho period {TEST_START}–{TEST_END}")
            return None

        # Load model
        loaded = BaseModel.load(str(model_path))

        # Prepend context cho sequence models (LSTM)
        model_cfg = config.get("model", {})
        seq_len = model_cfg.get("seq_len", 0)
        horizon = model_cfg.get("forecast_horizon", 1)
        ctx_len = seq_len + horizon - 1

        if ctx_len > 0 and "series_id" in train_df.columns:
            ctx = train_df.groupby("series_id", group_keys=False).tail(ctx_len)
            combined = pd.concat([ctx, test_df]).reset_index(drop=True)
            preds = loaded.predict(combined)
            preds = preds[len(ctx):]
        else:
            preds = loaded.predict(test_df)

        # True sales (inverse log nếu cần)
        y_true = test_df["unit_sales"].values.astype(float)
        if config.get("use_log_sales", False):
            y_true = np.expm1(y_true)

        dates = test_df["date"]
        return y_true, preds, dates, weights_from_frame(test_df)

    except Exception as e:
        print(f"  [ERROR] {model_name}: {e}")
        return None


def aggregate_by_date(y_true, y_pred, dates) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """Trung bình actual/predicted theo ngày (gộp tất cả stores)."""
    tmp = pd.DataFrame({"date": dates.values, "actual": y_true, "predicted": y_pred})
    grouped = tmp.groupby("date").mean()
    return grouped["actual"], grouped["predicted"], grouped.index


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ACTUAL VS FORECAST CHART
# ═══════════════════════════════════════════════════════════════════════════════
def plot_actual_vs_forecast(predictions: dict):
    """Biểu đồ overlay: 1 actual line + 5 predicted lines (trung bình qua stores, 30 ngày)."""
    fig, ax = plt.subplots(figsize=(16, 7))

    actual_plotted = False
    for model_name in MODEL_ORDER:
        if model_name not in predictions:
            continue
        actual_agg, pred_agg, dates = predictions[model_name]

        if not actual_plotted:
            ax.plot(dates, actual_agg, color="black", linewidth=2.5, label="Actual",
                    marker="o", markersize=4, zorder=10)
            actual_plotted = True

        ax.plot(dates, pred_agg, color=PALETTE[model_name], linewidth=1.8,
                label=model_name, linestyle="--", marker="s", markersize=3, alpha=0.85)

    ax.set_title("Actual vs Forecast — 30-Day Test Window\n(trung bình qua tất cả stores)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (mean across stores)")
    ax.legend(fontsize=10, loc="upper right")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_forecast.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved actual_vs_forecast.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ERROR BY MODEL — DAILY ERROR DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_error_by_model(predictions: dict):
    """Boxplot phân phối daily absolute percentage error theo model."""
    records = []
    for model_name in MODEL_ORDER:
        if model_name not in predictions:
            continue
        actual_agg, pred_agg, dates = predictions[model_name]
        # Daily APE (%) — tránh chia cho 0
        mask = actual_agg > 0
        ape = np.abs(pred_agg[mask] - actual_agg[mask]) / actual_agg[mask] * 100
        for v in ape:
            records.append({"Model": model_name, "APE (%)": v})

    if not records:
        return

    err_df = pd.DataFrame(records)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Boxplot
    order = [m for m in MODEL_ORDER if m in err_df["Model"].values]
    colors = [PALETTE[m] for m in order]
    sns.boxplot(data=err_df, x="Model", y="APE (%)", order=order, palette=colors, ax=ax1,
                fliersize=3, linewidth=1.2)
    ax1.set_title("Daily Error Distribution\n(Absolute Percentage Error per day)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("APE (%)")

    # Bar chart: median APE
    medians = err_df.groupby("Model")["APE (%)"].median().reindex(order)
    bars = ax2.bar(order, medians, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, medians):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_title("Median Daily APE (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("APE (%)")
    ax2.set_ylim(0, max(medians) * 1.25)

    fig.suptitle("Error Comparison — 30-Day Forecast", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_by_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved error_by_model.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("MODEL COMPARISON — 30-Day Forecast")
    print("=" * 60)

    # 1. Bảng so sánh CV metrics
    print("\n[1/4] Building CV comparison table...")
    cv_df = build_cv_table()
    save_table(cv_df)
    display_cols = ["Model", "MAE", "RMSE", "MAPE (%)"]
    if "RMSPE (%)" in cv_df.columns:
        display_cols.append("RMSPE (%)")
    display_cols.append("NWRMSLE")
    print(cv_df[display_cols].to_string(index=False))

    # 2. Error comparison bar chart
    print("\n[2/4] Creating error comparison chart...")
    plot_error_comparison(cv_df)

    # 3. Load models + predict
    print(f"\n[3/4] Loading models & predicting on {TEST_START} → {TEST_END}...")
    predictions = {}
    for model_name in MODELS:
        display = DISPLAY_NAMES[model_name]
        print(f"  Loading {display}...")
        result = load_and_predict(model_name)
        if result is not None:
            y_true, y_pred, dates, weights = result
            actual_agg, pred_agg, agg_dates = aggregate_by_date(y_true, y_pred, dates)
            predictions[display] = (actual_agg, pred_agg, agg_dates)
            metrics = evaluate_all(y_true, y_pred, weights)
            print(f"    ✓ {display}: NWRMSLE={metrics['nwrmsle']:.4f}, MAE={metrics['mae']:.1f}, RMSE={metrics['rmse']:.1f}")

    # 4. Charts
    print("\n[4/4] Creating charts...")
    if predictions:
        plot_actual_vs_forecast(predictions)
        plot_error_by_model(predictions)
    else:
        print("  [WARN] Không load được model nào → bỏ qua biểu đồ Actual vs Forecast")

    print(f"\n✓ Tất cả output lưu tại: {OUTPUT_DIR}/")
    print("  - comparison_table.csv / .md")
    print("  - error_comparison.png")
    if predictions:
        print("  - actual_vs_forecast.png")
        print("  - error_by_model.png")


if __name__ == "__main__":
    main()
