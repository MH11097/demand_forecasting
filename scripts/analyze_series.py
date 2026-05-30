"""CLI phân tích chuỗi thời gian theo Box-Jenkins.

Chạy kiểm định tính dừng, ACF/PACF, chẩn đoán phần dư trên N store.
Dùng cho thực nghiệm tương tác: chạy từng bước → xem kết quả → quyết định bước tiếp.

Usage:
    python scripts/analyze_series.py --model arima --n-stores 5 --step stationarity
    python scripts/analyze_series.py --model arima --n-stores 5 --step acf_pacf --d 1
    python scripts/analyze_series.py --model sarimax --n-stores 5 --step acf_pacf --d 1 --seasonal-diff
"""

import json
from pathlib import Path

import numpy as np
import typer

from src.analysis.acf_pacf import plot_acf_pacf, suggest_pq_range
from src.analysis.residual_diagnostics import diagnose_residuals
from src.analysis.stationarity import stationarity_summary, test_stationarity
from src.data.features import add_all_features
from src.data.loader import filter_stores, load_cleaned_data, load_raw_data
from src.utils.config import load_config
from src.utils.seed import set_seed

app = typer.Typer(help="Phân tích chuỗi thời gian theo Box-Jenkins.")


def _load_train_series(model_name: str, n_stores: int) -> dict[int, np.ndarray]:
    """Load dữ liệu train, trả dict {store_id: sales_array} cho N store đầu tiên.

    Chỉ lấy phần train (trước split date) để phân tích → không leak dữ liệu val/test.
    """
    config = load_config(model_name)
    set_seed(config.get("seed", 42))

    use_cleaned = config.get("data", {}).get("use_cleaned", False)
    if use_cleaned:
        df, _ = load_cleaned_data(config)
    else:
        df, _ = load_raw_data(config)
    df = filter_stores(df, config)
    df = add_all_features(df)

    # chỉ lấy phần train → phân tích trên dữ liệu mà model sẽ thấy khi fit
    train_end = config["split"]["train_end"]
    train_df = df[df["date"] <= train_end].copy()

    stores = sorted(train_df["series_id"].unique())[:n_stores]
    series_dict = {}
    for sid in stores:
        store_data = train_df[train_df["series_id"] == sid].sort_values("date")
        series_dict[sid] = store_data["unit_sales"].values.astype(float)
    return series_dict


def _apply_diff(series: np.ndarray, d: int, seasonal_diff: bool = False, period: int = 7) -> np.ndarray:
    """Áp dụng sai phân: d lần regular + 1 lần seasonal (nếu có).

    Regular diff (d=1): loại trend → chuỗi dừng.
    Seasonal diff: loại pattern lặp lại mỗi `period` ngày (7 cho weekly).
    """
    result = series.copy()
    for _ in range(d):
        result = np.diff(result)
    if seasonal_diff and len(result) > period:
        # seasonal diff: y_t - y_{t-period} → loại seasonal component
        result = result[period:] - result[:-period]
    return result


@app.command()
def analyze(
    model: str = typer.Option("arima", "--model", "-m", help="Model name (arima/sarimax)"),
    n_stores: int = typer.Option(5, "--n-stores", "-n", help="Số store phân tích"),
    step: str = typer.Option(..., "--step", help="Bước phân tích: stationarity, acf_pacf, residual"),
    d: int = typer.Option(0, "--d", help="Bậc sai phân cho ACF/PACF"),
    seasonal_diff: bool = typer.Option(False, "--seasonal-diff", help="Áp dụng seasonal differencing (period=7)"),
    nlags: int = typer.Option(40, "--nlags", help="Số lag tối đa cho ACF/PACF"),
    output_dir: str = typer.Option("results/analysis", "--output-dir", "-o", help="Thư mục lưu kết quả"),
):
    """Chạy phân tích chuỗi thời gian theo từng bước."""
    out = Path(output_dir) / model / step
    out.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading data for {model}, {n_stores} stores...")
    series_dict = _load_train_series(model, n_stores)
    typer.echo(f"Loaded {len(series_dict)} stores: {list(series_dict.keys())}")

    if step == "stationarity":
        _step_stationarity(series_dict, out)
    elif step == "acf_pacf":
        _step_acf_pacf(series_dict, d, seasonal_diff, nlags, out)
    elif step == "residual":
        typer.echo("Bước residual cần chạy sau khi train model. Dùng --diagnostics trong train.py.")
    else:
        typer.echo(f"Step không hợp lệ: {step}. Chọn: stationarity, acf_pacf, residual")


def _step_stationarity(series_dict: dict[int, np.ndarray], out: Path):
    """Bước 1: Kiểm định ADF + KPSS cho từng store → bảng tổng hợp + gợi ý d."""
    results = []
    for sid, sales in series_dict.items():
        typer.echo(f"  Testing store {sid} ({len(sales)} obs)...")
        res = test_stationarity(sales, store_id=sid)
        results.append(res)

    # bảng tổng hợp → user đọc để quyết định d chung cho tất cả store
    summary_df = stationarity_summary(results)
    csv_path = out / "stationarity_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    typer.echo(f"\nBảng tổng hợp đã lưu: {csv_path}")
    typer.echo(summary_df.to_string(index=False))

    # thống kê nhanh: bao nhiêu store gợi ý d=0 vs d=1
    d_counts = summary_df["suggested_d"].value_counts().to_dict()
    typer.echo(f"\nPhân bố d gợi ý: {d_counts}")

    # lưu kết quả chi tiết (từng level d=0, d=1...) dạng JSON
    detail_path = out / "stationarity_detail.json"
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    typer.echo(f"Chi tiết đã lưu: {detail_path}")


def _step_acf_pacf(
    series_dict: dict[int, np.ndarray],
    d: int,
    seasonal_diff: bool,
    nlags: int,
    out: Path,
):
    """Bước 2: Vẽ ACF/PACF + gợi ý vùng p, q cho từng store."""
    all_suggestions = []

    for sid, sales in series_dict.items():
        # áp dụng sai phân theo d đã chọn từ bước 1
        diffed = _apply_diff(sales, d, seasonal_diff)
        diff_label = f"d={d}" + (",D=1" if seasonal_diff else "")
        typer.echo(f"  Store {sid}: {len(sales)} obs → diff({diff_label}) → {len(diffed)} obs")

        # vẽ biểu đồ ACF/PACF → user đọc pattern trực quan
        plot_path = str(out / f"acf_pacf_store{sid}.png")
        plot_acf_pacf(diffed, nlags=nlags, title=f"Store {sid} ({diff_label})", save_path=plot_path)

        # gợi ý tự động dựa trên heuristic → tham khảo, user quyết định cuối cùng
        suggestion = suggest_pq_range(diffed, nlags=nlags)
        suggestion["store_id"] = int(sid)
        all_suggestions.append(suggestion)

        typer.echo(f"    p range: {suggestion['suggested_p_range']}, q range: {suggestion['suggested_q_range']}")
        if suggestion["has_weekly_pattern"]:
            typer.echo(f"    Weekly seasonal spikes at lags: {suggestion['seasonal_spikes_at_7']}")

    # lưu gợi ý tổng hợp → user tham khảo khi cấu hình grid search
    suggest_path = out / "pq_suggestions.json"
    with open(suggest_path, "w") as f:
        json.dump(all_suggestions, f, indent=2)
    typer.echo(f"\nGợi ý p/q đã lưu: {suggest_path}")
    typer.echo(f"Biểu đồ ACF/PACF đã lưu tại: {out}/")


if __name__ == "__main__":
    app()
