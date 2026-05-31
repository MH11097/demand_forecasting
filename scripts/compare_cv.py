"""So sánh cross-validation của tất cả models trên cùng điều kiện.

Chạy walk-forward CV cho từng model, thu thập metrics, in bảng so sánh.
Mặc định đánh giá trên 30 ngày dự đoán (eval_days=30) với 3 folds.

Usage:
    python scripts/compare_cv.py
    python scripts/compare_cv.py --models arima,xgboost,lstm
    python scripts/compare_cv.py --n-splits 5 --eval-days 30
"""

import json
import time
from pathlib import Path

import pandas as pd
import typer

from src.data.loader import load_raw_data, select_series
from src.evaluation.cross_validation import walk_forward_cv
from src.models import MODEL_REGISTRY, get_model_class
from src.utils.config import load_config
from src.utils.seed import set_seed

app = typer.Typer(help="So sánh cross-validation của tất cả models.")

ALL_MODELS = list(MODEL_REGISTRY.keys())


@app.command()
def compare(
    models: str = typer.Option(
        ",".join(ALL_MODELS), "--models", "-m",
        help="Danh sách models cách nhau bởi dấu phẩy (default: tất cả)"
    ),
    n_splits: int = typer.Option(3, "--n-splits", "-n", help="Số CV folds"),
    eval_days: int = typer.Option(30, "--eval-days", "-e", help="Số ngày đánh giá mỗi fold"),
    cv_strategy: str = typer.Option("expanding", "--cv", help="CV strategy: expanding hoặc sliding"),
    output_dir: str = typer.Option("results/comparison", "--output-dir", "-o", help="Thư mục lưu kết quả"),
):
    """Chạy walk-forward CV cho từng model và tạo bảng so sánh."""
    model_names = [m.strip() for m in models.split(",")]

    # validate tên model
    for name in model_names:
        if name not in MODEL_REGISTRY:
            typer.echo(f"Model không hợp lệ: {name}. Có: {ALL_MODELS}", err=True)
            raise typer.Exit(1)

    typer.echo(f"=== Cross-Validation Comparison ===")
    typer.echo(f"Models: {model_names}")
    typer.echo(f"CV: {cv_strategy}, {n_splits} folds, eval_days={eval_days}")
    typer.echo("")

    results = []

    for model_name in model_names:
        typer.echo(f"{'='*60}")
        typer.echo(f"[{model_name.upper()}] Loading config & data...")

        config = load_config(model_name)
        set_seed(config.get("seed", 42))

        # load data + chọn mẫu đại diện chuỗi (theo series_sample config)
        df, _ = load_raw_data(config)
        df = select_series(df, config)

        n_series = df["series_id"].nunique() if "series_id" in df.columns else 0
        typer.echo(f"[{model_name.upper()}] {len(df)} rows, {n_series} series")
        typer.echo(f"[{model_name.upper()}] Running {cv_strategy} CV ({n_splits} folds, {eval_days} eval days)...")

        model_class = get_model_class(model_name)
        start = time.time()
        cv_results = walk_forward_cv(
            model_class,
            config,
            df,
            n_splits=n_splits,
            expanding=(cv_strategy == "expanding"),
            eval_days=eval_days,
        )
        total_time = time.time() - start

        # per-fold log — dùng nwrmsle (metric chính thức)
        for fold in cv_results["folds"]:
            nwrmsle_v = fold.get("nwrmsle", "N/A")
            nwrmsle_s = f"{nwrmsle_v:.4f}" if isinstance(nwrmsle_v, float) else str(nwrmsle_v)
            typer.echo(
                f"  Fold {fold['fold']}: NWRMSLE={nwrmsle_s} "
                f"RMSE={fold.get('rmse', 0):.1f} MAE={fold.get('mae', 0):.1f} "
                f"({fold.get('training_time_seconds', 0):.1f}s)"
            )

        agg = cv_results["aggregated"]
        nwrmsle_mean = agg.get("nwrmsle_mean", "N/A")
        nwrmsle_std  = agg.get("nwrmsle_std",  "N/A")
        nwrmsle_mean_s = f"{nwrmsle_mean:.4f}" if isinstance(nwrmsle_mean, float) else str(nwrmsle_mean)
        nwrmsle_std_s  = f"{nwrmsle_std:.4f}"  if isinstance(nwrmsle_std, float)  else str(nwrmsle_std)
        typer.echo(f"[{model_name.upper()}] Mean NWRMSLE={nwrmsle_mean_s} ± {nwrmsle_std_s} ({total_time:.1f}s total)")

        # thu thập kết quả vào bảng
        row = {
            "model": model_name,
            "n_series": n_series,
            "nwrmsle_mean": agg.get("nwrmsle_mean"),
            "nwrmsle_std":  agg.get("nwrmsle_std"),
            "rmse_mean":  agg.get("rmse_mean"),
            "rmse_std":   agg.get("rmse_std"),
            "mae_mean":   agg.get("mae_mean"),
            "mae_std":    agg.get("mae_std"),
            "mape_mean":  agg.get("mape_mean"),
            "mape_std":   agg.get("mape_std"),
            "rmspe_mean": agg.get("rmspe_mean"),
            "rmspe_std":  agg.get("rmspe_std"),
            "avg_fold_time": round(total_time / max(n_splits, 1), 1),
            "total_time": round(total_time, 1),
        }
        results.append(row)

        # lưu CV detail cho từng model
        model_cv_dir = Path(output_dir) / model_name
        model_cv_dir.mkdir(parents=True, exist_ok=True)
        with open(model_cv_dir / "cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=2)

    # tạo bảng so sánh
    typer.echo(f"\n{'='*60}")
    typer.echo(f"=== COMPARISON TABLE (sorted by NWRMSLE) ===")
    typer.echo(f"{'='*60}\n")

    df_results = pd.DataFrame(results)
    sort_col = "nwrmsle_mean" if "nwrmsle_mean" in df_results.columns else "rmse_mean"
    if sort_col in df_results.columns:
        df_results = df_results.sort_values(sort_col).reset_index(drop=True)

    typer.echo(df_results.to_markdown(index=False))

    # lưu kết quả
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(out_path / "cv_comparison.csv", index=False)
    with open(out_path / "cv_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_path / "cv_comparison.md", "w") as f:
        f.write(f"# Cross-Validation Comparison\n\n")
        f.write(f"- CV: {cv_strategy}, {n_splits} folds, eval_days={eval_days}\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n")

    typer.echo(f"\nKết quả lưu tại: {out_path}/")


if __name__ == "__main__":
    app()
