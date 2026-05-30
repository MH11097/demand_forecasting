"""CLI script for comparing model results."""

from pathlib import Path

import typer

from src.evaluation.comparison import comparison_table, load_results
from src.utils.visualization import plot_metric_comparison

app = typer.Typer(help="Compare forecasting model results.")


@app.command()
def compare(
    results_dir: str = typer.Option("results", "--results-dir", "-d", help="Results directory"),
    output_dir: str = typer.Option("results/comparison", "--output-dir", "-o", help="Output directory"),
    model_filter: str = typer.Option("", "--model", "-m", help="Filter by model name (e.g. arima)"),
):
    """Generate comparison table and charts from model results."""
    # load tất cả result.json từ các lần chạy -> gom vào bảng so sánh
    results = load_results(results_dir, model_name=model_filter or None)
    if len(results) < 2:
        typer.echo(f"Need at least 2 results to compare, found {len(results)}")
        raise typer.Exit(1)

    df = comparison_table(results)
    typer.echo("\nModel Comparison (sorted by NWRMSLE):")
    typer.echo(df.to_markdown(index=False))

    # lọc 1 model -> lưu vào subfolder riêng, so sánh các hyperparams khác nhau của cùng model
    out_path = Path(output_dir) / model_filter if model_filter else Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path / "comparison.csv", index=False)
    with open(out_path / "comparison.md", "w") as f:
        f.write("# Model Comparison\n\n")
        f.write(df.to_markdown(index=False))

    # so sánh nhiều model -> trục x là tên model; so sánh 1 model -> trục x là experiment/hyperparams
    x_col = "model_name"
    if model_filter:
        x_col = "experiment_name" if df["experiment_name"].any() else "param_slug"

    # vẽ biểu đồ cho từng metric -> nhìn đa chiều, model tốt NWRMSLE chưa chắc tốt MAE
    for metric in ["nwrmsle", "rmse", "mae", "mape"]:
        if metric in df.columns:
            plot_metric_comparison(df, metric=metric, x_col=x_col, save_path=str(out_path / f"{metric}_comparison.png"))

    typer.echo(f"\nComparison saved to {out_path}")


if __name__ == "__main__":
    app()
