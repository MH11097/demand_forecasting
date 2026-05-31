"""Phase 3: ablation study biến ngoại sinh cho SARIMAX.

Cố định best params từ Phase 2, thử các tổ hợp exog_columns
→ tìm subset biến ngoại sinh tối ưu.

Dùng:
    python scripts/tune_sarimax_ablation.py                    # tự đọc Phase 2
    python scripts/tune_sarimax_ablation.py --n-stores 50
    python scripts/tune_sarimax_ablation.py --grid-path results/sarimax/tuning/grid_xxx/best_params.yaml
"""

import copy
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
import yaml

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.data.preprocessor import preprocess
from src.evaluation.metrics import evaluate_all, weights_from_frame
from src.models.sarimax import SARIMAXModel
from src.tuning.tuning_viz import plot_ablation_results
from src.utils.config import load_config
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Phase 3: ablation study biến ngoại sinh cho SARIMAX.")

# tổ hợp exog để thử — từ baseline (không biến) đến full (tất cả)
ABLATION_COMBOS = {
    "Baseline (no exog)": [],
    "+Promo": ["Promo"],
    "+SchoolHoliday": ["SchoolHoliday"],
    "+StateHoliday": ["StateHoliday"],
    "Promo+SchoolHoliday": ["Promo", "SchoolHoliday"],
    "Promo+StateHoliday": ["Promo", "StateHoliday"],
    "SchoolHoliday+StateHoliday": ["SchoolHoliday", "StateHoliday"],
    "Full (all)": ["Promo", "SchoolHoliday", "StateHoliday"],
}


def _find_latest_grid_best() -> Path | None:
    """Tìm best_params.yaml mới nhất từ Phase 2 grid search."""
    base = Path("results/sarimax/tuning")
    if not base.exists():
        return None
    dirs = sorted(base.glob("grid_*"), reverse=True)
    for d in dirs:
        p = d / "best_params.yaml"
        if p.exists():
            return p
    return None


@app.command()
def ablation(
    grid_path: str = typer.Option(None, "--grid-path", help="Đường dẫn best_params.yaml từ Phase 2"),
    n_stores: int = typer.Option(20, "--n-stores", "-n", help="Số store eval NWRMSLE"),
):
    """Ablation study: cố định best params, thay đổi tổ hợp exog_columns."""
    # đọc best params từ Phase 2
    yaml_path = Path(grid_path) if grid_path else _find_latest_grid_best()
    if yaml_path is None or not yaml_path.exists():
        typer.echo("Không tìm thấy best_params.yaml. Chạy tune_sarimax_grid.py trước!", err=True)
        raise typer.Exit(1)

    with open(yaml_path) as f:
        best_params = yaml.safe_load(f)
    typer.echo(f"Đọc best params từ: {yaml_path}")
    typer.echo(f"Best params: {best_params}")

    # load data
    config = load_config("sarimax")
    set_seed(config.get("seed", 42))
    # áp dụng best params từ Phase 2
    config["model"]["order"] = best_params["order"]
    config["model"]["seasonal_order"] = best_params["seasonal_order"]
    config["model"]["trend"] = best_params["trend"]

    typer.echo("Loading data...")
    df, _ = load_raw_data(config)
    df = filter_stores(df, config)
    df = add_all_features(df, feature_cfg=config.get("features", {}),
                          train_end=config.get("split", {}).get("train_end"))
    train_df, val_df, _, _ = preprocess(df, config)
    typer.echo(f"Data: {len(train_df)} train, {len(val_df)} val rows")

    # ablation — mỗi tổ hợp exog train riêng, eval NWRMSLE
    results = []
    total = len(ABLATION_COMBOS)
    for i, (exp_name, exog_cols) in enumerate(ABLATION_COMBOS.items(), 1):
        logger.info(f"[{i}/{total}] {exp_name}: exog={exog_cols}")

        cfg = copy.deepcopy(config)
        cfg["model"]["exog_columns"] = exog_cols

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAXModel(cfg)
                start = time.time()
                model.train(train_df, val_df)
                elapsed = time.time() - start

                predictions = model.predict(val_df)
                y_true = val_df["unit_sales"].values
                metrics = evaluate_all(y_true, predictions, weights_from_frame(val_df))

                row = {
                    "experiment": exp_name,
                    "exog_columns": ", ".join(exog_cols) if exog_cols else "(none)",
                    **metrics, "time_seconds": round(elapsed, 2),
                }
                results.append(row)
                logger.info(f"  NWRMSLE={metrics['nwrmsle']:.6f} ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"  FAILED: {e}")
            results.append({
                "experiment": exp_name,
                "exog_columns": ", ".join(exog_cols) if exog_cols else "(none)",
                "nwrmsle": float("inf"), "rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "rmspe": float("inf"),
                "time_seconds": 0,
            })

    results_df = pd.DataFrame(results)

    # lưu kết quả
    out_dir = Path("results/sarimax/tuning") / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "ablation_results.csv", index=False)
    plot_ablation_results(results_df, save_path=str(out_dir / "ablation_comparison.png"))

    # tạo markdown report ablation — dễ copy vào tiểu luận hoặc journal
    lines = ["# Kết quả Ablation Study SARIMAX", "", "## Bảng kết quả", ""]
    lines.append("| Thí nghiệm | Biến ngoại sinh | NWRMSLE | RMSE |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in results_df.iterrows():
        lines.append(f"| {row['experiment']} | {row['exog_columns']} | {row['nwrmsle']:.6f} | {row['rmse']:.4f} |")
    (out_dir / "ablation_results.md").write_text("\n".join(lines))

    # best config cuối cùng = best params + best exog
    best_row = results_df.loc[results_df["nwrmsle"].idxmin()]
    best_exog = [c.strip() for c in best_row["exog_columns"].split(",")] if best_row["exog_columns"] != "(none)" else []
    best_config = {**best_params, "exog_columns": best_exog}
    with open(out_dir / "best_config.yaml", "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)

    # in kết quả
    typer.echo(f"\n{'='*50}")
    typer.echo("KẾT QUẢ ABLATION STUDY (SARIMAX)")
    typer.echo(f"{'='*50}")
    typer.echo(results_df[["experiment", "exog_columns", "nwrmsle", "rmse"]].to_string(index=False))
    typer.echo(f"\nBest: {best_row['experiment']} — NWRMSLE={best_row['nwrmsle']:.6f}")
    typer.echo(f"Best config: {best_config}")
    typer.echo(f"\nKết quả lưu tại: {out_dir}")


if __name__ == "__main__":
    app()
