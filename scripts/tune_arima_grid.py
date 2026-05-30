"""Phase 2: grid search ARIMA — tự đọc kết quả Phase 1, tạo grid quanh best orders.

Dùng:
    python scripts/tune_arima_grid.py                    # tự đọc Phase 1 mới nhất
    python scripts/tune_arima_grid.py --n-stores 50      # tăng số store eval
    python scripts/tune_arima_grid.py --discovery-path results/arima/tuning/discovery_xxx/auto_arima_results.csv
"""

import copy
import itertools
import logging
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
import yaml

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.data.preprocessor import preprocess
from src.evaluation.metrics import evaluate_all
from src.models.arima import ARIMAModel
from src.tuning.tuning_viz import (
    generate_tuning_report,
    plot_param_sensitivity,
    plot_top_k_comparison,
    plot_tuning_heatmap,
)
from src.utils.config import load_config
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Phase 2: grid search ARIMA quanh kết quả auto_arima.")


def _find_latest_discovery() -> Path | None:
    """Tìm file CSV mới nhất từ Phase 1 discovery — sort theo tên thư mục (có timestamp)."""
    base = Path("results/arima/tuning")
    if not base.exists():
        return None
    # thư mục discovery_YYYYMMDD_HHMMSS → sort alphabetically = sort theo thời gian
    dirs = sorted(base.glob("discovery_*"), reverse=True)
    for d in dirs:
        csv = d / "auto_arima_results.csv"
        if csv.exists():
            return csv
    return None


def _build_grid_from_discovery(csv_path: Path) -> dict:
    """Đọc CSV Phase 1 → lấy top orders → tạo vùng p/d/q (±1) cho grid search."""
    df = pd.read_csv(csv_path)
    valid = df.dropna(subset=["p"])
    if len(valid) == 0:
        typer.echo("Không có kết quả hợp lệ trong discovery CSV!", err=True)
        raise typer.Exit(1)

    # đếm order phổ biến nhất → mở rộng ±1 để không bỏ sót cấu hình lân cận
    order_counter = Counter(zip(valid["p"].astype(int), valid["d"].astype(int), valid["q"].astype(int)))
    top_orders = order_counter.most_common(3)
    typer.echo(f"Top orders từ discovery: {top_orders}")

    # gộp tất cả p/d/q từ top orders rồi mở rộng ±1
    p_set, d_set, q_set = set(), set(), set()
    for (p, d, q), _ in top_orders:
        for delta in [-1, 0, 1]:
            p_set.add(max(0, p + delta))
            d_set.add(max(0, min(2, d + delta)))  # d tối đa 2
            q_set.add(max(0, q + delta))

    return {
        "p": sorted(p_set),
        "d": sorted(d_set),
        "q": sorted(q_set),
        "trend": ["n", "c", "t", "ct"],
    }


@app.command()
def grid_search(
    discovery_path: str = typer.Option(None, "--discovery-path", help="Đường dẫn CSV Phase 1 (tự tìm nếu không truyền)"),
    n_stores: int = typer.Option(20, "--n-stores", "-n", help="Số store dùng để eval NWRMSLE"),
):
    """Grid search ARIMA quanh best orders từ Phase 1 → tìm (p,d,q)+trend tối ưu."""
    # tìm CSV Phase 1 — ưu tiên CLI, fallback tự tìm mới nhất
    if discovery_path:
        csv_path = Path(discovery_path)
    else:
        csv_path = _find_latest_discovery()
    if csv_path is None or not csv_path.exists():
        typer.echo("Không tìm thấy kết quả Phase 1. Chạy tune_arima_discovery.py trước!", err=True)
        raise typer.Exit(1)
    typer.echo(f"Đọc discovery từ: {csv_path}")

    # tạo grid từ kết quả Phase 1
    param_space = _build_grid_from_discovery(csv_path)
    param_grid = [
        {"order": [p, d, q], "trend": trend}
        for p, d, q, trend in itertools.product(param_space["p"], param_space["d"], param_space["q"], param_space["trend"])
    ]
    typer.echo(f"Grid: p={param_space['p']}, d={param_space['d']}, q={param_space['q']}, trend={param_space['trend']}")
    typer.echo(f"Tổng tổ hợp: {len(param_grid)}")

    # load data
    config = load_config("arima")
    set_seed(config.get("seed", 42))
    typer.echo("Loading data...")
    df, _ = load_raw_data(config)
    df = filter_stores(df, config)
    df = add_all_features(df)
    train_df, val_df, _, _ = preprocess(df, config)
    typer.echo(f"Data: {len(train_df)} train, {len(val_df)} val rows")

    # grid search — mỗi combo train ARIMAModel riêng, eval NWRMSLE trên val set
    results = []
    for i, params in enumerate(param_grid, 1):
        param_str = f"order={params['order']}, trend={params['trend']}"
        logger.info(f"[{i}/{len(param_grid)}] {param_str}")

        cfg = copy.deepcopy(config)
        cfg["model"]["order"] = params["order"]
        cfg["model"]["trend"] = params["trend"]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMAModel(cfg)
                start = time.time()
                model.train(train_df, val_df)
                elapsed = time.time() - start

                predictions = model.predict(val_df)
                y_true = val_df["unit_sales"].values
                metrics = evaluate_all(y_true, predictions)

                row = {
                    "order": str(params["order"]),
                    "p": params["order"][0], "d": params["order"][1], "q": params["order"][2],
                    "trend": params["trend"],
                    **metrics, "time_seconds": round(elapsed, 2),
                }
                results.append(row)
                logger.info(f"  NWRMSLE={metrics['nwrmsle']:.6f} ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"  FAILED: {e}")
            results.append({
                "order": str(params["order"]),
                "p": params["order"][0], "d": params["order"][1], "q": params["order"][2],
                "trend": params["trend"],
                "nwrmsle": float("inf"), "rmse": float("inf"), "mae": float("inf"), "mape": float("inf"),
                "time_seconds": 0,
            })

    results_df = pd.DataFrame(results).sort_values("nwrmsle").reset_index(drop=True)

    # lưu kết quả
    out_dir = Path("results/arima/tuning") / f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "tuning_results.csv", index=False)
    (out_dir / "tuning_results.md").write_text(generate_tuning_report(results_df, model_name="ARIMA"))

    # best params → YAML để dùng ngay với train.py
    best = results_df.iloc[0]
    best_params = {"order": [int(best["p"]), int(best["d"]), int(best["q"])], "trend": best["trend"]}
    with open(out_dir / "best_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    # biểu đồ — reuse tuning_viz cho sensitivity, heatmap, top-K
    for param in ["p", "d", "q", "trend"]:
        plot_param_sensitivity(results_df, param, save_path=str(out_dir / f"sensitivity_{param}.png"))
    plot_tuning_heatmap(results_df, "p", "q", save_path=str(out_dir / "heatmap_p_vs_q.png"))
    plot_tuning_heatmap(results_df, "p", "trend", save_path=str(out_dir / "heatmap_p_vs_trend.png"))
    plot_top_k_comparison(results_df, k=min(10, len(results_df)), save_path=str(out_dir / "top10_comparison.png"))

    # in kết quả
    typer.echo(f"\n{'='*50}")
    typer.echo("KẾT QUẢ GRID SEARCH ARIMA")
    typer.echo(f"{'='*50}")
    typer.echo(f"Best NWRMSLE: {best['nwrmsle']:.6f}")
    typer.echo(f"Best params: order={best_params['order']}, trend={best_params['trend']}")
    typer.echo(f"\nTop 5:")
    typer.echo(results_df[["order", "trend", "nwrmsle", "rmse", "time_seconds"]].head().to_string(index=False))
    typer.echo(f"\nKết quả lưu tại: {out_dir}")


if __name__ == "__main__":
    app()
