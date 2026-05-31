"""Phase 2: grid search SARIMAX — tự đọc kết quả Phase 1, tạo grid quanh best orders.

Dùng:
    python scripts/tune_sarimax_grid.py                    # tự đọc Phase 1 mới nhất
    python scripts/tune_sarimax_grid.py --n-stores 50
    python scripts/tune_sarimax_grid.py --discovery-path results/sarimax/tuning/discovery_xxx/auto_arima_results.csv
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
from src.evaluation.metrics import evaluate_all, weights_from_frame
from src.models.sarimax import SARIMAXModel
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

app = typer.Typer(help="Phase 2: grid search SARIMAX quanh kết quả auto_arima.")


def _find_latest_discovery() -> Path | None:
    """Tìm CSV mới nhất từ Phase 1 discovery."""
    base = Path("results/sarimax/tuning")
    if not base.exists():
        return None
    dirs = sorted(base.glob("discovery_*"), reverse=True)
    for d in dirs:
        csv = d / "auto_arima_results.csv"
        if csv.exists():
            return csv
    return None


def _build_grid_from_discovery(csv_path: Path) -> dict:
    """Đọc CSV Phase 1 → tạo vùng order + seasonal_order (±1) cho grid search."""
    df = pd.read_csv(csv_path)
    valid = df.dropna(subset=["p"])
    if len(valid) == 0:
        typer.echo("Không có kết quả hợp lệ trong discovery CSV!", err=True)
        raise typer.Exit(1)

    # đếm (order, seasonal_order) phổ biến nhất → mở rộng ±1
    order_counter = Counter(zip(valid["p"].astype(int), valid["d"].astype(int), valid["q"].astype(int)))
    seasonal_counter = Counter(zip(valid["P"].astype(int), valid["D"].astype(int), valid["Q"].astype(int)))

    top_orders = order_counter.most_common(3)
    top_seasonal = seasonal_counter.most_common(3)
    typer.echo(f"Top orders: {top_orders}")
    typer.echo(f"Top seasonal: {top_seasonal}")

    # order: mở rộng ±1 quanh top values → tìm kiếm lân cận
    p_set, d_set, q_set = set(), set(), set()
    for (p, d, q), _ in top_orders:
        for delta in [-1, 0, 1]:
            p_set.add(max(0, p + delta))
            d_set.add(max(0, min(2, d + delta)))
            q_set.add(max(0, q + delta))

    # seasonal: chỉ dùng top values gốc (KHÔNG ±1) → tránh bùng nổ tổ hợp
    seasonal_tuples = [combo for combo, _ in seasonal_counter.most_common(3)]

    return {
        "p": sorted(p_set), "d": sorted(d_set), "q": sorted(q_set),
        "seasonal_tuples": seasonal_tuples,
        "trend": ["n", "c", "t", "ct"],
    }


@app.command()
def grid_search(
    discovery_path: str = typer.Option(None, "--discovery-path", help="Đường dẫn CSV Phase 1"),
    n_stores: int = typer.Option(20, "--n-stores", "-n", help="Số store eval NWRMSLE"),
    s: int = typer.Option(7, help="Chu kỳ mùa vụ (cố định)"),
):
    """Grid search SARIMAX quanh best orders từ Phase 1."""
    # tìm CSV Phase 1
    csv_path = Path(discovery_path) if discovery_path else _find_latest_discovery()
    if csv_path is None or not csv_path.exists():
        typer.echo("Không tìm thấy kết quả Phase 1. Chạy tune_sarimax_discovery.py trước!", err=True)
        raise typer.Exit(1)
    typer.echo(f"Đọc discovery từ: {csv_path}")

    space = _build_grid_from_discovery(csv_path)

    # tạo grid — order Cartesian (±1) × seasonal top values (gốc) × trend
    param_grid = [
        {"order": [p, d, q], "seasonal_order": [P, D, Q, s], "trend": trend}
        for p, d, q in itertools.product(space["p"], space["d"], space["q"])
        for (P, D, Q) in space["seasonal_tuples"]
        for trend in space["trend"]
    ]
    typer.echo(f"Grid: p={space['p']}, d={space['d']}, q={space['q']}, seasonal={space['seasonal_tuples']}, s={s}, trend={space['trend']}")
    typer.echo(f"Tổng tổ hợp: {len(param_grid)}")

    # load data
    config = load_config("sarimax")
    set_seed(config.get("seed", 42))
    typer.echo("Loading data...")
    df, _ = load_raw_data(config)
    df = filter_stores(df, config)
    df = add_all_features(df, feature_cfg=config.get("features", {}),
                          train_end=config.get("split", {}).get("train_end"))
    train_df, val_df, _, _ = preprocess(df, config)
    typer.echo(f"Data: {len(train_df)} train, {len(val_df)} val rows")

    # grid search — mỗi combo train SARIMAXModel riêng, eval NWRMSLE trên val set
    results = []
    for i, params in enumerate(param_grid, 1):
        logger.info(f"[{i}/{len(param_grid)}] order={params['order']}, seasonal={params['seasonal_order']}, trend={params['trend']}")
        cfg = copy.deepcopy(config)
        cfg["model"].update({"order": params["order"], "seasonal_order": params["seasonal_order"], "trend": params["trend"]})

        # base row chung cho cả success và failure
        base = {
            "order": str(params["order"]), "seasonal_order": str(params["seasonal_order"]),
            "p": params["order"][0], "d": params["order"][1], "q": params["order"][2],
            "P": params["seasonal_order"][0], "D": params["seasonal_order"][1], "Q": params["seasonal_order"][2],
            "trend": params["trend"],
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAXModel(cfg)
                start = time.time()
                model.train(train_df, val_df)
                elapsed = time.time() - start
                metrics = evaluate_all(
                    val_df["unit_sales"].values,
                    model.predict(val_df),
                    weights_from_frame(val_df),
                )
                results.append({**base, **metrics, "time_seconds": round(elapsed, 2)})
                logger.info(f"  NWRMSLE={metrics['nwrmsle']:.6f} ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"  FAILED: {e}")
            results.append({**base, "nwrmsle": float("inf"), "rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "rmspe": float("inf"), "time_seconds": 0})

    results_df = pd.DataFrame(results).sort_values("nwrmsle").reset_index(drop=True)

    # lưu kết quả
    out_dir = Path("results/sarimax/tuning") / f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "tuning_results.csv", index=False)
    (out_dir / "tuning_results.md").write_text(generate_tuning_report(results_df, model_name="SARIMAX"))

    best = results_df.iloc[0]
    best_params = {
        "order": [int(best["p"]), int(best["d"]), int(best["q"])],
        "seasonal_order": [int(best["P"]), int(best["D"]), int(best["Q"]), s],
        "trend": best["trend"],
    }
    with open(out_dir / "best_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    # biểu đồ
    for param in ["p", "d", "q", "P", "D", "Q", "trend"]:
        plot_param_sensitivity(results_df, param, save_path=str(out_dir / f"sensitivity_{param}.png"))
    plot_tuning_heatmap(results_df, "p", "q", save_path=str(out_dir / "heatmap_p_vs_q.png"))
    plot_tuning_heatmap(results_df, "P", "Q", save_path=str(out_dir / "heatmap_P_vs_Q.png"))
    plot_tuning_heatmap(results_df, "p", "trend", save_path=str(out_dir / "heatmap_p_vs_trend.png"))
    plot_top_k_comparison(results_df, k=min(10, len(results_df)), save_path=str(out_dir / "top10_comparison.png"))

    # in kết quả
    typer.echo(f"\n{'='*50}")
    typer.echo("KẾT QUẢ GRID SEARCH SARIMAX")
    typer.echo(f"{'='*50}")
    typer.echo(f"Best NWRMSLE: {best['nwrmsle']:.6f}")
    typer.echo(f"Best params: {best_params}")
    typer.echo(f"\nTop 5:")
    typer.echo(results_df[["order", "seasonal_order", "trend", "nwrmsle", "time_seconds"]].head().to_string(index=False))
    typer.echo(f"\nKết quả lưu tại: {out_dir}")


if __name__ == "__main__":
    app()
