"""Phase 1: auto_arima discovery cho SARIMAX — tìm order + seasonal_order tối ưu.

Chạy pmdarima auto_arima (seasonal=True, m=7) trên từng store
→ thống kê (order, seasonal_order) phổ biến nhất → lưu CSV.

Dùng:
    python scripts/tune_sarimax_discovery.py                 # 20 store mặc định
    python scripts/tune_sarimax_discovery.py --n-stores 50
"""

import logging
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from pmdarima import auto_arima

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.data.preprocessor import preprocess
from src.utils.config import load_config
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Phase 1: auto_arima discovery cho SARIMAX.")


@app.command()
def discover(
    n_stores: int = typer.Option(20, "--n-stores", "-n", help="Số store chạy auto_arima"),
    max_p: int = typer.Option(3, help="Bậc AR tối đa"),
    max_d: int = typer.Option(2, help="Bậc sai phân tối đa"),
    max_q: int = typer.Option(3, help="Bậc MA tối đa"),
    max_P: int = typer.Option(2, "--max-P", help="Bậc seasonal AR tối đa"),
    max_D: int = typer.Option(1, "--max-D", help="Bậc seasonal sai phân tối đa"),
    max_Q: int = typer.Option(2, "--max-Q", help="Bậc seasonal MA tối đa"),
    m: int = typer.Option(7, help="Chu kỳ mùa vụ (7=tuần)"),
):
    """Chạy auto_arima seasonal trên N store → tìm order + seasonal_order phổ biến nhất."""
    config = load_config("sarimax")
    set_seed(config.get("seed", 42))
    typer.echo("Loading data...")
    df, _ = load_raw_data(config)
    df = filter_stores(df, config)
    df = add_all_features(df, feature_cfg=config.get("features", {}),
                          train_end=config.get("split", {}).get("train_end"))
    train_df, _, _, _ = preprocess(df, config)

    stores = sorted(train_df["series_id"].unique())[:n_stores]
    typer.echo(f"Chạy auto_arima seasonal trên {len(stores)} store (m={m})...")

    # auto_arima per store — seasonal=True để tìm cả order lẫn seasonal_order
    results = []
    order_counter = Counter()
    for i, store_id in enumerate(stores, 1):
        store_data = train_df[train_df["series_id"] == store_id].sort_values("date")
        sales = store_data["unit_sales"].values.astype(float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = auto_arima(
                    sales,
                    max_p=max_p, max_d=max_d, max_q=max_q,
                    max_P=max_P, max_D=max_D, max_Q=max_Q,
                    m=m, seasonal=True, stepwise=True,
                    information_criterion="aic",
                    suppress_warnings=True, error_action="ignore",
                )
                order = model.order
                s_order = model.seasonal_order
                aic_val = model.aic()
                order_counter[(order, s_order)] += 1
                results.append({
                    "store": store_id,
                    "p": order[0], "d": order[1], "q": order[2],
                    "P": s_order[0], "D": s_order[1], "Q": s_order[2], "s": s_order[3],
                    "aic": round(aic_val, 2),
                })
                logger.info(f"[{i}/{len(stores)}] Store {store_id}: order={order}, seasonal={s_order}, AIC={aic_val:.2f}")
        except Exception as e:
            logger.warning(f"[{i}/{len(stores)}] Store {store_id}: FAILED — {e}")
            results.append({"store": store_id, "p": None, "d": None, "q": None, "P": None, "D": None, "Q": None, "s": None, "aic": None})

    # lưu CSV — Phase 2 sẽ đọc file này để build grid
    out_dir = Path("results/sarimax/tuning") / f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    csv_path = out_dir / "auto_arima_results.csv"
    results_df.to_csv(csv_path, index=False)

    # in tổng hợp
    typer.echo(f"\n{'='*50}")
    typer.echo("KẾT QUẢ AUTO_ARIMA DISCOVERY (SARIMAX)")
    typer.echo(f"{'='*50}")
    valid = results_df.dropna(subset=["p"])
    typer.echo(f"Thành công: {len(valid)}/{len(stores)} store")
    typer.echo(f"\nPhân phối (order, seasonal_order) — top 5:")
    for (order, s_order), count in order_counter.most_common(5):
        typer.echo(f"  {order} x {s_order}: {count} store ({count/len(valid)*100:.0f}%)")
    typer.echo(f"\nKết quả lưu tại: {csv_path}")


if __name__ == "__main__":
    app()
