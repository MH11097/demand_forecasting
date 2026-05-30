"""Phase 1: auto_arima discovery — tìm order (p,d,q) tối ưu cho ARIMA trên N store.

Chạy pmdarima auto_arima (AIC stepwise) trên từng store riêng lẻ
→ thống kê order phổ biến nhất → lưu CSV kết quả.

Dùng:
    python scripts/tune_arima_discovery.py                 # 20 store mặc định
    python scripts/tune_arima_discovery.py --n-stores 50   # 50 store
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

app = typer.Typer(help="Phase 1: auto_arima discovery cho ARIMA.")


@app.command()
def discover(
    n_stores: int = typer.Option(20, "--n-stores", "-n", help="Số store chạy auto_arima"),
    max_p: int = typer.Option(5, help="Bậc AR tối đa"),
    max_d: int = typer.Option(2, help="Bậc sai phân tối đa"),
    max_q: int = typer.Option(5, help="Bậc MA tối đa"),
):
    """Chạy auto_arima trên N store → tìm order phổ biến nhất."""
    # load data giống pipeline train: raw → sample → features → preprocess
    config = load_config("arima")
    set_seed(config.get("seed", 42))
    typer.echo("Loading data...")
    df, _ = load_raw_data(config)
    df = filter_stores(df, config)
    df = add_all_features(df, feature_cfg=config.get("features", {}),
                          train_end=config.get("split", {}).get("train_end"))
    train_df, _, _, _ = preprocess(df, config)

    stores = sorted(train_df["series_id"].unique())[:n_stores]
    typer.echo(f"Chạy auto_arima trên {len(stores)} store (max_p={max_p}, max_d={max_d}, max_q={max_q})...")

    # chạy auto_arima per store — mỗi store có pattern riêng nên order có thể khác nhau
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
                    seasonal=False, stepwise=True,
                    information_criterion="aic",
                    suppress_warnings=True, error_action="ignore",
                )
                order = model.order
                aic_val = model.aic()
                order_counter[order] += 1
                results.append({"store": store_id, "p": order[0], "d": order[1], "q": order[2], "aic": round(aic_val, 2)})
                logger.info(f"[{i}/{len(stores)}] Store {store_id}: order={order}, AIC={aic_val:.2f}")
        except Exception as e:
            logger.warning(f"[{i}/{len(stores)}] Store {store_id}: FAILED — {e}")
            results.append({"store": store_id, "p": None, "d": None, "q": None, "aic": None})

    # lưu kết quả CSV — Phase 2 sẽ đọc file này để tạo grid search
    out_dir = Path("results/arima/tuning") / f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    csv_path = out_dir / "auto_arima_results.csv"
    results_df.to_csv(csv_path, index=False)

    # in tổng hợp — order nào xuất hiện nhiều nhất = ứng viên global mạnh nhất
    typer.echo(f"\n{'='*50}")
    typer.echo("KẾT QUẢ AUTO_ARIMA DISCOVERY")
    typer.echo(f"{'='*50}")
    valid = results_df.dropna(subset=["p"])
    typer.echo(f"Thành công: {len(valid)}/{len(stores)} store")
    typer.echo(f"\nPhân phối order (top 5):")
    for order, count in order_counter.most_common(5):
        typer.echo(f"  {order}: {count} store ({count/len(valid)*100:.0f}%)")
    typer.echo(f"\nKết quả lưu tại: {csv_path}")


if __name__ == "__main__":
    app()
