"""Phase 1: Coarse Grid Search cho Prophet — quét 90 tổ hợp tham số vùng rộng.

Mục đích: tìm vùng tham số có NWRMSLE thấp, sau đó Phase 2 sẽ tinh chỉnh quanh vùng này.
Output: results/prophet/tuning/coarse_<timestamp>/ chứa CSV, MD report, YAML best params, biểu đồ.

Usage:
    python scripts/tune-prophet-coarse.py
"""

import argparse
import logging
import warnings

import yaml

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.data.preprocessor import preprocess
from src.tuning.grid_search import generate_param_grid, run_grid_search
from src.tuning.tuning_viz import (
    generate_tuning_report,
    plot_param_sensitivity,
    plot_top_k_comparison,
    plot_tuning_heatmap,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Prophet coarse grid search")
    return parser.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # === CONFIG ===
    config = load_config("prophet")
    set_seed(config.get("seed", 42))
    # === LOAD DATA 1 LẦN ===
    df, _ = load_raw_data(config)
    df = filter_stores(df, config)
    df = add_all_features(df)
    train_df, val_df, _, _ = preprocess(df, config)
    print(f"Data: {len(train_df)} train, {len(val_df)} val rows")

    # === COARSE GRID: 5×2×3×3 = 90 tổ hợp ===
    coarse_space = {
        "changepoint_prior_scale": [0.001, 0.01, 0.05, 0.1, 0.5],
        "seasonality_mode": ["additive", "multiplicative"],
        "seasonality_prior_scale": [0.1, 1.0, 10.0],
        "holidays_prior_scale": [0.1, 1.0, 10.0],
    }
    param_grid = generate_param_grid(coarse_space)
    print(f"Tổng tổ hợp: {len(param_grid)}")

    results_df = run_grid_search(config, param_grid, train_df, val_df)

    # === LƯU KẾT QUẢ ===
    from datetime import datetime
    from pathlib import Path

    out_dir = Path("results/prophet/tuning/coarse_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "tuning_results.csv", index=False)
    (out_dir / "tuning_results.md").write_text(generate_tuning_report(results_df))

    # Best params → YAML để Phase 2 đọc tự động
    best = results_df.iloc[0]
    best_params = {k: best[k].item() if hasattr(best[k], "item") else best[k] for k in coarse_space.keys()}
    with open(out_dir / "best_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    # Biểu đồ sensitivity + heatmap + top-K
    for param in coarse_space.keys():
        plot_param_sensitivity(results_df, param, save_path=str(out_dir / f"sensitivity_{param}.png"))
    plot_tuning_heatmap(results_df, "changepoint_prior_scale", "seasonality_mode", save_path=str(out_dir / "heatmap_cps_vs_mode.png"))
    plot_tuning_heatmap(results_df, "changepoint_prior_scale", "seasonality_prior_scale", save_path=str(out_dir / "heatmap_cps_vs_sps.png"))
    plot_tuning_heatmap(results_df, "seasonality_prior_scale", "holidays_prior_scale", save_path=str(out_dir / "heatmap_sps_vs_hps.png"))
    plot_top_k_comparison(results_df, k=min(10, len(results_df)), save_path=str(out_dir / "top10_comparison.png"))

    print(f"\nBest NWRMSLE: {best['nwrmsle']:.6f}")
    print(f"Best params: {best_params}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
