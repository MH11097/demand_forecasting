"""Phase 2: Fine Grid Search cho Prophet — tinh chỉnh quanh vùng tốt nhất từ Phase 1.

Tự động đọc best_params.yaml từ results/prophet/tuning/coarse_* (thư mục mới nhất).
Nếu không tìm thấy → dùng giá trị default + in warning.
Thêm 2 tham số mới: changepoint_range, n_changepoints.

Usage:
    python scripts/tune-prophet-fine.py
"""

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path

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

# Giá trị mặc định nếu không tìm thấy best_params.yaml từ Phase 1
DEFAULT_BEST = {
    "changepoint_prior_scale": 0.05,
    "seasonality_mode": "multiplicative",
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prophet fine grid search")
    return parser.parse_args()


def _find_latest_best_params() -> dict | None:
    """Tìm file best_params.yaml mới nhất trong results/prophet/tuning/coarse_*/.

    Sắp theo tên thư mục giảm dần (timestamp trong tên → thư mục mới nhất đứng đầu).
    """
    tuning_dir = Path("results/prophet/tuning")
    coarse_dirs = sorted(tuning_dir.glob("coarse_*"), reverse=True)
    for d in coarse_dirs:
        yaml_path = d / "best_params.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                params = yaml.safe_load(f)
            print(f"Đọc best params từ Phase 1: {yaml_path}")
            return params
    return None


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # === CONFIG ===
    config = load_config("prophet")
    set_seed(config.get("seed", 42))
    # === LOAD DATA ===
    df, _ = load_raw_data(config)
    df = filter_stores(df, config)
    df = add_all_features(df)
    train_df, val_df, _, _ = preprocess(df, config)
    print(f"Data: {len(train_df)} train, {len(val_df)} val rows")

    # === BEST PARAMS TỪ PHASE 1 ===
    best_from_coarse = _find_latest_best_params()
    if best_from_coarse is None:
        print("WARNING: Không tìm thấy best_params.yaml từ Phase 1 → dùng giá trị default")
        best_from_coarse = DEFAULT_BEST
    print(f"Best params Phase 1: {best_from_coarse}")

    best_cps = best_from_coarse["changepoint_prior_scale"]
    best_mode = best_from_coarse["seasonality_mode"]
    best_sps = best_from_coarse["seasonality_prior_scale"]
    best_hps = best_from_coarse["holidays_prior_scale"]

    # Fine grid: xoay quanh best CPS (×0.25 → ×4) + thêm changepoint_range, n_changepoints
    fine_space = {
        "changepoint_prior_scale": sorted({best_cps * f for f in [0.25, 0.5, 1.0, 2.0, 4.0]}),
        "seasonality_mode": [best_mode],
        "seasonality_prior_scale": [best_sps],
        "holidays_prior_scale": [best_hps],
        "changepoint_range": [0.8, 0.85, 0.9, 0.95],
        "n_changepoints": [15, 25, 35],
    }
    param_grid = generate_param_grid(fine_space)
    print(f"Tổng tổ hợp: {len(param_grid)}")

    results_df = run_grid_search(config, param_grid, train_df, val_df)

    # === LƯU KẾT QUẢ ===
    out_dir = Path("results/prophet/tuning/fine_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "tuning_results.csv", index=False)
    (out_dir / "tuning_results.md").write_text(generate_tuning_report(results_df))

    # Best params → YAML để Phase 3 đọc tự động
    best = results_df.iloc[0]
    best_params = {k: best[k].item() if hasattr(best[k], "item") else best[k] for k in fine_space.keys()}
    with open(out_dir / "best_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    # Biểu đồ cho 3 tham số biến đổi + heatmap tương tác
    for param in ["changepoint_prior_scale", "changepoint_range", "n_changepoints"]:
        plot_param_sensitivity(results_df, param, save_path=str(out_dir / f"sensitivity_{param}.png"))
    plot_tuning_heatmap(results_df, "changepoint_prior_scale", "changepoint_range", save_path=str(out_dir / "heatmap_cps_vs_cr.png"))
    plot_tuning_heatmap(results_df, "changepoint_range", "n_changepoints", save_path=str(out_dir / "heatmap_cr_vs_ncp.png"))
    plot_top_k_comparison(results_df, k=min(10, len(results_df)), save_path=str(out_dir / "top10_comparison.png"))

    print(f"\nBest NWRMSLE: {best['nwrmsle']:.6f}")
    print(f"Best params: {best_params}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
