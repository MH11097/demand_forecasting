"""Phase 3: Ablation Study cho Prophet — đo đóng góp biên từng regressor.

Cố định best hyperparams từ Phase 2, thay đổi tổ hợp regressors (6 thí nghiệm).
Tự động đọc best_params.yaml từ results/prophet/tuning/fine_* (thư mục mới nhất).
Nếu không tìm thấy fine → thử đọc từ coarse_*.

Usage:
    python scripts/tune-prophet-ablation.py
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
from src.tuning.grid_search import run_ablation_study
from src.tuning.tuning_viz import generate_tuning_report, plot_ablation_results
from src.utils.config import load_config
from src.utils.seed import set_seed

# Giá trị mặc định nếu không tìm thấy best_params.yaml
DEFAULT_BEST = {
    "changepoint_prior_scale": 0.05,
    "n_changepoints": 25,
    "changepoint_range": 0.8,
    "seasonality_mode": "multiplicative",
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prophet ablation study")
    return parser.parse_args()


def _find_latest_best_params() -> dict | None:
    """Tìm best_params.yaml mới nhất: ưu tiên fine_* → fallback coarse_*.

    Phase 2 (fine) có đầy đủ 6 tham số (gồm changepoint_range, n_changepoints).
    Phase 1 (coarse) chỉ có 4 tham số cơ bản → thiếu 2 tham số sẽ dùng default.
    """
    tuning_dir = Path("results/prophet/tuning")
    # Ưu tiên fine trước, rồi mới coarse
    for prefix in ["fine_*", "coarse_*"]:
        dirs = sorted(tuning_dir.glob(prefix), reverse=True)
        for d in dirs:
            yaml_path = d / "best_params.yaml"
            if yaml_path.exists():
                with open(yaml_path) as f:
                    params = yaml.safe_load(f)
                print(f"Đọc best params từ: {yaml_path}")
                return params
    return None


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # === CONFIG ===
    config = load_config("prophet")
    set_seed(config.get("seed", 42))
    # store_type đã được set trong config (type "c" cho prophet)

    # === BEST PARAMS TỪ PHASE 2 (hoặc Phase 1 fallback) ===
    best_params = _find_latest_best_params()
    if best_params is None:
        print("WARNING: Không tìm thấy best_params.yaml → dùng giá trị default")
        best_params = DEFAULT_BEST
    print(f"Best params: {best_params}")

    # Ghi đè tham số model trong config
    for key, value in best_params.items():
        config.setdefault("model", {})[key] = value

    # === LOAD DATA ===
    df, _ = load_raw_data(config)
    df = filter_stores(df, config)
    df = add_all_features(df, feature_cfg=config.get("features", {}),
                          train_end=config.get("split", {}).get("train_end"))
    train_df, val_df, _, _ = preprocess(df, config)
    print(f"Data: {len(train_df)} train, {len(val_df)} val rows")

    # === ABLATION: 6 tổ hợp regressors ===
    ablation_combos = {
        "Baseline (no regressor)": [],
        "+Promo": ["Promo"],
        "+SchoolHoliday": ["SchoolHoliday"],
        "+StateHoliday": ["StateHoliday"],
        "Promo+SchoolHoliday": ["Promo", "SchoolHoliday"],
        "Full (all)": ["Promo", "SchoolHoliday", "StateHoliday"],
    }

    ablation_df = run_ablation_study(config, ablation_combos, train_df, val_df)

    # === LƯU KẾT QUẢ ===
    out_dir = Path("results/prophet/tuning/ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_df.to_csv(out_dir / "tuning_results.csv", index=False)
    (out_dir / "tuning_results.md").write_text(generate_tuning_report(ablation_df))
    plot_ablation_results(ablation_df, save_path=str(out_dir / "ablation_comparison.png"))

    # In kết quả
    print("\nKết quả Ablation Study:")
    print(ablation_df[["experiment", "regressors", "nwrmsle", "rmse"]].to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
