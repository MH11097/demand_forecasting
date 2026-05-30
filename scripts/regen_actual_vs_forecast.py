"""Regenerate results/comparison/actual_vs_forecast.png.

Load 5 models mới nhất, predict trên 30-day test window, vẽ overlay.
KHÔNG đụng comparison_table hay error_comparison.
Tái sử dụng load_and_predict / aggregate_by_date / plot_actual_vs_forecast từ compare_30day.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_30day import (
    DISPLAY_NAMES,
    MODELS,
    TEST_END,
    TEST_START,
    aggregate_by_date,
    load_and_predict,
    plot_actual_vs_forecast,
)
from src.evaluation.metrics import evaluate_all


def main():
    print(f"Loading models & predicting on {TEST_START} → {TEST_END}...")
    predictions = {}
    for model_name in MODELS:
        display = DISPLAY_NAMES[model_name]
        print(f"  Loading {display}...")
        result = load_and_predict(model_name)
        if result is None:
            continue
        y_true, y_pred, dates, weights = result
        actual_agg, pred_agg, agg_dates = aggregate_by_date(y_true, y_pred, dates)
        predictions[display] = (actual_agg, pred_agg, agg_dates)
        m = evaluate_all(y_true, y_pred, weights)
        print(f"    ✓ {display}: NWRMSLE={m['nwrmsle']:.4f}, MAE={m['mae']:.1f}, RMSE={m['rmse']:.1f}")

    if not predictions:
        print("[WARN] Không load được model nào.")
        return

    plot_actual_vs_forecast(predictions)


if __name__ == "__main__":
    main()
