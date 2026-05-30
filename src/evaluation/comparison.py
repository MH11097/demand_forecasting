"""Utilities for comparing model results."""

import json
from pathlib import Path

import pandas as pd


def load_results(results_dir: str, model_name: str | None = None) -> list[dict]:
    """Scan results directory for result.json files and load them.

    Args:
        results_dir: Root results directory.
        model_name: If provided, only load results for this model.
    """
    # quét tất cả result.json trong results/ -> thu thập kết quả từ nhiều lần chạy để so sánh
    results = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return results

    # nếu chỉ định model -> thu hẹp phạm vi tìm, tránh load kết quả model khác
    search_path = results_path / model_name if model_name else results_path
    if not search_path.exists():
        return results

    for result_file in search_path.rglob("result.json"):
        with open(result_file) as f:
            data = json.load(f)
        if model_name and data.get("model_name") != model_name:
            continue
        results.append(data)
    return results


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from list of result dicts."""
    # gom tất cả run vào 1 bảng, sort theo NWRMSLE -> dễ nhìn model nào tốt nhất
    rows = []
    for r in results:
        row = {
            "model_name": r["model_name"],
            "run_id": r["run_id"],
            "param_slug": r.get("param_slug", ""),
            "experiment_name": r.get("experiment_name", ""),
            "training_time_seconds": r.get("training_time_seconds", 0),
        }
        # Prefer the final holdout score. Validation can be absent in the default
        # config and must not silently rank as a perfect 0.0 run.
        reported = r.get("test_metrics") or r.get("metrics", {})
        row["evaluation_split"] = "test" if r.get("test_metrics") else "validation"
        row.update(reported)
        rows.append(row)

    df = pd.DataFrame(rows)
    # ưu tiên sort theo NWRMSLE (primary); fallback rmse nếu thiếu
    for key in ("nwrmsle", "rmse"):
        if key in df.columns:
            df = df.sort_values(key).reset_index(drop=True)
            break
    return df
