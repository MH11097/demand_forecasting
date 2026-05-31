"""Ablation study script - evaluate feature importance by iteratively adding features."""

import json
import time
from itertools import combinations
from pathlib import Path

import pandas as pd
import typer
import numpy as np

from src.data.features import add_all_features
from src.data.loader import load_cleaned_data, load_raw_data, prepare_holdout_panel
from src.data.preprocessor import preprocess
from src.evaluation.metrics import evaluate_all, weights_from_frame
from src.models import get_model_class
from src.utils.config import load_config
from src.utils.seed import set_seed

app = typer.Typer(help="Run ablation study to evaluate feature importance.")

# Base features to start with (Favorita schema: store/item ids + calendar)
BASE_FEATURES = [
    "store_nbr",
    "item_nbr",
    "dayofweek",
    "year",
    "month",
    "weekofyear",
    "dayofmonth",
    "is_weekend",
]

# Candidate feature pool (Favorita): exog + lag/rolling — XGBoost chọn động khi train,
# ablation thêm dần từng nhóm để đo đóng góp. Cột không có trong df sẽ bị _filter_features loại.
CANDIDATE_FEATURES = BASE_FEATURES + [
    "onpromotion",
    "perishable",
    "dcoilwtico",
    "is_holiday",
    "unit_sales_lag_1",
    "unit_sales_lag_7",
    "unit_sales_lag_14",
    "unit_sales_lag_30",
    "unit_sales_rolling_mean_7",
    "unit_sales_rolling_mean_14",
    "unit_sales_rolling_mean_30",
    "unit_sales_rolling_std_7",
    "unit_sales_rolling_std_14",
    "unit_sales_rolling_std_30",
]

# Additional features to test (these will be added incrementally to base)
ADDITIONAL_FEATURES = [col for col in CANDIDATE_FEATURES if col not in BASE_FEATURES]


def _filter_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """Filter feature list to only include columns that exist in dataframe."""
    return [f for f in feature_list if f in df.columns]


def _train_and_evaluate(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
) -> dict:
    """Train trên train_df, evaluate trên test_df riêng biệt (không data leakage)."""
    model_class = get_model_class(model_name)
    model = model_class(config)

    # Override feature list để chỉ dùng subset features đang test
    model._get_features = lambda df: df[feature_cols].fillna(0)
    model.feature_cols = feature_cols

    # Train trên train_df only — không truyền val để tránh leakage
    start = time.time()
    model.train(train_df)
    elapsed = time.time() - start

    # Evaluate trên test_df riêng biệt
    predictions = model.predict(test_df)
    y_true = test_df["unit_sales"].values
    metrics = evaluate_all(y_true, predictions, weights_from_frame(test_df))

    return {
        "n_features": len(feature_cols),
        "training_time": elapsed,
        "nwrmsle": metrics["nwrmsle"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "rmspe": metrics["rmspe"],
    }


@app.command()
def run(
    model_name: str = typer.Option("xgboost", help="Model to use for ablation study"),
    max_stores: int = typer.Option(
        None, help="Limit number of stores for faster testing"
    ),
    use_cleaned: bool = typer.Option(True, help="Use pre-cleaned data"),
    seed: int = typer.Option(42, help="Random seed"),
    eval_days: int = typer.Option(
        None, "--eval-days", help="Limit evaluation to first N days (default: None)"
    ),
):
    """Run ablation study starting from base features."""
    set_seed(seed)

    # Load config — tắt early_stopping vì không dùng val set
    config = load_config(model_name, {})
    config["model"]["early_stopping_rounds"] = 0
    if max_stores is not None:
        config["max_stores"] = max_stores

    typer.echo(f"Loading data... (use_cleaned={use_cleaned})")
    if use_cleaned:
        df, _ = load_cleaned_data(config)
    else:
        df, _ = load_raw_data(config)
    df = prepare_holdout_panel(df, config)

    # Add all features
    typer.echo("Adding features...")
    df = add_all_features(
        df,
        feature_cfg=config.get("features", {}),
        train_end=config.get("split", {}).get("train_end"),
    )

    # Preprocess
    typer.echo("Preprocessing...")
    train_df, val_df, test_df, scaler = preprocess(df, config)

    # Evaluate trên test_df — tách biệt hoàn toàn khỏi train
    typer.echo(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # Filter available features
    available_base = _filter_features(df, BASE_FEATURES)
    available_additional = _filter_features(df, ADDITIONAL_FEATURES)

    typer.echo(f"\nBase features ({len(available_base)}): {available_base}")
    typer.echo(
        f"Additional features to test ({len(available_additional)}): {available_additional}"
    )

    results = []

    # Test 1: Base features only
    typer.echo("\n" + "=" * 80)
    typer.echo(f"Training with BASE features only...")
    typer.echo(f"Features: {available_base}")

    try:
        metrics = _train_and_evaluate(
            model_name, train_df, test_df, available_base, config
        )
        result = {
            "combination_id": 0,
            "features": available_base,
            "n_total_features": len(available_base),
            "added_feature": "none",
            **metrics,
        }
        results.append(result)
        typer.echo(
            f"✓ NWRMSLE: {metrics['nwrmsle']:.6f} | Time: {metrics['training_time']:.1f}s"
        )
    except Exception as e:
        typer.echo(f"✗ Failed: {e}", err=True)

    # Test 2: Incrementally add features one by one
    typer.echo("\n" + "=" * 80)
    typer.echo(f"Testing INCREMENTAL feature additions...")

    current_features = available_base.copy()

    for idx, feature in enumerate(available_additional, 1):
        if feature not in current_features:
            current_features.append(feature)
        else:
            continue

        typer.echo(
            f"\n[{idx}/{len(available_additional)}] Adding '{feature}' (now {len(current_features)} features)..."
        )

        try:
            metrics = _train_and_evaluate(
                model_name, train_df, test_df, current_features, config
            )
            result = {
                "combination_id": idx,
                "features": current_features.copy(),
                "n_total_features": len(current_features),
                "added_feature": feature,
                **metrics,
            }
            results.append(result)
            typer.echo(
                f"✓ NWRMSLE: {metrics['nwrmsle']:.6f} | Time: {metrics['training_time']:.1f}s"
            )
        except Exception as e:
            typer.echo(f"✗ Failed: {e}", err=True)

    # Save results
    results_dir = Path(config.get("results_dir", "results")) / "ablation_study"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Sort by NWRMSLE
    sorted_results = sorted(results, key=lambda x: x["nwrmsle"])

    # Save detailed results as JSON
    output_json = results_dir / "ablation_results_detailed.json"
    with open(output_json, "w") as f:
        json.dump(sorted_results, f, indent=2)
    typer.echo(f"\n✓ Detailed results saved to {output_json}")

    # Create summary CSV
    summary_data = []
    for result in sorted_results:
        summary_data.append(
            {
                "combination_id": result["combination_id"],
                "added_feature": result["added_feature"],
                "n_features": result["n_total_features"],
                "nwrmsle": result["nwrmsle"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
                "rmspe": result["rmspe"],
                "time_s": result["training_time"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_csv = results_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    typer.echo(f"✓ Summary saved to {summary_csv}")

    # Print summary table
    typer.echo("\n" + "=" * 80)
    typer.echo("ABLATION STUDY RESULTS (sorted by NWRMSLE):")
    typer.echo("=" * 80)
    print(summary_df.to_string(index=False))

    # Print best combination
    best = sorted_results[0]
    typer.echo("\n" + "=" * 80)
    typer.echo("BEST COMBINATION:")
    typer.echo("=" * 80)
    typer.echo(f"Added feature: {best['added_feature']}")
    typer.echo(f"Total features: {best['n_total_features']}")
    typer.echo(f"NWRMSLE: {best['nwrmsle']:.6f}")
    typer.echo(f"Features: {best['features']}")

    # Calculate improvements over base
    base_nwrmsle = sorted_results[0]["nwrmsle"]  # Base features performance
    typer.echo("\n" + "=" * 80)
    typer.echo("FEATURE IMPACT (improvement over base):")
    typer.echo("=" * 80)
    for result in sorted_results[1:]:
        improvement = ((base_nwrmsle - result["nwrmsle"]) / base_nwrmsle) * 100
        typer.echo(
            f"+{result['added_feature']:35s} | NWRMSLE Δ: {improvement:+7.2f}% | "
            f"NWRMSLE: {result['nwrmsle']:.6f}"
        )


if __name__ == "__main__":
    app()
