"""Randomized search for XGBoost hyperparameter optimization with time series validation."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.data.features import add_all_features
from src.data.loader import load_cleaned_data, load_raw_data
from src.data.preprocessor import preprocess
from src.evaluation.metrics import nwrmsle
from src.utils.config import load_config
from src.utils.seed import set_seed

app = typer.Typer(help="RandomizedSearchCV for XGBoost hyperparameter tuning.")

# Features for XGBoost model
FEATURE_COLS = [
    "store_nbr",
    "item_nbr",
    "onpromotion",
    "perishable",
    "dcoilwtico",
    "is_holiday",
    "dayofweek",
    "year",
    "month",
    "weekofyear",
    "dayofmonth",
    "is_weekend",
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


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract available features from dataframe."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].fillna(0)


@app.command()
def randomized_search(
    model_name: str = "xgboost",
    use_cleaned: bool = True,
    cv_splits: int = 5,
    n_jobs: int = -1,
    quick: bool = False,
    n_iter: int | None = None,
    max_stores: int | None = None,
):
    """
    Run RandomizedSearchCV for XGBoost.

    Args:
        model_name: Model name (default: xgboost)
        use_cleaned: Use pre-cleaned data (default: True)
        cv_splits: Number of cross-validation splits (default: 5)
        n_jobs: Number of parallel jobs (-1 = all cores, default: -1)
        quick: Quick search with fewer parameter combinations (default: False)
        n_iter: Number of random combinations to try (default: auto-determined)
        max_stores: Limit number of stores for faster testing (default: None)
    """
    typer.echo(
        "randomized_search_xgboost.py đã tắt: sklearn TimeSeriesSplit trên panel "
        "store-item không tạo rolling-origin folds đúng. Dùng grid_search_xgboost.py "
        "và `python scripts/evaluate.py --model xgboost --cv expanding`.",
        err=True,
    )
    raise typer.Exit(1)

    typer.echo("=== RandomizedSearchCV for XGBoost ===")

    # Load configuration and data
    config = load_config(model_name)
    seed = config.get("seed", 42)
    set_seed(seed)

    typer.echo("Loading data...")
    if use_cleaned:
        df, _ = load_cleaned_data(config)
    else:
        df, _ = load_raw_data(config)

    typer.echo("Adding features...")
    df = add_all_features(df)

    typer.echo("Preprocessing...")
    train_df, val_df, test_df, scaler = preprocess(df, config)

    # Combine train and val for GridSearchCV
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    X = get_features(combined_df)
    y = combined_df["unit_sales"].values

    typer.echo(f"Data shape: X={X.shape}, y={y.shape}")

    # Define parameter grid
    if quick:
        param_grid = {
            "max_depth": [5, 7, 9],
            "learning_rate": [0.05, 0.1, 0.2],
            "n_estimators": [500, 1000, 1500],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0, 0.5],
            "reg_lambda": [1.0, 1.5],
        }
        # Auto-determine n_iter if not provided
        if n_iter is None:
            n_iter = 20  # Quick: test 20 random combinations
        typer.echo(f"Quick search: {n_iter} random combinations")
    else:
        param_grid = {
            "max_depth": [5, 7, 9, 11],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [500, 1000, 1500],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 1.0],
            "reg_lambda": [0.5, 2.0],
        }
        # Auto-determine n_iter if not provided
        if n_iter is None:
            n_iter = 50  # Full: test 50 random combinations
        typer.echo(f"Full search: {n_iter} random combinations")

    # Create base model
    base_model = xgb.XGBRegressor(random_state=seed, n_jobs=1)

    # Custom scoring function using NWRMSLE
    # sklearn scorer signature: scorer(estimator, X_test, y_test)
    def nwrmsle_scorer(estimator, X_test, y_test):
        y_pred = estimator.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)  # Ensure non-negative predictions
        return -nwrmsle(
            y_test, y_pred
        )  # Negative because sklearn expects higher is better

    # Create time series cross-validator (replaces random splits)
    ts_cv = TimeSeriesSplit(n_splits=cv_splits)

    typer.echo("\n" + "=" * 80)
    typer.echo("TIME SERIES RANDOMIZEDSEARCHCV (Walk-Forward Validation)")
    typer.echo("=" * 80)

    # Calculate total parameter combinations (for reference)
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    typer.echo(f"Total possible combinations: {total_combos}")
    typer.echo(f"Testing with n_iter: {n_iter} random combinations")
    typer.echo(f"CV splits (TimeSeriesSplit): {cv_splits}")
    typer.echo(f"Total model evaluations: {n_iter * cv_splits}\n")

    # Run RandomizedSearchCV with time series split
    typer.echo(f"Running RandomizedSearchCV with TimeSeriesSplit...")
    start_time = time.time()

    random_search_cv = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,  # Number of random combinations to try
        cv=ts_cv,  # Time series split instead of random K-fold
        scoring=nwrmsle_scorer,
        n_jobs=n_jobs,
        random_state=seed,  # For reproducibility
        verbose=2,  # Verbose output to see progress
    )

    random_search_cv.fit(X, y)
    elapsed = time.time() - start_time

    # Results
    typer.echo(f"\n" + "=" * 80)
    typer.echo(
        f"RandomizedSearchCV completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)"
    )
    typer.echo(f"Best score (NWRMSLE): {-random_search_cv.best_score_:.4f}")
    typer.echo(
        f"Best parameters:\n{json.dumps(random_search_cv.best_params_, indent=2)}"
    )
    typer.echo("=" * 80)

    # Save detailed results
    results_dir = (
        Path(config.get("results_dir", "results")) / "xgboost" / "randomized_search"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters
    best_params_path = results_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(random_search_cv.best_params_, f, indent=2)
    typer.echo(f"Best parameters saved to {best_params_path}")

    # Save CV results with run indexing
    cv_results_df = pd.DataFrame(random_search_cv.cv_results_)

    # Add run index for each parameter combination
    cv_results_df.insert(0, "run_index", range(1, len(cv_results_df) + 1))

    cv_results_path = results_dir / "cv_results.csv"
    cv_results_df.to_csv(cv_results_path, index=False)
    typer.echo(f"CV results saved to {cv_results_path}")

    # Log each run with details
    all_runs_log = []
    for idx, row in cv_results_df.iterrows():
        run_entry = {
            "run_index": int(row["run_index"]),
            "params": {
                k.replace("param_", ""): row[k]
                for k in row.index
                if k.startswith("param_")
            },
            "mean_test_score": float(-row["mean_test_score"]),  # Convert back to NWRMSLE
            "std_test_score": float(row["std_test_score"]),
            "mean_train_score": float(-row["mean_train_score"]),
            "rank_test_score": int(row["rank_test_score"]),
        }
        all_runs_log.append(run_entry)

    # Save all runs as JSON
    all_runs_path = results_dir / "all_runs.json"
    with open(all_runs_path, "w") as f:
        json.dump(all_runs_log, f, indent=2)
    typer.echo(f"All runs log saved to {all_runs_path}")

    # Display all runs with index in console
    typer.echo(f"\nAll {len(all_runs_log)} runs (indexed):")
    typer.echo("-" * 100)
    for run in all_runs_log[: min(20, len(all_runs_log))]:  # Show first 20
        typer.echo(
            f"[Run {run['run_index']}] NWRMSLE: {run['mean_test_score']:.4f} | Params: {run['params']}"
        )
    if len(all_runs_log) > 20:
        typer.echo(f"... ({len(all_runs_log) - 20} more runs)")
    typer.echo("-" * 100)

    # Top 10 results
    top_10_idx = np.argsort(random_search_cv.cv_results_["rank_test_score"])[:10]
    top_10 = cv_results_df.iloc[top_10_idx][
        [
            "run_index",
            "param_max_depth",
            "param_learning_rate",
            "param_n_estimators",
            "param_subsample",
            "param_colsample_bytree",
            "mean_test_score",
        ]
    ].copy()
    top_10["mean_test_score"] = -top_10["mean_test_score"]  # Convert back to NWRMSLE

    typer.echo("\nTop 10 parameter combinations (by NWRMSLE):")
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        typer.echo(
            f"[Run {int(row['run_index'])}] Rank {idx}: NWRMSLE={row['mean_test_score']:.4f}"
        )

    top_10_path = results_dir / "top_10_params.csv"
    top_10.to_csv(top_10_path, index=False)
    typer.echo(f"Top 10 saved to {top_10_path}")

    # Summary
    summary = {
        "total_combinations": len(random_search_cv.cv_results_["params"]),
        "best_nwrmsle": float(-random_search_cv.best_score_),
        "best_params": random_search_cv.best_params_,
        "cv_splits": cv_splits,
        "training_time_seconds": elapsed,
        "data_shape": {"n_samples": X.shape[0], "n_features": X.shape[1]},
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    app()
