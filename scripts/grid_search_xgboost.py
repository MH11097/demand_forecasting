"""Grid search for XGBoost — train/test split (giống train.py), không CV mỗi combo."""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from src.data.features import add_all_features
from src.data.loader import load_raw_data, prepare_holdout_panel
from src.data.preprocessor import preprocess
from src.evaluation.metrics import evaluate_all, weights_from_frame
from src.models.xgboost_model import XGBoostModel
from src.utils.config import load_config
from src.utils.seed import set_seed

app = typer.Typer(help="Grid search XGBoost dùng train/test split (giống train.py).")


@app.command()
def grid_search(model_name: str = "xgboost"):
    """
    Grid search XGBoost — train trên train set, evaluate trên test set.

    Workflow: tìm best params bằng grid search → sau đó chạy CV 1 lần với best params.
    Tham số cố định: max_depth=9, subsample=1.0, colsample_bytree=0.8
    Tham số tìm kiếm: learning_rate, n_estimators, reg_alpha, reg_lambda
    """
    typer.echo("=== XGBoost Grid Search (train/test split) ===")

    config = load_config(model_name)
    set_seed(config.get("seed", 42))

    # Load data + features 1 lần, preprocess 1 lần → dùng chung cho mọi combo
    typer.echo("Loading data & features...")
    df, _ = load_raw_data(config)
    df = prepare_holdout_panel(df, config)
    df = add_all_features(
        df,
        feature_cfg=config.get("features", {}),
        train_end=config.get("split", {}).get("train_end"),
    )

    typer.echo("Preprocessing...")
    train_df, val_df, test_df, scaler = preprocess(df, config)
    typer.echo(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # Parameter grid — chỉ tune 4 tham số
    param_grid = {
        "learning_rate": [0.03, 0.05],
        "n_estimators": [500, 800, 1000],
        "reg_alpha": [0.5, 1, 1.5],
        "reg_lambda": [1.5, 2, 2.5],
    }

    param_names = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    n_combos = len(combinations)

    typer.echo(f"\nParameter Grid:")
    for param, values in param_grid.items():
        typer.echo(f"  {param}: {values}")
    typer.echo(f"Total combinations: {n_combos}")
    typer.echo("=" * 80)

    all_results = []
    start_total = time.time()

    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))

        # Tạo config cho combo này — tắt early_stopping vì đang tune n_estimators trực tiếp
        run_config = load_config(model_name)
        run_config["model"]["early_stopping_rounds"] = 0
        for k, v in params.items():
            run_config["model"][k] = v

        model = XGBoostModel(run_config)
        start = time.time()
        # train với val_df để early_stopping hoạt động (giống train.py)
        model.train(train_df, val_df if len(val_df) > 0 else None)
        preds = model.predict(test_df)
        y_true = test_df["unit_sales"].values
        metrics = evaluate_all(y_true, preds, weights_from_frame(test_df))
        elapsed = time.time() - start

        typer.echo(
            f"[{idx}/{n_combos}] NWRMSLE={metrics['nwrmsle']:.4f} | "
            f"lr={params['learning_rate']}, n_est={params['n_estimators']}, "
            f"alpha={params['reg_alpha']}, lambda={params['reg_lambda']} | {elapsed:.0f}s"
        )

        all_results.append({
            "params": params,
            "nwrmsle": metrics["nwrmsle"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "elapsed": elapsed,
        })

    total_elapsed = time.time() - start_total

    # Xếp hạng theo NWRMSLE
    all_results.sort(key=lambda x: x["nwrmsle"])
    for i, r in enumerate(all_results):
        r["rank"] = i + 1

    # Save results
    results_dir = (
        Path(config.get("results_dir", "results")) / "xgboost" / "grid_search_custom"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Top 10
    top_10 = all_results[:10]
    typer.echo("\n" + "=" * 80)
    typer.echo(f"Grid search completed in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    typer.echo(f"\nTop 10 (by NWRMSLE on test set):")

    top_10_rows = []
    for r in top_10:
        p = r["params"]
        typer.echo(
            f"  Rank {r['rank']}: NWRMSLE={r['nwrmsle']:.4f} | "
            f"lr={p['learning_rate']}, n_est={p['n_estimators']}, "
            f"alpha={p['reg_alpha']}, lambda={p['reg_lambda']}"
        )
        top_10_rows.append({
            "rank_test_score": r["rank"],
            "mean_test_score": round(r["nwrmsle"], 6),
            "param_learning_rate": p["learning_rate"],
            "param_n_estimators": p["n_estimators"],
            "param_reg_alpha": p["reg_alpha"],
            "param_reg_lambda": p["reg_lambda"],
        })

    pd.DataFrame(top_10_rows).to_csv(results_dir / "top_10_params.csv", index=False)

    # Best params
    best = all_results[0]
    with open(results_dir / "best_params.json", "w") as f:
        json.dump(best["params"], f, indent=2)

    # All runs
    all_runs_log = [{
        "rank": r["rank"],
        "params": r["params"],
        "nwrmsle": r["nwrmsle"],
        "rmse": round(r["rmse"], 2),
        "mae": round(r["mae"], 2),
        "mape": round(r["mape"], 6),
    } for r in all_results]
    with open(results_dir / "all_runs.json", "w") as f:
        json.dump(all_runs_log, f, indent=2)

    # Summary
    summary = {
        "total_combinations": n_combos,
        "best_nwrmsle": best["nwrmsle"],
        "best_params": best["params"],
        "training_time_seconds": total_elapsed,
        "data_shape": {
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "n_features": 18,
        },
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    typer.echo(f"\nResults saved to {results_dir}")
    typer.echo(f"\nNext: chạy CV với best params:")
    typer.echo(
        f"  python scripts/evaluate.py --model xgboost --cv expanding --n-splits 5 --eval-days 30 "
        f"--set model.learning_rate={best['params']['learning_rate']} "
        f"--set model.n_estimators={best['params']['n_estimators']} "
        f"--set model.reg_alpha={best['params']['reg_alpha']} "
        f"--set model.reg_lambda={best['params']['reg_lambda']}"
    )


if __name__ == "__main__":
    app()
