"""CLI script for evaluating trained models."""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import typer

from src.data.features import add_all_features, apply_log_transform
from src.data.loader import load_cleaned_data, prepare_holdout_panel, select_series
from src.data.preprocessor import preprocess
from src.evaluation.cross_validation import walk_forward_cv
from src.evaluation.metrics import evaluate_all, weights_from_frame
from src.models import get_model_class
from src.models.base import BaseModel
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.visualization import plot_predictions, plot_residuals
from scripts.train import _parse_overrides

app = typer.Typer(help="Evaluate trained forecasting models.")


@app.command()
def evaluate(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    set_values: list[str] = typer.Option(
        None, "--set", "-s", help="Override config key=value"
    ),
    run_dir: str = typer.Option(None, "--run-dir", help="Path to saved run directory"),
    cv: str = typer.Option(None, "--cv", help="CV strategy: 'expanding' or 'sliding'"),
    n_splits: int = typer.Option(5, "--n-splits", help="Number of CV folds"),
    eval_days: int = typer.Option(
        None, "--eval-days", help="Limit evaluation to first N days (default: None)"
    ),
):
    """Evaluate a model on validation/test data or run cross-validation."""
    overrides = _parse_overrides(set_values)
    config = load_config(model, overrides)
    set_seed(config.get("seed", 42))

    if cv:
        typer.echo(f"Running {cv} walk-forward CV with {n_splits} splits...")
        df, _ = load_cleaned_data(config)
        # chọn mẫu đại diện chuỗi cho per-series models (theo series_sample config)
        df = select_series(df, config)
        if run_dir:
            typer.echo(
                "CV với --run-dir không hợp lệ: saved model đã thấy tương lai của các "
                "fold lịch sử. Bỏ --run-dir để retrain theo từng fold.",
                err=True,
            )
            raise typer.Exit(1)
        else:
            # CV chuẩn: retrain từ đầu mỗi fold (walk-forward thực sự)
            model_class = get_model_class(model)
            cv_results = walk_forward_cv(
                model_class,
                config,
                df,
                n_splits=n_splits,
                expanding=(cv == "expanding"),
                eval_days=eval_days,
            )

        typer.echo(f"CV Results ({n_splits} folds):")
        for fold in cv_results["folds"]:
            typer.echo(f"  Fold {fold['fold']}: NWRMSLE={fold['nwrmsle']:.4f}")
        typer.echo(f"Aggregated: {cv_results['aggregated']}")

        # Save CV results
        results_dir = Path(config.get("results_dir", "results")) / model / "cv"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=2)
        typer.echo(f"CV results saved to {results_dir}")
        return

    # load model đã train từ trước -> không cần train lại, chỉ predict và đánh giá
    if run_dir:
        model_path = Path(run_dir) / "model.pkl"
    else:
        # không chỉ định run cụ thể -> lấy run mới nhất (sort theo tên thư mục = timestamp)
        results_base = Path(config.get("results_dir", "results")) / model
        if not results_base.exists():
            typer.echo(f"No results found for {model}", err=True)
            raise typer.Exit(1)
        runs = sorted(results_base.iterdir())
        run_dir = str(runs[-1])
        model_path = Path(run_dir) / "model.pkl"

    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading model from {model_path}")
    loaded_model = BaseModel.load(str(model_path))

    # Load and preprocess data
    df, _ = load_cleaned_data(config)
    # chọn mẫu đại diện chuỗi cho per-series models (theo series_sample config)
    df = select_series(df, config)
    df = prepare_holdout_panel(df, config)
    # train_end → group-mean encodings train-only (chống leakage)
    df = add_all_features(
        df,
        feature_cfg=config.get("features", {}),
        train_end=config.get("split", {}).get("train_end"),
    )
    if config.get("use_log_sales", False):
        typer.echo("Applying log1p transform to Sales and derived features...")
        df = apply_log_transform(df)
    train_df, val_df, test_df, _ = preprocess(df, config)

    use_log = config.get("use_log_sales", False)

    def _get_true_sales(split_df):
        """Trả về sales gốc (inverse log nếu cần) để tính metrics đúng scale."""
        s = split_df["unit_sales"].values.astype(float)
        return np.expm1(s) if use_log else s

    # Prepend context (seq_len + H - 1 rows) để mọi ngày val/test có H-step prediction hợp lệ
    _seq_len_e = config.get("model", {}).get("seq_len", 30)
    _horizon_e = config.get("model", {}).get("forecast_horizon", 1)
    _ctx_len_e = _seq_len_e + _horizon_e - 1

    def _predict_ctx(context_df, target_df):
        if len(context_df) == 0:
            return loaded_model.predict(target_df)
        combined = pd.concat([context_df, target_df]).reset_index(drop=True)
        return loaded_model.predict(combined)[len(context_df) :]

    train_ctx = (
        train_df.groupby("series_id", group_keys=False).tail(_ctx_len_e)
        if "series_id" in train_df.columns
        else train_df.tail(_ctx_len_e)
    )
    val_ctx = (
        val_df.groupby("series_id", group_keys=False).tail(_ctx_len_e)
        if "series_id" in val_df.columns
        else val_df.tail(_ctx_len_e)
    )

    eval_sets = [
        ("validation", val_df, _predict_ctx(train_ctx, val_df)),
        ("test", test_df, _predict_ctx(val_ctx, test_df)),
    ]

    # đánh giá trên cả val và test -> val để tune, test để báo cáo kết quả cuối cùng
    for split_name, split_df, preds in eval_sets:
        if len(split_df) == 0:
            continue
        y_true = _get_true_sales(split_df)
        metrics = evaluate_all(y_true, preds, weights_from_frame(split_df))
        typer.echo(f"{split_name.capitalize()} metrics: {metrics}")

        # Save plots
        plot_predictions(
            y_true,
            preds,
            title=f"{model} - {split_name}",
            save_path=f"{run_dir}/{split_name}_predictions.png",
        )
        plot_residuals(
            y_true,
            preds,
            title=f"{model} - {split_name}",
            save_path=f"{run_dir}/{split_name}_residuals.png",
        )

    typer.echo(f"Plots saved to {run_dir}")


if __name__ == "__main__":
    app()
