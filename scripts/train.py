"""CLI script for training models.

Hai chế độ chạy:
- CLI mode: `python scripts/train.py --model arima` → dùng cho CI/automation
- Interactive mode: `python scripts/train.py` (không args) → hiện menu chọn model + tham số
"""

import ast
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from src.data.features import add_all_features, apply_log_transform
from src.data.loader import load_cleaned_data, load_raw_data, select_series
from src.data.preprocessor import preprocess
from src.models import MODEL_REGISTRY, get_model_class
from src.utils.config import load_config, make_param_slug
from src.utils.seed import set_seed
from src.utils.visualization import (
    plot_loss_curve,
    plot_multi_horizon,
    plot_predictions,
    plot_predictions_zoomed,
    plot_residuals,
)
from src.evaluation.metrics import evaluate_all

app = typer.Typer(help="Train time series forecasting models.")


def _parse_overrides(set_values: list[str] | None) -> dict:
    """Parse key=value overrides từ CLI. Hỗ trợ list, int, float, null."""
    if not set_values:
        return {}
    overrides = {}
    for item in set_values:
        key, _, val = item.partition("=")
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            if val.lower() == "null":
                val = None
        overrides[key] = val
    return overrides


def _evaluate_split(model, split_df, split_name: str, model_name: str, out_dir: str):
    """Đánh giá model trên 1 split (val/test), in metrics + vẽ biểu đồ."""
    if len(split_df) == 0:
        return None
    try:
        metrics = model.evaluate(split_df)
    except Exception as e:
        typer.echo(f"  Warning: evaluate {split_name} failed: {e}", err=True)
        return None
    typer.echo(f"  {split_name} metrics: {metrics}")
    try:
        preds = model.predict(split_df)
        y_true = split_df["unit_sales"].values
        plot_predictions(y_true, preds, title=f"{model_name} - {split_name}", save_path=str(Path(out_dir) / f"{split_name}_predictions.png"))
        plot_residuals(y_true, preds, title=f"{model_name} - {split_name}", save_path=str(Path(out_dir) / f"{split_name}_residuals.png"))
    except Exception as e:
        typer.echo(f"  Warning: không vẽ được chart {split_name}: {e}", err=True)
    return metrics


def _train_single(model_name: str, overrides: dict, experiment_name: str = ""):
    """Train 1 model + evaluate trên val & test set."""
    config = load_config(model_name, overrides)
    set_seed(config.get("seed", 42))

    typer.echo(f"\n{'─'*50}")
    typer.echo(f"Config for [{model_name}]:")
    typer.echo(json.dumps(config, indent=2, default=str))
    typer.echo(f"{'─'*50}\n")

    param_slug = make_param_slug(config)

    _horizon  = config.get("model", {}).get("forecast_horizon", 1)
    _log_tag  = "log" if str(config.get("use_log_sales", "false")).lower() == "true" else "nolog"
    _strategy = config.get("forecast_strategy", "direct")
    param_slug = f"{param_slug}__h{_horizon}__{_log_tag}__{_strategy}"

    typer.echo("Loading data...")
    use_cleaned = config.get("data", {}).get("use_cleaned", False)
    if use_cleaned:
        typer.echo("Using pre-cleaned data...")
        df, _ = load_cleaned_data(config)
    else:
        df, _ = load_raw_data(config)

    # chọn mẫu đại diện chuỗi cho per-series models (ARIMA/SARIMAX/Prophet)
    # XGBoost/LSTM dùng toàn bộ nếu series_sample.n_series=null
    df = select_series(df, config)

    typer.echo("Adding features...")
    feature_cfg = config.get("features", {})
    df = add_all_features(df, feature_cfg=feature_cfg)

    if config.get("use_log_sales", False):
        typer.echo("Applying log1p transform to sales and derived features...")
        df = apply_log_transform(df)

    typer.echo("Preprocessing...")
    train_df, val_df, test_df, scaler = preprocess(df, config)

    typer.echo(f"Training {model_name}... ({len(train_df)} train rows, {len(val_df)} val rows)")
    model_class = get_model_class(model_name)
    model = model_class(config)

    start = time.time()
    train_info = model.train(train_df, val_df)
    elapsed = time.time() - start
    typer.echo(f"Training done in {elapsed:.1f}s. Info: {train_info}")

    use_log = config.get("use_log_sales", False)

    def _get_true_sales(split_df):
        sales = split_df["unit_sales"].values.astype(float)
        return np.expm1(sales) if use_log else sales

    # context window: seq_len + H - 1 rows để ngày đầu val/test có đủ lịch sử
    _seq_len  = config.get("model", {}).get("seq_len", 30)
    _horizon  = config.get("model", {}).get("forecast_horizon", 1)
    _ctx_len  = _seq_len + _horizon - 1

    def _predict_with_context(context_df, target_df):
        """Nối context_df vào đầu target_df, predict, trả về predictions cho target_df."""
        if len(target_df) == 0:
            return np.array([])
        if len(context_df) == 0:
            return model.predict(target_df)
        combined = pd.concat([context_df, target_df]).reset_index(drop=True)
        preds_full = model.predict(combined)
        return preds_full[len(context_df):]

    # context theo series_id → không trộn dữ liệu giữa các chuỗi khác nhau
    train_ctx_val = train_df.groupby("series_id", group_keys=False).tail(_ctx_len) \
        if "series_id" in train_df.columns else train_df.tail(_ctx_len)
    val_predictions = _predict_with_context(train_ctx_val, val_df)
    y_true_val = _get_true_sales(val_df)
    metrics = evaluate_all(y_true_val, val_predictions)
    typer.echo(f"Validation metrics: {metrics}")

    def _build_test_context():
        """Trả về (combined_df, n_ctx): combined đã có context đủ ctx_len rows."""
        if "series_id" in test_df.columns:
            min_rows = test_df.groupby("series_id").size().min()
        else:
            min_rows = len(test_df)
        if min_rows < _ctx_len + 1:
            ctx_source = val_df if len(val_df) > 0 else train_df
            ctx_tail = ctx_source.groupby("series_id", group_keys=False).tail(_ctx_len) \
                if "series_id" in ctx_source.columns else ctx_source.tail(_ctx_len)
            combined = pd.concat([ctx_tail, test_df]).reset_index(drop=True)
            return combined, len(ctx_tail)
        return test_df, 0

    _test_combined, _n_ctx = _build_test_context()

    test_predictions_full = model.predict(_test_combined)
    test_predictions = test_predictions_full[_n_ctx:]
    y_true_test = _get_true_sales(test_df)
    test_metrics = evaluate_all(y_true_test, test_predictions)
    typer.echo(f"Test metrics:       {test_metrics}")

    results = model.get_result_template(metrics, param_slug=param_slug, experiment_name=experiment_name)
    results["test_metrics"] = test_metrics
    results["metadata"] = {
        "n_series": df["series_id"].nunique() if "series_id" in df.columns else 0,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "feature_count": len(getattr(model, "feature_cols", [])),
    }

    train_l = train_info.get("train_losses", [])
    val_l   = train_info.get("val_losses", [])
    epochs_config = config.get("model", {}).get("epochs")
    epochs_done   = train_info.get("epochs_trained")
    results["training_info"] = {
        "device": str(getattr(model, "device", "unknown")),
        "forecast_strategy": config.get("forecast_strategy", "direct"),
        "loss_fn": config.get("loss_fn", "mse"),
        "forecast_horizon_days": config.get("model", {}).get("forecast_horizon", 1),
        "seq_len": config.get("model", {}).get("seq_len"),
        "use_log_sales": use_log,
        "patience": config.get("model", {}).get("patience", 10),
        "min_delta": config.get("model", {}).get("min_delta", 0.0),
        "weight_decay": config.get("model", {}).get("weight_decay", 0.0),
        "epochs_config": epochs_config,
        "epochs_trained": epochs_done,
        "early_stopped": (epochs_done < epochs_config)
            if (epochs_done is not None and epochs_config is not None) else None,
    }
    results["loss_summary"] = {
        "best_train_loss":  round(min(train_l), 6)          if train_l else None,
        "final_train_loss": round(train_l[-1], 6)           if train_l else None,
        "best_val_loss":    round(min(val_l), 6)            if val_l   else None,
        "final_val_loss":   round(val_l[-1], 6)             if val_l   else None,
        "best_val_epoch":   int(np.argmin(val_l)) + 1       if val_l   else None,
    }
    out_dir = model.save_results(results, config.get("results_dir", "results"))

    # Vẽ chart từ predictions ĐÃ tính với context (KHÔNG re-predict contextless để
    # tránh metric rác với sequence models: test slice mỗi chuỗi < seq_len+H → window
    # không tạo được. test_metrics giữ giá trị đúng tính ở trên với context).
    typer.echo("Evaluating...")
    try:
        if len(val_df) > 0:
            plot_predictions(y_true_val, val_predictions, title=f"{model_name} - validation",
                             save_path=str(Path(out_dir) / "validation_predictions.png"))
            plot_residuals(y_true_val, val_predictions, title=f"{model_name} - validation",
                           save_path=str(Path(out_dir) / "validation_residuals.png"))
        plot_predictions(y_true_test, test_predictions, title=f"{model_name} - test",
                         save_path=str(Path(out_dir) / "test_predictions.png"))
        plot_residuals(y_true_test, test_predictions, title=f"{model_name} - test",
                       save_path=str(Path(out_dir) / "test_residuals.png"))
    except Exception as e:
        typer.echo(f"  Warning: không vẽ được chart: {e}", err=True)
    typer.echo(f"  test metrics: {test_metrics}")

    model_path = Path(out_dir) / "model.pkl"
    model.save(str(model_path))

    loss_history = {
        "train_losses": train_info.get("train_losses", []),
        "val_losses":   train_info.get("val_losses", []),
    }
    loss_path = Path(out_dir) / "loss_history.json"
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)

    try:
        predictions = val_predictions
        y_true = y_true_val
        plot_predictions(y_true, predictions, title=f"{model_name} - Val Predictions", save_path=str(Path(out_dir) / "val_predictions.png"))
        plot_predictions_zoomed(y_true, predictions, title=f"{model_name} - Val Predictions (Zoomed)", save_path=str(Path(out_dir) / "val_predictions_zoomed.png"))
        plot_residuals(y_true, predictions, title=f"{model_name} - Val Residuals", save_path=str(Path(out_dir) / "val_residuals.png"))
        plot_predictions(y_true_test, test_predictions, title=f"{model_name} - Test Predictions", save_path=str(Path(out_dir) / "test_predictions.png"))
        plot_predictions_zoomed(y_true_test, test_predictions, title=f"{model_name} - Test Predictions (Zoomed)", save_path=str(Path(out_dir) / "test_predictions_zoomed.png"))
        plot_residuals(y_true_test, test_predictions, title=f"{model_name} - Test Residuals", save_path=str(Path(out_dir) / "test_residuals.png"))

        # forecast fan: chỉ có với multioutput/recursive
        strategy = config.get("forecast_strategy", "direct")
        if strategy in ("multioutput", "recursive") and hasattr(model, "predict_all_horizons"):
            try:
                val_fan = model.predict_all_horizons(val_df)
                test_fan_full = model.predict_all_horizons(_test_combined)
                test_fan = test_fan_full[_n_ctx:]
                # series_id thay cho store_ids; không có open_flags trong dataset này
                _val_series_ids  = val_df["series_id"].values  if "series_id" in val_df.columns  else None
                _tst_series_ids  = test_df["series_id"].values if "series_id" in test_df.columns else None
                plot_multi_horizon(
                    y_true_val, val_fan,
                    title=f"{model_name} - Val Forecast Fan (H={config.get('model',{}).get('forecast_horizon',1)})",
                    store_ids=_val_series_ids,
                    open_flags=None,
                    save_path=str(Path(out_dir) / "val_forecast_fan.png"),
                )
                plot_multi_horizon(
                    y_true_test, test_fan,
                    title=f"{model_name} - Test Forecast Fan (H={config.get('model',{}).get('forecast_horizon',1)})",
                    store_ids=_tst_series_ids,
                    open_flags=None,
                    save_path=str(Path(out_dir) / "test_forecast_fan.png"),
                )
            except Exception as fan_e:
                typer.echo(f"Warning: could not generate forecast fan chart: {fan_e}", err=True)

        if loss_history["train_losses"]:
            plot_loss_curve(
                loss_history["train_losses"],
                loss_history["val_losses"] or None,
                title=f"{model_name} - Loss Curve",
                save_path=str(Path(out_dir) / "loss_curve.png"),
            )
    except Exception as e:
        typer.echo(f"Warning: could not generate charts: {e}", err=True)

    results["files_generated"] = sorted(f.name for f in Path(out_dir).iterdir() if f.is_file())
    with open(Path(out_dir) / "result.json", "w", encoding="utf-8") as _f:
        json.dump(results, _f, indent=2, default=str)

    typer.echo(f"Results saved to {out_dir}")
    return {"out_dir": out_dir, "metrics": metrics}


@app.command()
def train(
    model: str = typer.Option(None, "--model", "-m", help="Model name, 'all', or omit for interactive"),
    set_values: list[str] = typer.Option(None, "--set", "-s", help="Override config key=value"),
    experiment_name: str = typer.Option("", "--experiment-name", "-e", help="Optional experiment name"),
    diagnostics: bool = typer.Option(False, "--diagnostics", help="Chạy chẩn đoán phần dư sau training"),
):
    """Train a forecasting model. Không truyền --model → vào interactive mode."""
    overrides = _parse_overrides(set_values)

    if model is None:
        from scripts import interactive_train_helpers as ith
        model, interactive_overrides = ith.interactive_select()
        overrides = {**interactive_overrides, **overrides}

    if model == "all":
        for name in MODEL_REGISTRY:
            typer.echo(f"\n{'='*60}")
            typer.echo(f"Training: {name}")
            typer.echo(f"{'='*60}")
            try:
                result = _train_single(name, overrides, experiment_name)
                if diagnostics and result:
                    trained_model, train_df, _, _, run_out_dir = result
                    _run_diagnostics(trained_model, train_df, name, run_out_dir)
            except Exception as e:
                typer.echo(f"ERROR training {name}: {e}", err=True)
    else:
        result = _train_single(model, overrides, experiment_name)
        if diagnostics and result:
            trained_model, train_df, _, _, run_out_dir = result
            _run_diagnostics(trained_model, train_df, model, run_out_dir)


@app.command(name="grid-search")
def grid_search(
    model: str = typer.Option(..., "--model", "-m", help="Tên model, vd: lstm, xgboost"),
    grid: str = typer.Option(
        None,
        "--grid",
        "-g",
        help=(
            'JSON dict ánh xạ tên param -> danh sách giá trị. '
            'Vd: \'{"model.hidden_size":[32,64],"model.num_layers":[1,2]}\''
        ),
    ),
    grid_file: str = typer.Option(
        None,
        "--grid-file",
        "-f",
        help="Đường dẫn tới file JSON chứa grid config.",
    ),
    set_values: list[str] = typer.Option(None, "--set", "-s", help="Override config cố định cho mọi run"),
    experiment_name: str = typer.Option("", "--experiment-name", "-e", help="Tiền tố experiment name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Chỉ in danh sách combinations, không train"),
):
    """Chạy grid search qua các tổ hợp hyperparameter."""
    if grid_file:
        try:
            with open(grid_file, encoding="utf-8") as _gf:
                grid_dict: dict = json.load(_gf)
            grid_dict = {k: v for k, v in grid_dict.items() if not k.startswith("_")}
        except Exception as e:
            typer.echo(f"ERROR: cannot read --grid-file '{grid_file}'. {e}", err=True)
            raise typer.Exit(1)
    elif grid:
        try:
            grid_dict = json.loads(grid)
        except json.JSONDecodeError as e:
            typer.echo(f"ERROR: --grid must be valid JSON. {e}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo("ERROR: provide --grid or --grid-file.", err=True)
        raise typer.Exit(1)

    param_names  = list(grid_dict.keys())
    param_values = [v if isinstance(v, list) else [v] for v in grid_dict.values()]
    combinations = list(itertools.product(*param_values))
    total = len(combinations)

    typer.echo(f"\nGrid search: {model} | {total} combinations")
    typer.echo(f"Params: {param_names}")
    for i, combo in enumerate(combinations):
        typer.echo(f"  [{i+1:3d}/{total}] " + ", ".join(f"{k}={v}" for k, v in zip(param_names, combo)))

    if dry_run:
        typer.echo("\n--dry-run: không train.")
        return

    base_overrides = _parse_overrides(set_values)
    summary: list[dict] = []
    start_all = time.time()

    for i, combo in enumerate(combinations):
        overrides = {**base_overrides}
        for k, v in zip(param_names, combo):
            overrides[k] = v

        combo_label = "__".join(f"{k.split('.')[-1]}={v}" for k, v in zip(param_names, combo))
        exp_name = f"{experiment_name}__{combo_label}" if experiment_name else combo_label

        typer.echo(f"\n{'='*65}")
        typer.echo(f"[{i+1}/{total}] {combo_label}")
        typer.echo(f"{'='*65}")

        try:
            result_info = _train_single(model, overrides, exp_name)
            summary.append({"combo": combo_label, "status": "ok", "result": result_info})
        except Exception as e:
            typer.echo(f"ERROR: {e}", err=True)
            summary.append({"combo": combo_label, "status": "error", "error": str(e)})

    elapsed = time.time() - start_all
    typer.echo(f"\n{'='*65}")
    typer.echo(f"Grid search done: {total} runs in {elapsed/60:.1f} min")
    typer.echo(f"{'='*65}")
    ok  = [s for s in summary if s["status"] == "ok"]
    err = [s for s in summary if s["status"] == "error"]
    typer.echo(f"  OK: {len(ok)} | Errors: {len(err)}")
    if ok:
        typer.echo("\n  Results for successful runs (sorted by NWRMSLE):")
        ok_sorted = sorted(ok, key=lambda s: s["result"]["metrics"].get("nwrmsle", float("inf")))
        for rank, s in enumerate(ok_sorted, 1):
            m = s["result"]["metrics"]
            metric_val = m.get("nwrmsle", "N/A")
            metric_str = f"{metric_val:.4f}" if isinstance(metric_val, float) else str(metric_val)
            typer.echo(f"  #{rank:2d}  NWRMSLE={metric_str}  {s['combo']}")
    if err:
        for s in err:
            typer.echo(f"  [ERROR] {s['combo']}: {s['error']}", err=True)


if __name__ == "__main__":
    app()
