# Shared Forecast Exogenous Information Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the educational benchmark use one explainable forecast-time external-information contract across exog-capable models while keeping ARIMA as the no-exog baseline.

**Architecture:** Add a focused `src/data/forecast_exog.py` contract module. SARIMAX consumes the explicit temporal contract, Prophet consumes the same non-Fourier signals as regressors while retaining native seasonality, and XGBoost/LSTM consume the shared contract plus sales-history features. Keep oil in the cleaned cache for optional experiments but disable it in the default model-facing feature configuration.

**Tech Stack:** Python 3.11, pandas, NumPy, statsmodels, Prophet, XGBoost, PyTorch, pytest, uv

---

### Task 1: Add The Shared Forecast Exog Contract

**Files:**
- Create: `src/data/forecast_exog.py`
- Create: `tests/test_forecast_exog.py`

- [ ] **Step 1: Write the failing contract tests**

```python
"""Tests for the educational shared forecast-exog contract."""

import pandas as pd
import pytest

from src.data.features import fourier_cols
from src.data.forecast_exog import (
    global_model_feature_cols,
    prophet_regressor_cols,
    require_columns,
    temporal_forecast_exog_cols,
)


def _feature_cfg():
    return {
        "use_fourier": True,
        "use_holiday": True,
        "use_payday": True,
        "use_promo": True,
        "use_perishable": True,
        "use_oil": False,
    }


def test_temporal_contract_is_stable_and_explainable():
    assert temporal_forecast_exog_cols(_feature_cfg()) == [
        *fourier_cols(),
        "is_holiday", "is_event", "is_payday", "onpromotion",
    ]


def test_prophet_uses_native_seasonality_instead_of_explicit_fourier():
    assert prophet_regressor_cols(_feature_cfg()) == [
        "is_holiday", "is_event", "is_payday", "onpromotion",
    ]


def test_global_models_use_shared_exog_plus_sales_history_only():
    df = pd.DataFrame(columns=[
        "store_nbr", "item_nbr", "fourier_w_sin_1", "is_holiday", "is_event",
        "is_payday", "onpromotion", "perishable", "unit_sales_lag_1",
        "unit_sales_rolling_mean_7", "series_dow_avg", "zero_sales_last_28",
        "dcoilwtico", "oil_lag_7", "promo_count_7", "month", "is_imputed",
    ])
    cols = global_model_feature_cols(df, _feature_cfg(), include_entity_ids=True)
    assert cols == [
        "store_nbr", "item_nbr", "fourier_w_sin_1", "is_holiday", "is_event",
        "is_payday", "onpromotion", "perishable", "unit_sales_lag_1",
        "unit_sales_rolling_mean_7", "series_dow_avg", "zero_sales_last_28",
    ]


def test_global_models_can_opt_into_oil_for_follow_up_experiments():
    df = pd.DataFrame(columns=["dcoilwtico", "oil_lag_7"])
    cfg = {**_feature_cfg(), "use_oil": True}
    assert global_model_feature_cols(df, cfg) == ["dcoilwtico", "oil_lag_7"]


def test_required_columns_fail_clearly():
    with pytest.raises(ValueError, match="SARIMAX missing required forecast exog columns: is_payday"):
        require_columns(pd.DataFrame({"onpromotion": [0]}), ["onpromotion", "is_payday"], "SARIMAX")
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_forecast_exog.py -q
```

Expected: FAIL during import because `src.data.forecast_exog` does not exist.

- [ ] **Step 3: Implement the focused contract helper**

```python
"""Shared forecast-time external-information contract for the educational benchmark."""

from collections.abc import Iterable

import pandas as pd

from src.data.features import fourier_cols

_HOLIDAY_COLS = ["is_holiday", "is_event"]
_PAYDAY_COLS = ["is_payday"]
_PROMO_COLS = ["onpromotion"]
_STATIC_GLOBAL_COLS = ["perishable"]
_OPTIONAL_OIL_COLS = ["dcoilwtico", "oil_lag_7"]
_HISTORY_EXACT_COLS = [
    "series_dow_avg", "store_avg", "item_avg", "family_avg", "zero_sales_last_28",
]
_HISTORY_PREFIXES = ("unit_sales_lag_", "unit_sales_rolling_")


def _enabled(feature_cfg: dict | None, key: str) -> bool:
    return (feature_cfg or {}).get(key, True)


def _unique_available(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(col for col in cols if col in df.columns))


def temporal_forecast_exog_cols(feature_cfg: dict | None = None) -> list[str]:
    cols: list[str] = []
    if _enabled(feature_cfg, "use_fourier"):
        cols += fourier_cols()
    if _enabled(feature_cfg, "use_holiday"):
        cols += _HOLIDAY_COLS
    if _enabled(feature_cfg, "use_payday"):
        cols += _PAYDAY_COLS
    if _enabled(feature_cfg, "use_promo"):
        cols += _PROMO_COLS
    return cols


def prophet_regressor_cols(feature_cfg: dict | None = None) -> list[str]:
    return [c for c in temporal_forecast_exog_cols(feature_cfg) if not c.startswith("fourier_")]


def global_model_feature_cols(
    df: pd.DataFrame,
    feature_cfg: dict | None = None,
    *,
    include_entity_ids: bool = False,
) -> list[str]:
    cols = ["store_nbr", "item_nbr"] if include_entity_ids else []
    cols += temporal_forecast_exog_cols(feature_cfg)
    if _enabled(feature_cfg, "use_perishable"):
        cols += _STATIC_GLOBAL_COLS
    if _enabled(feature_cfg, "use_oil"):
        cols += _OPTIONAL_OIL_COLS
    cols += [c for c in df.columns if c.startswith(_HISTORY_PREFIXES)]
    cols += _HISTORY_EXACT_COLS
    return _unique_available(df, cols)


def require_columns(df: pd.DataFrame, cols: Iterable[str], consumer: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{consumer} missing required forecast exog columns: {', '.join(missing)}"
        )
```

- [ ] **Step 4: Run contract tests to verify GREEN**

Run:

```bash
uv run pytest tests/test_forecast_exog.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/data/forecast_exog.py tests/test_forecast_exog.py
git commit -m "feat: add shared forecast exog contract"
```

### Task 2: Make SARIMAX Consume The Shared Temporal Contract

**Files:**
- Modify: `src/models/sarimax.py`
- Modify: `configs/sarimax.yaml`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write the failing SARIMAX test**

```python
import numpy as np

from src.data.features import fourier_cols
from src.models.sarimax import SARIMAXModel


def test_sarimax_extracts_shared_temporal_exog_in_stable_order():
    cols = [*fourier_cols(), "is_holiday", "is_event", "is_payday", "onpromotion"]
    df = pd.DataFrame({col: [i, i + 1] for i, col in enumerate(cols)})
    model = SARIMAXModel({"features": {
        "use_fourier": True, "use_holiday": True, "use_payday": True, "use_promo": True,
    }})
    assert np.array_equal(model._get_exog(df), df[cols].values.astype(float))
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
uv run pytest tests/test_models.py::test_sarimax_extracts_shared_temporal_exog_in_stable_order -q
```

Expected: FAIL because SARIMAX currently extracts Fourier columns only.

- [ ] **Step 3: Update SARIMAX**

Replace the Fourier-only import and constructor assignment:

```python
from src.data.forecast_exog import require_columns, temporal_forecast_exog_cols
```

```python
self._exog_cols = temporal_forecast_exog_cols(config.get("features", {}))
```

Replace `_get_exog()` with:

```python
def _get_exog(self, df: pd.DataFrame) -> np.ndarray | None:
    """Extract shared temporal exog in the same stable order for fit and forecast."""
    if not self._exog_cols:
        return None
    require_columns(df, self._exog_cols, "SARIMAX")
    return df[self._exog_cols].values.astype(float)
```

Update `configs/sarimax.yaml` comments to state that SARIMAX receives Fourier,
holiday/event, payday, and promotion schedule columns. Remove the stale comment
claiming that `exog_columns` does not need configuration because only Fourier is
used.

- [ ] **Step 4: Run SARIMAX and contract tests**

Run:

```bash
uv run pytest tests/test_models.py::test_sarimax_extracts_shared_temporal_exog_in_stable_order tests/test_forecast_exog.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/models/sarimax.py configs/sarimax.yaml tests/test_models.py
git commit -m "feat: align sarimax with shared forecast exog"
```

### Task 3: Make Prophet Consume Shared Non-Fourier Regressors

**Files:**
- Modify: `src/models/prophet_model.py`
- Modify: `configs/prophet.yaml`
- Modify: `tests/test_models.py`
- Modify: `docs/superpowers/specs/2026-05-31-shared-forecast-exog-design.md`

- [ ] **Step 1: Write the failing Prophet frame test**

```python
from src.models.prophet_model import ProphetModel


def test_prophet_frame_includes_shared_non_fourier_regressors():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-01"]),
        "unit_sales": [2.0],
        "is_holiday": [1],
        "is_event": [0],
        "is_payday": [0],
        "onpromotion": [1],
    })
    model = ProphetModel({"features": {
        "use_fourier": True, "use_holiday": True, "use_payday": True, "use_promo": True,
    }})
    train = model._prepare_prophet_df(df)
    future = model._prepare_prophet_df(df, include_target=False)
    assert train.columns.tolist() == [
        "ds", "y", "is_holiday", "is_event", "is_payday", "onpromotion",
    ]
    assert future.columns.tolist() == [
        "ds", "is_holiday", "is_event", "is_payday", "onpromotion",
    ]
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
uv run pytest tests/test_models.py::test_prophet_frame_includes_shared_non_fourier_regressors -q
```

Expected: FAIL because Prophet currently prepares only `ds` and `y`.

- [ ] **Step 3: Update Prophet**

Import the helper:

```python
from src.data.forecast_exog import prophet_regressor_cols, require_columns
```

Initialize the stable regressor contract:

```python
self.regressor_cols = prophet_regressor_cols(config.get("features", {}))
```

Replace `_prepare_prophet_df()` with:

```python
def _prepare_prophet_df(
    self, df: pd.DataFrame, *, include_target: bool = True
) -> pd.DataFrame:
    """Build Prophet fit/predict frame with shared forecast-time regressors."""
    require_columns(df, self.regressor_cols, "Prophet")
    cols = ["date"]
    if include_target:
        cols.append("unit_sales")
    cols += self.regressor_cols
    return df[cols].copy().rename(columns={"date": "ds", "unit_sales": "y"})
```

Before `m.fit(pdf)`, register the regressors:

```python
for regressor in self.regressor_cols:
    m.add_regressor(regressor)
```

During prediction, build a future frame without the target:

```python
future = self._prepare_prophet_df(group, include_target=False)
```

Remove `_prepare_holidays()` and stop passing a train-only holidays table into
`Prophet(...)`. Update `configs/prophet.yaml` and the design spec to explain that
holiday/event flags are forecast-time regressors. This avoids silently omitting
future holidays from predictions.

- [ ] **Step 4: Run Prophet and contract tests**

Run:

```bash
uv run pytest tests/test_models.py::test_prophet_frame_includes_shared_non_fourier_regressors tests/test_forecast_exog.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/models/prophet_model.py configs/prophet.yaml tests/test_models.py docs/superpowers/specs/2026-05-31-shared-forecast-exog-design.md
git commit -m "feat: align prophet with shared forecast exog"
```

### Task 4: Align Global Model Feature Selection And Disable Oil By Default

**Files:**
- Modify: `src/models/xgboost_model.py`
- Modify: `src/models/torch_utils.py`
- Modify: `configs/features.yaml`
- Modify: `tests/test_models.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing global-model tests**

Add imports:

```python
from src.models.torch_utils import _select_feature_cols
from src.utils.config import load_config
```

Add tests:

```python
def _global_feature_frame():
    return pd.DataFrame({
        "store_nbr": [1], "item_nbr": [10], "unit_sales": [2.0],
        "fourier_w_sin_1": [0.1], "is_holiday": [1], "is_event": [0],
        "is_payday": [0], "onpromotion": [1], "perishable": [1],
        "unit_sales_lag_1": [1.0], "series_dow_avg": [2.0],
        "dcoilwtico": [50.0], "oil_lag_7": [49.0], "promo_count_7": [2],
        "month": [1], "is_imputed": [0],
    })


def test_global_models_share_explainable_external_information():
    cfg = {"features": {
        "use_fourier": True, "use_holiday": True, "use_payday": True,
        "use_promo": True, "use_perishable": True, "use_oil": False,
    }}
    df = _global_feature_frame()
    xgb_cols = XGBoostModel(cfg)._get_features(df).columns.tolist()
    lstm_cols = _select_feature_cols(df, config=cfg)
    assert xgb_cols == [
        "store_nbr", "item_nbr", "fourier_w_sin_1", "is_holiday", "is_event",
        "is_payday", "onpromotion", "perishable", "unit_sales_lag_1", "series_dow_avg",
    ]
    assert lstm_cols == xgb_cols[2:]
```

In `tests/test_config.py`, add:

```python
def test_default_educational_benchmark_disables_oil_features():
    config = load_config()
    assert config["exog"]["use_oil"] is True
    assert config["features"]["use_oil"] is False
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_models.py::test_global_models_share_explainable_external_information tests/test_config.py::test_default_educational_benchmark_disables_oil_features -q
```

Expected: FAIL because dynamic numeric feature selection still includes unrelated
numeric columns and `features.use_oil` defaults to `true`.

- [ ] **Step 3: Reuse the contract in XGBoost and LSTM**

In `src/models/xgboost_model.py`, import:

```python
from src.data.forecast_exog import global_model_feature_cols
```

Replace `_get_features()` implementation with:

```python
def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Use entity IDs, shared forecast exog, and leakage-safe sales-history features."""
    self.feature_cols = global_model_feature_cols(
        df, self.config.get("features", {}), include_entity_ids=True
    )
    return df[self.feature_cols].fillna(0)
```

Remove `_EXCLUDE` and `_REQUIRED` from `src/models/xgboost_model.py`. Keep the
NumPy import because `predict()` still uses `np.expm1()` and `np.clip()`.

In `src/models/torch_utils.py`, import:

```python
from src.data.forecast_exog import global_model_feature_cols
```

Replace `_select_feature_cols()` with:

```python
def _select_feature_cols(df, config: dict | None = None) -> list[str]:
    """Use shared forecast exog and leakage-safe sales-history features."""
    return global_model_feature_cols(df, (config or {}).get("features", {}))
```

Remove `_EXCLUDE_FROM_FEATURES`.

In `configs/features.yaml`, set:

```yaml
use_oil: false
```

Keep `configs/base.yaml -> exog.use_oil: true` so oil remains cached for optional
experiments without forcing a cache rebuild. The model-facing feature flag controls
the primary benchmark.

- [ ] **Step 4: Run global-model and config tests**

Run:

```bash
uv run pytest tests/test_models.py tests/test_config.py tests/test_forecast_exog.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/xgboost_model.py src/models/torch_utils.py configs/features.yaml tests/test_models.py tests/test_config.py
git commit -m "feat: align global models with shared forecast exog"
```

### Task 5: Update The Educational README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README model and feature documentation**

Update the model table and explanatory text so it states:

```markdown
- **Shared external-information benchmark:** models that support external
  information receive the same explainable forecast-time sources: calendar
  seasonality, holiday/event flags, payday, planned promotion, and product
  perishability where useful.
- **ARIMA** intentionally remains a no-exog univariate baseline.
- **SARIMAX** receives explicit Fourier terms plus holiday/event, payday, and
  planned-promotion columns.
- **Prophet** represents weekly/yearly seasonality internally and receives
  holiday/event, payday, and planned-promotion regressors.
- **XGBoost/LSTM** receive the shared sources plus leakage-safe historical-sales
  features. `perishable` is useful here because these are global models.
- **Oil and transactions are excluded from the primary educational benchmark:**
  their future values are not naturally known at forecast origin. Oil remains in
  the cleaned cache for optional follow-up experiments.
```

Remove stale statements that SARIMAX uses Fourier only or that oil is enabled in
the primary benchmark. Explain that `perishable` is omitted from per-series
SARIMAX/Prophet because it is constant within one series.

- [ ] **Step 2: Check README diff**

Run:

```bash
git diff --check -- README.md
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: explain shared forecast exog benchmark"
```

### Task 6: Verify The Integrated Change

**Files:**
- Verify only

- [ ] **Step 1: Compile changed Python modules**

Run:

```bash
uv run python -m compileall src tests
```

Expected: exit code `0`.

- [ ] **Step 2: Run the complete automated suite**

Run:

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run lightweight configuration smoke checks**

Run:

```bash
uv run python - <<'PY'
from src.data.forecast_exog import prophet_regressor_cols, temporal_forecast_exog_cols
from src.utils.config import load_config

sarimax = load_config("sarimax")
prophet = load_config("prophet")
print("sarimax:", temporal_forecast_exog_cols(sarimax["features"]))
print("prophet:", prophet_regressor_cols(prophet["features"]))
print("oil model-facing default:", sarimax["features"]["use_oil"])
PY
```

Expected:

```text
sarimax: [10 Fourier columns, 'is_holiday', 'is_event', 'is_payday', 'onpromotion']
prophet: ['is_holiday', 'is_event', 'is_payday', 'onpromotion']
oil model-facing default: False
```

- [ ] **Step 4: Decide whether to run full model training**

Do not rebuild the cleaned cache: oil remains cached and only the model-facing flag
changes. Run full ARIMA/SARIMAX training only if the user wants refreshed benchmark
artifacts because each command fits 40 per-series statistical models and may take
material time:

```bash
uv run python scripts/train.py train --model arima
uv run python scripts/train.py train --model sarimax
```

- [ ] **Step 5: Inspect final repository state**

Run:

```bash
git status --short --branch
git log --oneline -n 8
```

Expected: only intentional changes or a clean worktree after commits.
