# Shared Forecast Exogenous Information Design

## Purpose

Align the educational benchmark so models that support external information receive
the same explainable information sources. Keep the benchmark simple enough to
describe clearly in a thesis presentation.

The comparison is source-based rather than column-identical. Each model may encode
the same source in the form appropriate for its architecture. For example, Prophet
uses built-in weekly and yearly seasonality while SARIMAX receives Fourier columns.

## Primary Benchmark Contract

The primary benchmark uses external information that is easy to explain and
available at forecast time:

| Source | Representation | Availability |
| --- | --- | --- |
| Calendar seasonality | Weekly and yearly Fourier terms, or model-native equivalent | Derived from date |
| Holiday and event calendar | `is_holiday`, `is_event` | Known in advance |
| Payday cycle | `is_payday` | Derived from date |
| Promotion schedule | `onpromotion` | Assumed planned in advance |
| Product perishability | `perishable` | Static item metadata |

`dcoilwtico` and `transactions` are excluded from the primary benchmark. Their
future values are not naturally known at forecast origin, so including them would
make the educational comparison harder to explain. They remain available for a
future optional experiment.

## Model Contract

| Model | External information behavior |
| --- | --- |
| ARIMA | No external information. Retained as the univariate baseline. |
| SARIMAX | Receives Fourier terms, holiday/event flags, `is_payday`, and `onpromotion`. |
| Prophet | Uses native weekly/yearly seasonality plus holiday/event, `is_payday`, and `onpromotion` regressors. |
| XGBoost | Receives the shared sources plus historical sales-derived features. |
| LSTM | Receives the shared sources plus historical sales-derived features. |

`perishable` is passed to global models XGBoost and LSTM. It is not passed to
per-series SARIMAX or Prophet because its value is constant within one
`(store_nbr, item_nbr)` series and cannot explain changes over time.

Historical sales-derived features such as lag, rolling statistics, group means,
and zero-sales counts are not external information. XGBoost and LSTM may use them
because they are part of those models' history representation. SARIMAX, Prophet,
and ARIMA model history through their native time-series structures.

## Implementation Shape

1. Add a small shared feature-contract helper that lists the explainable forecast
   exogenous columns.
2. Update SARIMAX to read its supported columns from that contract instead of
   selecting only Fourier terms.
3. Update Prophet to use holiday/event, `is_payday`, and `onpromotion` regressors
   while retaining native seasonality.
4. Disable oil in the default educational benchmark configuration and documentation.
5. Keep XGBoost and LSTM behavior compatible with the shared source contract and
   verify that excluded oil columns are not selected by default.
6. Remove or revise stale documentation that claims SARIMAX uses Fourier only.

## Data Flow

```text
date, holidays_events.csv, train.onpromotion, items.perishable
                         |
                         v
             shared forecast information
                         |
         +---------------+----------------+
         |               |                |
         v               v                v
      SARIMAX          Prophet       XGBoost / LSTM
  explicit exog    native + regressors   exog + history

      ARIMA remains the no-exog baseline
```

## Error Handling

- SARIMAX and Prophet must fail with a clear message if a configured required
  forecast exogenous column is missing.
- Prediction must use the same exogenous column order as training.
- Optional static metadata must not be forced into per-series models when it is
  constant within a series.

## Testing

- Unit test the shared exogenous column contract.
- Unit test that SARIMAX extracts the shared supported columns in stable order.
- Unit test that Prophet prepares the configured regressors for both fit and
  prediction frames.
- Unit test that default XGBoost and LSTM feature selection excludes oil.
- Run the full pytest suite and a lightweight smoke check for ARIMA and SARIMAX
  configuration loading.

## Documentation

Update `README.md` to explain:

- why the benchmark uses a shared external-information set;
- why Fourier remains useful alongside real external information;
- why oil and transactions are excluded from the primary benchmark;
- why ARIMA intentionally remains a no-exog baseline;
- why `perishable` is useful for global models but not per-series models.
