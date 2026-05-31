# DAIRY Family Scope Design

## Goal

Simplify the Favorita forecasting benchmark so it remains practical to train and
easy to explain in a report. The benchmark will forecast daily SKU sales only for
the `DAIRY` product family at stores in cluster `1`.

This is intentionally a scoped educational benchmark. Its results describe the
selected DAIRY subset and must not be presented as representative of the full
Favorita assortment.

## Dataset Scope

Apply these filters consistently to the cleaned dataset:

1. Keep the existing store filter: `cluster = [1]`.
2. Add an item metadata filter: `family = ["DAIRY"]`.
3. Keep every item and every `(store_nbr, item_nbr)` series that matches both
   filters.
4. Preserve the existing date range, temporal split, and zero-fill behavior.

The filter must run after joining `items.csv`, because `family` is item metadata,
and before zero-fill, because early filtering avoids expanding rows that will be
discarded.

Based on the current cluster-1 cache, DAIRY contains about `774,048` dense
training rows through `2017-05-15`, within the target range of `500k-800k`.
The exact count must be verified after rebuilding the cleaned cache.

## Returns

Do not remove items that contain negative `unit_sales`. Keep the current behavior:

- retain EDA-only return indicators;
- clip negative target values to zero before training and evaluation.

Returns are rare in the measured cluster-1 cache. Dropping complete items would
discard valid sales history and distort the selected family.

## Configuration

Add an optional `item_filter` configuration block:

```yaml
item_filter:
  by: family
  value: [DAIRY]
```

The data cleaner must support filtering by joined item metadata. The cleaned-cache
signature must include `item_filter`, so a cache created for another product
scope cannot be reused silently.

## Documentation

Update the project documentation to state:

- the benchmark scope is DAIRY products in store cluster `1`;
- model outputs and comparisons apply only to this subset;
- the family filter is applied before zero-fill;
- negative sales remain clipped to zero rather than removing affected products.

## Verification

Add data-layer tests for:

1. filtering rows by `family`;
2. building a dataset with both the cluster and DAIRY filters;
3. rejecting a cleaned cache when `item_filter` differs from the manifest.

Run the full test suite, rebuild `data/cleaned/train_cleaned.feather`, then verify:

- the manifest contains `item_filter.by = family` and `item_filter.value = [DAIRY]`;
- all cleaned rows have `family = DAIRY`;
- dense train rows through `2017-05-15` are within `500k-800k`.
