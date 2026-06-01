#!/usr/bin/env python3
"""Debug script: check if Phase 1 & 2 features are being created."""

import pandas as pd
from src.data.features import add_all_features

# Load cleaned data
df = pd.read_feather("data/cleaned/train_cleaned.feather")
print(f"Original columns: {list(df.columns)}")
print(f"Original shape: {df.shape}")
print(f"Has 'family' column: {'family' in df.columns}")
print(f"Has 'onpromotion' column: {'onpromotion' in df.columns}")

# Filter to DAIRY like the training does
if "family" in df.columns:
    df = df[df["family"] == "DAIRY"].copy()
    print(f"After DAIRY filter: {df.shape}")

# Create series_id (required by add_all_features)
df["series_id"] = df["store_nbr"].astype(str) + "_" + df["item_nbr"].astype(str)
df = df.sort_values(["series_id", "date"]).reset_index(drop=True)

# Add features
config_features = {
    "use_time": True,
    "use_fourier": True,
    "use_lag": True,
    "lag_windows": [1, 7, 14, 28, 30, 60, 90, 365],
    "use_rolling": True,
    "rolling_windows": [7, 14, 30],
    "rolling_stats": ["mean", "std", "median"],
    "use_group_mean": True,
    "use_payday": True,
    "use_zero_sales": True,
    "use_promo": True,
    "use_item_level_agg": True,
    "use_family_level_agg": True,
}

df_before = df.copy()
print(f"\nBefore add_all_features: {df_before.shape[1]} columns")
df_after = add_all_features(
    df_before, feature_cfg=config_features, train_end="2017-05-15"
)
print(f"After add_all_features: {df_after.shape[1]} columns")

print(
    f"\nNew item-level columns: {[c for c in df_after.columns if c.startswith('item_')]}"
)
print(
    f"New family-level columns: {[c for c in df_after.columns if c.startswith('family_')]}"
)
print(
    f"New promo future columns: {[c for c in df_after.columns if 'future' in c or ('promo' in c and 'count' in c)]}"
)
print(
    f"New conditional promo columns: {[c for c in df_after.columns if 'with_promo' in c or 'no_promo' in c]}"
)

print(f"\nAll new columns after add_all_features:")
new_cols = sorted([c for c in df_after.columns if c not in df_before.columns])
print(f"Count: {len(new_cols)}")
for c in new_cols:
    print(f"  {c}")
