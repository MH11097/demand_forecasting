"""Phân tích variance giữa các store trước Phase B (per-store normalization).

Tạo ra 5 biểu đồ lưu vào results/store_variance/:
  1. distribution_store_mean.png  — phân phối mean Sales per store
  2. distribution_store_cv.png    — Coefficient of Variation (std/mean) per store
  3. mean_vs_std_scatter.png      — scatter mean vs std, identify high-variance stores
  4. boxplot_sample_stores.png    — box plot Sales cho 30 store mẫu
  5. cdf_store_mean.png           — CDF của store mean (scale range)

Chạy:
    python scripts/analyze_store_variance.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.data.features import add_all_features
from src.data.loader import load_raw_data
from src.utils.config import load_config


# ── config & load ──────────────────────────────────────────────────────────────
OUT_DIR = Path("results/store_variance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
config = load_config("lstm", {})
df_raw, _ = load_raw_data(config)

# chỉ dùng training period để tránh bias từ validation/test
train_end = config["split"]["train_end"]
train_df = df_raw[df_raw["date"] <= train_end].copy()
print(f"  Training rows: {len(train_df):,}  |  Series: {train_df['series_id'].nunique()}")

# thống kê per-series (mỗi series_id = 1 chuỗi store×item)
store_stats = (
    train_df.groupby("series_id")["unit_sales"]
    .agg(
        mean="mean",
        std="std",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count="count",
    )
    .reset_index()
)
store_stats["cv"] = store_stats["std"] / store_stats["mean"]          # Coefficient of Variation
store_stats["iqr"] = store_stats["q75"] - store_stats["q25"]          # InterQuartile Range
store_stats["log_mean"] = np.log1p(store_stats["mean"])

print(f"\nStore stats summary:")
print(store_stats[["mean", "std", "cv", "median"]].describe().round(0).to_string())
print(f"\n  Mean sales range : {store_stats['mean'].min():.0f}  –  {store_stats['mean'].max():.0f}"
      f"  (ratio {store_stats['mean'].max()/store_stats['mean'].min():.1f}×)")
print(f"  CV range         : {store_stats['cv'].min():.3f}  –  {store_stats['cv'].max():.3f}")


# ── helpers ────────────────────────────────────────────────────────────────────
STYLE = dict(edgecolor="white", linewidth=0.4)

def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 1: Distribution of store mean Sales ───────────────────────────────────
print("\nPlotting...")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Distribution of Store Mean Daily Sales (Training set)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.hist(store_stats["mean"], bins=50, color="#4C72B0", **STYLE)
ax.axvline(store_stats["mean"].mean(), color="red", linewidth=1.5, linestyle="--", label=f"Global mean = {store_stats['mean'].mean():.0f}")
ax.axvline(store_stats["mean"].median(), color="orange", linewidth=1.5, linestyle="--", label=f"Median = {store_stats['mean'].median():.0f}")
ax.set_xlabel("Mean Daily Sales (€)")
ax.set_ylabel("Number of Stores")
ax.set_title("Raw scale")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(store_stats["log_mean"], bins=50, color="#55A868", **STYLE)
ax.axvline(store_stats["log_mean"].mean(), color="red", linewidth=1.5, linestyle="--", label=f"Global mean (log) = {store_stats['log_mean'].mean():.2f}")
ax.set_xlabel("log1p(Mean Daily Sales)")
ax.set_ylabel("Number of Stores")
ax.set_title("Log scale (closer to normal after log1p)")
ax.legend(fontsize=9)
fig.tight_layout()
_save(fig, "distribution_store_mean.png")

# ── Plot 2: CV distribution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
n_high = (store_stats["cv"] > 0.6).sum()
n_med  = ((store_stats["cv"] > 0.4) & (store_stats["cv"] <= 0.6)).sum()
n_low  = (store_stats["cv"] <= 0.4).sum()

colors = ["#2ecc71" if v <= 0.4 else "#f39c12" if v <= 0.6 else "#e74c3c"
          for v in store_stats["cv"]]
ax.hist(store_stats["cv"], bins=40, color="#4C72B0", **STYLE)
ax.axvline(0.4, color="#f39c12", linewidth=1.5, linestyle="--", label=f"CV=0.4  ({n_low} stores below)")
ax.axvline(0.6, color="#e74c3c", linewidth=1.5, linestyle="--", label=f"CV=0.6  ({n_high} stores above)")
ax.set_xlabel("Coefficient of Variation = std / mean")
ax.set_ylabel("Number of Stores")
ax.set_title(
    f"Store Sales Variability — Coefficient of Variation\n"
    f"Low CV ≤0.4: {n_low} stores  |  Medium 0.4–0.6: {n_med}  |  High CV >0.6: {n_high}",
    fontsize=11,
)
ax.legend(fontsize=9)
fig.tight_layout()
_save(fig, "distribution_store_cv.png")

# ── Plot 3: Mean vs Std scatter ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(
    store_stats["mean"], store_stats["std"],
    c=store_stats["cv"], cmap="RdYlGn_r", vmin=0.2, vmax=0.8,
    alpha=0.7, s=20, linewidths=0,
)
plt.colorbar(sc, ax=ax, label="CV (std/mean)")

# label top-10 highest CV stores
top10 = store_stats.nlargest(10, "cv")
for _, row in top10.iterrows():
    ax.annotate(
        f" {int(row['Store'])}",
        (row["mean"], row["std"]),
        fontsize=7, color="#c0392b", alpha=0.85,
    )

# identity line: CV=0.3, 0.5 reference lines
x_ref = np.linspace(0, store_stats["mean"].max() * 1.05, 200)
for cv_ref, color, label in [(0.3, "#2ecc71", "CV=0.3"), (0.5, "#f39c12", "CV=0.5"), (0.7, "#e74c3c", "CV=0.7")]:
    ax.plot(x_ref, cv_ref * x_ref, "--", linewidth=1, color=color, alpha=0.8, label=label)

ax.set_xlabel("Store Mean Daily Sales (€)")
ax.set_ylabel("Store Std Daily Sales (€)")
ax.set_title(
    "Store Mean vs Std  (color = CV)\n"
    "Per-store normalization removes this spread → model learns residual pattern only",
    fontsize=11,
)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
_save(fig, "mean_vs_std_scatter.png")

# ── Plot 4: Box plot – sample 30 stores (low / mid / high mean) ───────────────
n_sample = 30
stores_sorted = store_stats.sort_values("mean")["series_id"].values
# lấy đều từ low → high mean
idx = np.round(np.linspace(0, len(stores_sorted) - 1, n_sample)).astype(int)
sampled_stores = stores_sorted[idx]

plot_data = [train_df[train_df["series_id"] == s]["unit_sales"].values for s in sampled_stores]
means_ordered = [store_stats.loc[store_stats["series_id"] == s, "mean"].values[0] for s in sampled_stores]

fig, ax = plt.subplots(figsize=(16, 5))
bp = ax.boxplot(
    plot_data,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=0.8),
    flierprops=dict(marker=".", markersize=1.5, alpha=0.3),
    showfliers=True,
)

# tô màu theo median
medians = [np.median(d) for d in plot_data]
cmap = plt.cm.RdYlGn
norm = plt.Normalize(min(medians), max(medians))
for patch, med in zip(bp["boxes"], medians):
    patch.set_facecolor(cmap(norm(med)))
    patch.set_alpha(0.75)

ax.set_xticks(range(1, n_sample + 1))
ax.set_xticklabels([f"#{s}\n({m:,.0f})" for s, m in zip(sampled_stores, means_ordered)],
                    fontsize=6.5, rotation=45, ha="right")
ax.set_ylabel("Daily Sales (€)")
ax.set_title(
    f"Sales Distribution for {n_sample} Sampled Stores (sorted by mean)  —  Training Period\n"
    "Color: green=low sales, red=high sales",
    fontsize=11,
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fig.tight_layout()
_save(fig, "boxplot_sample_stores.png")

# ── Plot 5: CDF of store mean ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("CDF of Store Mean Daily Sales — how spread the scale is", fontsize=12, fontweight="bold")

sorted_means = np.sort(store_stats["mean"].values)
cdf = np.arange(1, len(sorted_means) + 1) / len(sorted_means)

ax = axes[0]
ax.plot(sorted_means, cdf, color="#4C72B0", linewidth=1.8)
ax.fill_betweenx(cdf, sorted_means, alpha=0.15, color="#4C72B0")
for pct in [0.1, 0.25, 0.5, 0.75, 0.9]:
    val = np.percentile(sorted_means, pct * 100)
    ax.axvline(val, color="gray", linewidth=0.8, linestyle=":")
    ax.text(val, pct + 0.02, f"P{int(pct*100)}\n{val:,.0f}", fontsize=7.5, ha="center", va="bottom", color="#333")
ax.set_xlabel("Store Mean Daily Sales (€)")
ax.set_ylabel("CDF")
ax.set_title("Raw scale")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax = axes[1]
sorted_log = np.sort(store_stats["log_mean"].values)
ax.plot(sorted_log, cdf, color="#55A868", linewidth=1.8)
ax.fill_betweenx(cdf, sorted_log, alpha=0.15, color="#55A868")
# ratio annotation
ratio_10_90 = np.exp(np.percentile(sorted_log, 90)) / np.exp(np.percentile(sorted_log, 10))
ax.set_xlabel("log1p(Store Mean Sales)")
ax.set_ylabel("CDF")
ax.set_title(f"Log scale  |  P90/P10 ratio = {ratio_10_90:.1f}×  → need per-store norm")
fig.tight_layout()
_save(fig, "cdf_store_mean.png")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"{'STORE VARIANCE SUMMARY':^55}")
print(f"{'─'*55}")
print(f"  Total stores           : {len(store_stats):>6}")
print(f"  Mean sales  min/max    : {store_stats['mean'].min():>8,.0f}  /  {store_stats['mean'].max():>8,.0f}")
print(f"  Scale ratio max/min    : {store_stats['mean'].max()/store_stats['mean'].min():>8.1f}×")
print(f"  Median CV (std/mean)   : {store_stats['cv'].median():>8.3f}")
print(f"  Stores with CV > 0.5   : {(store_stats['cv'] > 0.5).sum():>6}  ({(store_stats['cv'] > 0.5).mean()*100:.0f}%)")
print(f"  P90/P10 mean ratio     : {np.percentile(store_stats['mean'],90)/np.percentile(store_stats['mean'],10):>8.1f}×")
print(f"\n  → Phase B justified if ratio > 5× or >30% stores have CV > 0.5")
print(f"{'─'*55}")
print(f"\nAll charts saved to: {OUT_DIR.resolve()}")
