"""One-off: đo độ thưa (implicit zeros) của Favorita cluster đã lọc.

Trả lời câu hỏi quyết định thiết kế zero-fill: reindex-to-full-range phình bao nhiêu
dòng, có vừa RAM không, và bao nhiêu series bắt đầu bán giữa chừng (cần bound theo
first-observed-sale thay vì global start).
"""

import sys
import numpy as np
import pandas as pd

from src.data.cleaner import build_dataset
from src.utils.config import load_config


def main():
    config = load_config("base", {})
    print("Building dataset (load 125M-row train -> filter cluster)...", flush=True)
    df = build_dataset(config)
    print(f"Filtered rows: {len(df):,} | cols: {len(df.columns)}", flush=True)
    print(f"mem: {df.memory_usage(deep=True).sum()/1e9:.2f} GB", flush=True)

    g = df.groupby(["store_nbr", "item_nbr"])
    n_rows = g.size()
    dmin = g["date"].min()
    dmax = g["date"].max()
    global_min, global_max = df["date"].min(), df["date"].max()
    print(f"\nGlobal date span: {global_min.date()} -> {global_max.date()} "
          f"({(global_max-global_min).days+1} days)", flush=True)

    # span theo từng series (first-observed-sale -> last)
    span_self = (dmax - dmin).dt.days + 1
    span_global = (global_max - global_min).days + 1

    n_series = len(n_rows)
    print(f"\n# series (store,item): {n_series:,}", flush=True)
    print(f"Observed rows total: {int(n_rows.sum()):,}", flush=True)

    # reindex theo span riêng từng series (bound first-sale) vs global
    proj_self = int(span_self.sum())
    proj_global = span_global * n_series
    print(f"\nProjected reindexed rows:", flush=True)
    print(f"  bound first-sale (per-series span): {proj_self:,} "
          f"(x{proj_self/n_rows.sum():.2f} vs observed)", flush=True)
    print(f"  naive global range:                 {proj_global:,} "
          f"(x{proj_global/n_rows.sum():.2f} vs observed)", flush=True)
    print(f"  fake leading zeros avoided: {proj_global-proj_self:,}", flush=True)

    # bao nhiêu series bắt đầu sau global_min (mid-history introductions)
    late = (dmin > global_min).sum()
    print(f"\n# series first-sold AFTER {global_min.date()}: {late:,} "
          f"({100*late/n_series:.1f}%)", flush=True)

    # sparsity: observed / per-series-span
    sparsity = (n_rows.sum() / proj_self)
    print(f"\nFill ratio (observed / per-series-span): {sparsity:.3f} "
          f"-> {100*(1-sparsity):.1f}% days have NO row (implicit zero)", flush=True)

    # ước lượng RAM sau reindex (giả định ~ cùng bytes/row của df hiện tại)
    bytes_per_row = df.memory_usage(deep=True).sum() / len(df)
    print(f"\nEst. RAM after per-series reindex: "
          f"{bytes_per_row*proj_self/1e9:.2f} GB (current bytes/row)", flush=True)


if __name__ == "__main__":
    main()
