"""Clean Favorita Grocery Sales Forecasting data: load → join → clean → save.

Wrapper CLI quanh src.data.cleaner.build_dataset(): load tất cả file Favorita
(train + stores/items/oil/holidays/transactions), join metadata, merge exog,
clip unit_sales âm, lọc 1 nhóm store (store_filter), sort → lưu ra cleaned.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cleaner import build_dataset, save, validate
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Clean Favorita Grocery Sales data")
    parser.add_argument("--raw-dir",  default=None,           help="Raw data directory (override config)")
    parser.add_argument("--out-dir",  default="data/cleaned", help="Output directory")
    parser.add_argument("--format",   default="feather", choices=["feather", "csv"],
                        help="feather khuyến nghị: 12.5M dòng sau zero-fill, CSV rất nặng")
    parser.add_argument("--verbose",  action="store_true",    help="Print quality report")
    args = parser.parse_args()

    # build_dataset cần config dict → load qua load_config, override raw_dir nếu có
    config = load_config()
    if args.raw_dir:
        config.setdefault("data", {})["raw_dir"] = args.raw_dir

    print("Loading + cleaning raw data...")
    df = build_dataset(config)
    print(f"  cleaned: {len(df):,} rows")

    if args.verbose:
        print("\n--- Quality report ---")
        validate(df)

    save(df, f"{args.out_dir}/train_cleaned", args.format, config=config)
    print("Done.")


if __name__ == "__main__":
    main()
