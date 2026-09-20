#!/usr/bin/env python
"""
End-to-end batch demo — the one command that proves the whole thing works.

    python run_batch.py                 # uses data/dataset.csv if present
    python run_batch.py --sample        # synthetic data, no CSV needed

Steps: load feed -> price with Models 1/2/3 -> tumbling-window Model 4 ->
save priced CSV + daily JSONL -> render a Bokeh dashboard.
"""
from __future__ import annotations

import argparse
import pathlib

from src.data_loader import load_dataset, make_sample
from src.pricing_models import price_all, tumbling_daily_price
from src.visualize import build_dashboard


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the parking pricing pipeline.")
    ap.add_argument("--sample", action="store_true", help="use synthetic data")
    ap.add_argument("--csv", type=str, default=None, help="path to dataset.csv")
    ap.add_argument("--lot", type=str, default=None, help="lot to plot")
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    if args.sample:
        print("1/4  building synthetic sample feed ...")
        feed = make_sample(n_lots=5, n_days=5)
    else:
        print("1/4  loading dataset ...")
        feed = load_dataset(args.csv)
    print(f"     {len(feed):,} readings across {feed['SystemCodeNumber'].nunique()} lots")

    print("2/4  pricing with Models 1 (baseline) / 2 (demand) / 3 (competitive) ...")
    priced = price_all(feed)
    priced.to_csv(outdir / "priced_output.csv", index=False)

    print("3/4  Model 4: tumbling-window daily prices -> JSONL ...")
    daily = tumbling_daily_price(feed)
    daily.to_json(outdir / "daily_price_final.jsonl", orient="records", lines=True)

    print("4/4  rendering Bokeh dashboard ...")
    html = build_dashboard(priced, lot=args.lot, out_path=outdir / "dashboard.html")

    print(f"\nwrote: {outdir/'priced_output.csv'}, {outdir/'daily_price_final.jsonl'}, {html}")
    print("\ndemand-model price by lot (min/mean/max):")
    print(priced.groupby("SystemCodeNumber")["Price_Model2"]
          .agg(["min", "mean", "max"]).round(2).head())


if __name__ == "__main__":
    main()
