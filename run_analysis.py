#!/usr/bin/env python
"""
Modeling + evaluation report — the data-science story on top of the pricing engine.

    python run_analysis.py            # uses data/dataset.csv
    python run_analysis.py --sample   # synthetic data, no CSV needed

Fits the learned demand models (linear + gradient boosting), prints metrics and
feature effects, then runs the revenue/utilisation evaluation vs a static price.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.data_loader import load_dataset, make_sample
from src.evaluate import evaluate, summarize
from src.learned_model import (
    fit_demand_models,
    gboost_importances,
    linear_feature_effects,
)
from src.pricing_models import price_all

pd.set_option("display.width", 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    feed = make_sample(n_lots=5, n_days=8) if args.sample else load_dataset(args.csv)
    print(f"loaded {len(feed):,} readings across {feed['SystemCodeNumber'].nunique()} lots\n")

    print("=" * 64)
    print("1) LEARNED DEMAND MODEL  (target: occupancy rate as a demand proxy)")
    print("=" * 64)
    fits = fit_demand_models(feed)
    metrics = pd.DataFrame([{
        "model": r.name, "MAE": round(r.mae, 4),
        "RMSE": round(r.rmse, 4), "R2": round(r.r2, 3)
    } for r in fits.values()])
    print(metrics.to_string(index=False))

    print("\nlinear demand — standardised coefficients (which signals drive demand):")
    print(linear_feature_effects(fits["linear"]).head(7).to_string(index=False))

    print("\ngradient boosting — top feature importances:")
    print(gboost_importances(fits["gboost"]).to_string(index=False))

    print("\n" + "=" * 64)
    print("2) EVALUATION  (revenue + utilisation vs a static $10 price)")
    print("=" * 64)
    priced = price_all(feed)
    res = evaluate(priced)
    print(res.to_string(index=False))
    print("\nTakeaway:", summarize(res))
    print("\nNote: elasticity is an explicit assumption (0.3). With real data you'd")
    print("measure it via an A/B test on price and observed occupancy response.")


if __name__ == "__main__":
    main()
