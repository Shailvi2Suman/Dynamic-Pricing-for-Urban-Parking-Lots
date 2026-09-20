"""
Evaluation — does dynamic pricing actually beat a flat price?

There is no ground-truth on how drivers respond to price, so we evaluate under
an explicit, stated assumption: a simple linear price elasticity. When a lot
charges above base, some demand is deterred; below base, a little is drawn in:

    realized_occupancy_rate = observed_rate * (1 - elasticity * (price/base - 1))

Then per reading:  revenue = price * realized_occupancy(count).
We compare each model against a STATIC $10 price on two axes a city cares about:
  * revenue   (did we earn more?)
  * utilisation smoothness (std of occupancy — dynamic pricing should flatten
    peaks/troughs, easing overcrowding and waste).

INTERVIEW HONESTY: the elasticity is an assumption, not measured. State it. The
value is the *framework* — "here's how I'd quantify impact and what I'd measure
with real data (A/B test on price, observed occupancy response)."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

ELASTICITY_DEFAULT = 0.3   # 10% over base => ~3% less demand (assumption)


def _realized_occupancy(observed_rate, price, elasticity):
    factor = 1 - elasticity * (price / C.BASE_PRICE - 1)
    return (observed_rate * np.clip(factor, 0, None)).clip(0, 1)


def evaluate(priced: pd.DataFrame, elasticity: float = ELASTICITY_DEFAULT) -> pd.DataFrame:
    """Return a per-model revenue/utilisation comparison vs. a static $10 price."""
    d = priced.copy()
    d["obs_rate"] = (d["Occupancy"] / d["Capacity"]).clip(0, 1)

    strategies = {
        "static_$10": pd.Series(C.BASE_PRICE, index=d.index),
        "model1_baseline": d["Price_Model1"],
        "model2_demand": d["Price_Model2"],
        "model3_competitive": d["Price_Model3"],
    }

    rows = []
    for name, price in strategies.items():
        realized = _realized_occupancy(d["obs_rate"], price, elasticity)
        revenue = (price * realized * d["Capacity"]).sum()
        rows.append({
            "strategy": name,
            "total_revenue": round(float(revenue), 0),
            "mean_utilisation": round(float(realized.mean()), 3),
            "utilisation_std": round(float(realized.std()), 3),
        })
    res = pd.DataFrame(rows)
    base_rev = res.loc[res["strategy"] == "static_$10", "total_revenue"].iloc[0]
    res["revenue_vs_static_%"] = round((res["total_revenue"] / base_rev - 1) * 100, 1)
    return res


def summarize(res: pd.DataFrame) -> str:
    """One-line, interview-ready takeaway from the comparison."""
    best = res.sort_values("total_revenue", ascending=False).iloc[0]
    m2 = res.loc[res["strategy"] == "model2_demand"].iloc[0]
    return (
        f"Under an assumed elasticity, the demand model shifts revenue "
        f"{m2['revenue_vs_static_%']:+.1f}% vs a static $10 price and changes "
        f"utilisation smoothness (std {m2['utilisation_std']}). "
        f"Best revenue: {best['strategy']} ({best['revenue_vs_static_%']:+.1f}%)."
    )
