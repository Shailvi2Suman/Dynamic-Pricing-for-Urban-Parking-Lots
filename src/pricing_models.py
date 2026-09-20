"""
The four dynamic-pricing models, matching the Summer Analytics notebook.

    Model 1  Baseline linear     cumulative price += alpha*(occ/cap) per reading
    Model 2  Demand-based        price = base*(1 + lambda*normalised_demand)
    Model 3  Competitive         Model 2 nudged vs. sklearn-NearestNeighbors rivals
    Model 4  Tumbling window     one daily price per lot from max-min occupancy

All logic is pure pandas/numpy/scikit-learn so it is fast, deterministic, and
unit-testable. The Pathway streaming layer calls the same Model 2 function, so
batch and stream can never disagree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config as C


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _clip(x, base: float = C.BASE_PRICE):
    return np.clip(x, base * C.PRICE_FLOOR_MULT, base * C.PRICE_CEIL_MULT)


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["occ_rate"] = (out["Occupancy"] / out["Capacity"]).clip(0, 1)
    out["traffic"] = out["TrafficConditionNearby"].map(C.TRAFFIC_LEVELS).fillna(0.5)
    out["vehicle_w"] = out["VehicleType"].map(C.VEHICLE_WEIGHTS).fillna(1.0)
    return out


# --------------------------------------------------------------------------
# Model 1 — baseline linear (cumulative, daily reset)
# --------------------------------------------------------------------------
def baseline_price(df: pd.DataFrame) -> pd.Series:
    """Cumulative: price starts at base each day, then += alpha*(occ/cap).

    Reproduces the notebook's running-increment logic, but resets per day and
    clips so the price stays smooth and bounded over the full 73-day series.
    """
    f = _encode(df)
    out = pd.Series(index=df.index, dtype=float, name="Price_Model1")
    day = f["Timestamp"].dt.date if "Timestamp" in f else None
    grouper = [C.LOT_COL, day] if day is not None else [C.LOT_COL]
    for _, idx in f.groupby(grouper).groups.items():
        price = C.BASE_PRICE
        for i in idx:
            out.loc[i] = _clip(price)
            price += C.BASELINE_ALPHA * f.loc[i, "occ_rate"]
    return out


# --------------------------------------------------------------------------
# Model 2 — demand-based
# --------------------------------------------------------------------------
def _raw_demand(f: pd.DataFrame) -> pd.Series:
    d = C.DEMAND
    return (
        d["alpha"] * f["occ_rate"]
        + d["beta"] * f["QueueLength"]
        - d["gamma"] * f["traffic"]
        + d["delta"] * f["IsSpecialDay"]
        + d["epsilon"] * f["vehicle_w"]
    )


def demand_price(df: pd.DataFrame) -> pd.Series:
    """price = base * (1 + lambda * normalised_demand), demand normalised per lot."""
    f = _encode(df)
    f["raw"] = _raw_demand(f)
    # min-max normalise within each lot (matches the notebook's groupby-transform)
    f["norm"] = f.groupby(C.LOT_COL)["raw"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )
    price = C.BASE_PRICE * (1 + C.DEMAND_LAMBDA * f["norm"])
    return pd.Series(_clip(price), index=df.index, name="Price_Model2")


# --------------------------------------------------------------------------
# Model 3 — competitive (scikit-learn NearestNeighbors)
# --------------------------------------------------------------------------
def _neighbor_map(df: pd.DataFrame) -> dict[str, list[str]]:
    """k-NN over per-lot mean lat/lon -> each lot's nearest rival lots."""
    lots = (df.groupby(C.LOT_COL)[["Latitude", "Longitude"]]
            .mean().reset_index())
    k = min(C.NN_NEIGHBORS, len(lots))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(lots[["Latitude", "Longitude"]].to_numpy())
    _, idx = nn.kneighbors(lots[["Latitude", "Longitude"]].to_numpy())
    codes = lots[C.LOT_COL].tolist()
    nmap: dict[str, list[str]] = {}
    for i, code in enumerate(codes):
        nbrs = [codes[j] for j in idx[i] if codes[j] != code]
        nmap[code] = nbrs
    return nmap


def competitive_price(df: pd.DataFrame) -> pd.Series:
    """Nudge Model 2 by +/-5% vs. the average price of nearest-neighbour lots."""
    work = df.copy()
    work["Price_Model2"] = demand_price(df).values
    nmap = _neighbor_map(work)

    out = work["Price_Model2"].copy().rename("Price_Model3")
    for ts, snap in work.groupby("Timestamp"):
        price_by_lot = snap.set_index(C.LOT_COL)["Price_Model2"].to_dict()
        for i, row in snap.iterrows():
            rivals = [price_by_lot[b] for b in nmap.get(row[C.LOT_COL], [])
                      if b in price_by_lot]
            if not rivals:
                continue
            avg = float(np.mean(rivals))
            own = row["Price_Model2"]
            if own < avg:
                out.loc[i] = _clip(own * (1 + C.COMPETITION_STEP))
            elif own > avg:
                out.loc[i] = _clip(own * (1 - C.COMPETITION_STEP))
    return out


# --------------------------------------------------------------------------
# Model 4 — tumbling-window daily aggregation
# --------------------------------------------------------------------------
def tumbling_daily_price(df: pd.DataFrame) -> pd.DataFrame:
    """One price per lot per day from within-day (max-min) occupancy demand.

    Emulates a Pathway daily tumbling window in batch: for each lot/day, demand
    = (max_occupancy - min_occupancy) / capacity, then base*(1 + lambda*demand).
    Returns a tidy daily frame (the '.jsonl' deliverable).
    """
    f = df.copy()
    f["date"] = f["Timestamp"].dt.date
    g = f.groupby([C.LOT_COL, "date"]).agg(
        capacity=("Capacity", "first"),
        occ_max=("Occupancy", "max"),
        occ_min=("Occupancy", "min"),
    ).reset_index()
    demand = (g["occ_max"] - g["occ_min"]) / g["capacity"]
    g["daily_price"] = _clip(C.BASE_PRICE * (1 + C.DAILY_LAMBDA * demand)).round(2)
    g["date"] = g["date"].astype(str)   # readable ISO date in the JSONL output
    return g[[C.LOT_COL, "date", "daily_price"]]


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------
def price_all(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Model 1/2/3 price columns to the input frame."""
    out = df.copy()
    out["Price_Model1"] = baseline_price(df).round(2).values
    out["Price_Model2"] = demand_price(df).round(2).values
    out["Price_Model3"] = competitive_price(df).round(2).values
    return out
