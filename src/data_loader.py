"""
Load the real Summer Analytics parking dataset into a canonical frame.

The dataset (14 lots x 73 days x 18 readings) ships in data/dataset.csv with
columns:
    ID, SystemCodeNumber, Capacity, Latitude, Longitude, Occupancy,
    VehicleType, TrafficConditionNearby, QueueLength, IsSpecialDay,
    LastUpdatedDate, LastUpdatedTime

`load_dataset` parses the split date/time into a single Timestamp and sorts by
lot then time — the order pricing depends on. `make_sample` produces a tiny
frame in the SAME schema for fast, file-independent unit tests.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
DEFAULT_CSV = DATA_DIR / "dataset.csv"


def load_dataset(path: str | pathlib.Path | None = None) -> pd.DataFrame:
    """Read the real dataset and return a tidy, time-sorted frame."""
    path = pathlib.Path(path) if path else DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the Summer Analytics dataset.csv "
            f"in data/, or call make_sample() for synthetic data."
        )
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(
        df["LastUpdatedDate"] + " " + df["LastUpdatedTime"], dayfirst=True
    )
    df = df.drop(columns=["LastUpdatedDate", "LastUpdatedTime", "ID"], errors="ignore")
    df = df.sort_values(["SystemCodeNumber", "Timestamp"]).reset_index(drop=True)
    df = _validate_and_clean(df)
    return df


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly on fatal problems; flag-and-fix recoverable data issues.

    The real dataset has ~1.3% of rows with occupancy slightly above capacity
    (max ratio ~1.04 — a marginally over-full lot). Rather than crash or let an
    occupancy rate > 1 leak into pricing, we report the count and clip. This is
    the data-integrity habit a production feed needs: catch it at the door.
    """
    required = {"SystemCodeNumber", "Capacity", "Latitude", "Longitude",
                "Occupancy", "VehicleType", "TrafficConditionNearby",
                "QueueLength", "IsSpecialDay", "Timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")   # fatal
    if df["Capacity"].le(0).any():
        raise ValueError("Integrity: non-positive capacity.")      # fatal

    over = int((df["Occupancy"] > df["Capacity"]).sum())
    if over:
        print(f"[data-integrity] {over} rows ({100*over/len(df):.2f}%) had "
              f"occupancy > capacity — clipping to capacity.")
        df["Occupancy"] = df[["Occupancy", "Capacity"]].min(axis=1)
    return df


def make_sample(n_lots: int = 3, n_days: int = 2, seed: int = 7) -> pd.DataFrame:
    """Small synthetic frame in the real schema, for tests (no CSV needed)."""
    rng = np.random.default_rng(seed)
    base_lat, base_lon = 26.14, 91.73
    rows = []
    start = pd.Timestamp("2016-10-04 08:00:00")
    for L in range(n_lots):
        code = f"LOT{L+1:02d}"
        cap = int(rng.choice([300, 450, 577]))
        lat, lon = base_lat + rng.normal(0, 0.01), base_lon + rng.normal(0, 0.01)
        for d in range(n_days):
            special = int(rng.random() < 0.1)
            for s in range(18):  # 18 readings/day
                ts = start + pd.Timedelta(days=d, minutes=30 * s)
                occ_rate = float(np.clip(0.5 + 0.35 * np.sin(s / 17 * np.pi)
                                         + rng.normal(0, 0.06), 0.05, 0.99))
                rows.append({
                    "SystemCodeNumber": code,
                    "Capacity": cap,
                    "Latitude": lat, "Longitude": lon,
                    "Occupancy": int(occ_rate * cap),
                    "VehicleType": rng.choice(["car", "bike", "truck", "cycle"]),
                    "TrafficConditionNearby": rng.choice(["low", "average", "high"]),
                    "QueueLength": int(max(0, rng.poisson(3) * (occ_rate > 0.85))),
                    "IsSpecialDay": special,
                    "Timestamp": ts,
                })
    return pd.DataFrame(rows).sort_values(
        ["SystemCodeNumber", "Timestamp"]).reset_index(drop=True)
