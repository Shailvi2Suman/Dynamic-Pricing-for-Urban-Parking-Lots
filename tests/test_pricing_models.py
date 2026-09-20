"""
Unit tests for the pricing models — run against a small synthetic sample in the
real schema (no dependency on the large dataset.csv, so CI stays fast).

These assert the things that actually matter for a pricing system: price
bounds, monotonicity in occupancy, determinism, tumbling-window shape, and
output integrity.
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_loader import make_sample
from src.pricing_models import (
    baseline_price,
    competitive_price,
    demand_price,
    price_all,
    tumbling_daily_price,
)

LO, HI = C.BASE_PRICE * C.PRICE_FLOOR_MULT, C.BASE_PRICE * C.PRICE_CEIL_MULT


@pytest.fixture(scope="module")
def feed():
    return make_sample(n_lots=3, n_days=2, seed=7)


def test_sample_is_deterministic():
    pd.testing.assert_frame_equal(make_sample(seed=7), make_sample(seed=7))


def test_sample_integrity(feed):
    assert (feed["Occupancy"] <= feed["Capacity"]).all()
    assert feed["QueueLength"].ge(0).all()
    assert not feed.isna().any().any()


@pytest.mark.parametrize("fn", [baseline_price, demand_price, competitive_price])
def test_prices_bounded(feed, fn):
    p = fn(feed)
    assert p.between(LO, HI).all()
    assert not p.isna().any()


def test_demand_reacts_to_queue():
    row = {"SystemCodeNumber": "L1", "Capacity": 100, "Occupancy": 80,
           "Latitude": 26.1, "Longitude": 91.7, "TrafficConditionNearby": "low",
           "VehicleType": "car", "IsSpecialDay": 0,
           "Timestamp": pd.Timestamp("2016-10-04 09:00")}
    # need >1 row per lot for min-max normalisation to have a range
    df = pd.DataFrame([
        {**row, "QueueLength": 0},
        {**row, "QueueLength": 9, "Timestamp": pd.Timestamp("2016-10-04 09:30")},
    ])
    p = demand_price(df)
    assert p.iloc[1] >= p.iloc[0]  # a longer queue should not lower price


def test_baseline_accumulates_within_day():
    base = {"SystemCodeNumber": "L1", "Capacity": 100, "Latitude": 26.1,
            "Longitude": 91.7, "TrafficConditionNearby": "low",
            "VehicleType": "car", "IsSpecialDay": 0, "QueueLength": 0}
    df = pd.DataFrame([
        {**base, "Occupancy": 90, "Timestamp": pd.Timestamp("2016-10-04 08:00")},
        {**base, "Occupancy": 90, "Timestamp": pd.Timestamp("2016-10-04 08:30")},
    ])
    p = baseline_price(df)
    assert p.iloc[0] == C.BASE_PRICE          # first reading of the day = base
    assert p.iloc[1] > p.iloc[0]              # then it accumulates


def test_tumbling_window_shape(feed):
    daily = tumbling_daily_price(feed)
    # one row per lot per day
    assert len(daily) == feed["SystemCodeNumber"].nunique() * 2
    assert daily["daily_price"].between(LO, HI).all()


def test_price_all_columns_and_finite(feed):
    out = price_all(feed)
    for col in ("Price_Model1", "Price_Model2", "Price_Model3"):
        assert col in out.columns
    assert np.isfinite(out[["Price_Model1", "Price_Model2", "Price_Model3"]]
                       .to_numpy()).all()
