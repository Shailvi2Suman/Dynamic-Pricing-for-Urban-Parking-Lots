"""
Central configuration for the dynamic pricing engine.

Every tunable coefficient and business rule lives here so pricing behaviour is
auditable in one place — a reviewer can see exactly what drives a price without
reading model code. These values match the Summer Analytics 2025 notebook.
"""
from __future__ import annotations

# --- Base pricing ---------------------------------------------------------
BASE_PRICE: float = 10.0          # every model starts from a $10 anchor
PRICE_FLOOR_MULT: float = 0.5     # price never below 0.5x base
PRICE_CEIL_MULT: float = 2.0      # price never above 2.0x base
# Bounding keeps prices smooth/explainable and prevents runaway feedback loops
# (a hard requirement of the problem statement).

# --- Model 1: baseline linear (cumulative within a day) -------------------
BASELINE_ALPHA: float = 2.0
BASELINE_DAILY_RESET: bool = True
# The notebook accumulates price += alpha*(occ/cap) per reading. Left to run
# across all 73 days that grows without bound, so we reset to base at the start
# of each day and clip — smooth and explainable, matching the brief's intent.

# --- Model 2: demand-based ------------------------------------------------
# raw_demand = a*occ_rate + b*queue - c*traffic + d*special + e*vehicle_weight
DEMAND = {
    "alpha":   2.0,   # occupancy rate
    "beta":    1.5,   # queue length
    "gamma":   1.0,   # traffic (subtracted: congestion dampens demand)
    "delta":   2.0,   # special day
    "epsilon": 1.0,   # vehicle weight
}
DEMAND_LAMBDA: float = 1.2   # strength of demand->price (notebook's final value)

# --- Model 3: competitive (scikit-learn NearestNeighbors) -----------------
NN_NEIGHBORS: int = 3        # k nearest lots (includes self, so 2 rivals)
COMPETITION_STEP: float = 0.05   # +/-5% nudge vs. neighbours' average price

# --- Model 4: tumbling-window daily aggregation ---------------------------
DAILY_LAMBDA: float = 1.0    # strength of daily (max-min occupancy) demand

# --- Feature encodings (match the notebook exactly) -----------------------
VEHICLE_WEIGHTS = {"car": 1.0, "bike": 0.5, "truck": 1.5, "cycle": 0.3}
TRAFFIC_LEVELS = {"low": 0.2, "medium": 0.5, "average": 0.5, "high": 1.0}

# --- Canonical schema (as loaded from the real dataset) -------------------
LOT_COL = "SystemCodeNumber"
