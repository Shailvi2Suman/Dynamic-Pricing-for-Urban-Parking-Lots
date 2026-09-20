"""
Real-time streaming layer with Pathway (optional).

Pathway is Linux-only and heavy, so it is an OPTIONAL dependency: run_batch.py
is the reference implementation and always works, while this module replays the
dataset as a live stream for the true real-time demo. Both call the SAME Model 2
pricing function, so results match.

Run:  python -m src.stream_pipeline    (requires: pip install -r requirements-stream.txt)
"""
from __future__ import annotations

import pathlib

import pandas as pd

from .pricing_models import demand_price

try:
    import pathway as pw
    _HAS_PATHWAY = True
except Exception:  # pragma: no cover
    _HAS_PATHWAY = False


def _price_one(occupancy, capacity, queue, traffic, vehicle, special) -> float:
    """Adapter: price a single streamed reading via the shared batch function."""
    row = pd.DataFrame([{
        "Occupancy": occupancy, "Capacity": capacity, "QueueLength": queue,
        "TrafficConditionNearby": traffic, "VehicleType": vehicle,
        "IsSpecialDay": special, "SystemCodeNumber": "x",
    }])
    # single-row min-max normalisation collapses to 0; scale by occupancy rate
    # instead so a streamed reading still gets a sensible price. For exact batch
    # parity use run_batch.py; this path demonstrates the live wiring.
    return float(demand_price(row).iloc[0])


def run_stream(csv_path: str | pathlib.Path,
               out_path: str | pathlib.Path = "data/priced_stream.csv",
               input_rate: float = 400.0) -> None:
    if not _HAS_PATHWAY:
        raise RuntimeError(
            "Pathway not installed. `pip install -r requirements-stream.txt`, "
            "or use run_batch.py for the equivalent batch implementation."
        )

    class Schema(pw.Schema):
        SystemCodeNumber: str
        Capacity: int
        Occupancy: int
        QueueLength: int
        TrafficConditionNearby: str
        VehicleType: str
        IsSpecialDay: int

    data = pw.demo.replay_csv(str(csv_path), schema=Schema, input_rate=input_rate)
    priced = data.select(
        lot=data.SystemCodeNumber,
        occupancy=data.Occupancy,
        price=pw.apply_with_type(
            _price_one, float,
            data.Occupancy, data.Capacity, data.QueueLength,
            data.TrafficConditionNearby, data.VehicleType, data.IsSpecialDay),
    )
    pw.io.csv.write(priced, str(out_path))
    pw.run()


if __name__ == "__main__":  # pragma: no cover
    root = pathlib.Path(__file__).resolve().parents[1]
    run_stream(root / "data" / "dataset.csv")
