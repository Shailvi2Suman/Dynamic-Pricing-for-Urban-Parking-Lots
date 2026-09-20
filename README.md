# Dynamic Pricing for Urban Parking Lots

A real-time dynamic-pricing engine for 14 urban parking lots. Prices adjust to
demand from live signals (occupancy, queue length, nearby traffic, special days,
vehicle mix) and to what **competing lots nearby** are charging — with a
tumbling-window model that emits one smoothed price per lot per day.

Built for the IIT Guwahati **Summer Analytics 2025** capstone (Consulting &
Analytics Club × Pathway). Rebuilt here as a clean, tested, reproducible package
with CI, Docker, and a Pathway streaming layer.

---

## Quickstart

```bash
pip install -r requirements.txt
python run_batch.py            # runs on data/dataset.csv
open data/dashboard.html       # Bokeh dashboard for one lot
```

No dataset handy? `python run_batch.py --sample` runs on synthetic data in the
same schema. With Docker: `docker compose up --build`.

---

## The dataset

`data/dataset.csv` — 14 lots × 73 days × 18 readings/day (08:00–16:30, every
30 min). Columns: `SystemCodeNumber, Capacity, Latitude, Longitude, Occupancy,
VehicleType, TrafficConditionNearby, QueueLength, IsSpecialDay,
LastUpdatedDate, LastUpdatedTime`.

**Data integrity:** ~1.3% of rows have occupancy marginally above capacity
(max ratio 1.04). The loader **flags the count and clips to capacity** rather
than crash or let an occupancy rate > 1 leak into pricing — the "catch it at the
door" habit a production feed needs.

---

## The four models

**Model 1 — Baseline linear.** Cumulative: `price += α·(occupancy/capacity)` per
reading, starting from $10. It **resets each day** and is clipped, so it stays
smooth over the full 73-day series (the raw running sum would grow unbounded).

**Model 2 — Demand-based.** A demand score from five features, min-max
normalised **per lot**, mapped to price:

```
demand = 2.0·occ_rate + 1.5·queue − 1.0·traffic + 2.0·special_day + 1.0·vehicle_weight
price  = 10 · (1 + 1.2 · normalised_demand)          # clipped to [0.5×, 2×]
```

**Model 3 — Competitive (scikit-learn NearestNeighbors).** Fit `NearestNeighbors`
on each lot's mean lat/lon to find the nearest rival lots, then nudge Model 2 by
**±5%** vs. the rivals' average price at the same timestamp: cheaper than rivals →
nudge up; pricier → nudge down. Prices stay within the [0.5×, 2×] band.

**Model 4 — Tumbling-window daily aggregation.** One price per lot per day from
within-day demand `(max_occupancy − min_occupancy)/capacity`, emulating a
Pathway daily tumbling window. Written to `data/daily_price_final.jsonl`.

### Assumptions worth stating out loud (interviewers ask)
- Traffic is a demand **dampener** (heavy congestion nearby lowers willingness
  to enter) — one signed coefficient in `config.py`; flip it if you disagree.
- Vehicle weights: car 1.0, bike 0.5, truck 1.5, cycle 0.3. Traffic: low 0.2,
  average 0.5, high 1.0.
- Demand is normalised per lot, so each lot's price spans its own demand range.
- Every price is bounded to **[0.5×, 2× base]** — the brief's smoothness rule and
  a guard against runaway feedback.

---

## Architecture

```
 data_loader ──► pricing_models (Models 1·2·3 + Model 4 window) ──► priced CSV + daily JSONL
        │               ▲                                                    │
        │               │ (same Model-2 function)                            ▼
        └──► stream_pipeline ── Pathway replay ──►                        visualize (Bokeh)
```

Batch and stream call the **same** pricing function, so the real-time path can
never silently disagree with the batch path — one source of pricing truth.

| File | Role |
|------|------|
| `src/config.py` | Every coefficient, weight, and bound in one auditable place |
| `src/data_loader.py` | Load + integrity-clean the real dataset; synthetic sample for tests |
| `src/pricing_models.py` | Models 1–4, pure pandas/numpy/scikit-learn |
| `src/learned_model.py` | Learned demand model (linear + gradient boosting) |
| `src/evaluate.py` | Revenue/utilisation evaluation vs a static price |
| `src/stream_pipeline.py` | Optional Pathway real-time streaming layer |
| `src/visualize.py` | Bokeh dashboard (3 model prices + occupancy driver) |
| `run_batch.py` / `run_analysis.py` | Pricing pipeline / modeling+eval report |
| `tests/` | bounds · monotonicity · determinism · window shape · integrity · model · eval |

---

## Modeling & evaluation (the data-science layer)

```bash
python run_analysis.py        # learned demand model + revenue/utilisation eval
```

**Learned demand model** (`src/learned_model.py`). Model 2's weights are
hand-tuned; here we *learn* how demand (proxied by occupancy rate) responds to
conditions, with a proper train/test split and metrics:

| model | MAE | RMSE | R² |
|-------|-----|------|----|
| LinearRegression (interpretable) | 0.105 | 0.135 | 0.70 |
| GradientBoosting (accuracy) | 0.069 | 0.091 | **0.86** |

Feature effects are exposed too — on the real data, `Capacity`, `hour`, and
`day-of-week` dominate demand, which is a more honest picture than the hand-set
weights alone. (Target is occupancy rate as a *demand proxy* — there's no
ground-truth price — and `QueueLength` correlates with occupancy, so its
importance is read with that caveat.)

**Evaluation** (`src/evaluate.py`). Under an explicit linear price-elasticity
assumption, each pricing model is compared to a static $10 price on revenue and
utilisation smoothness:

| strategy | revenue vs static | utilisation std (lower = smoother) |
|----------|------------------|-------------------------------------|
| static $10 | — | 0.245 |
| Model 2 · demand | **+22%** | 0.209 |
| Model 3 · competitive | +22% | 0.211 |
| Model 1 · baseline | +30% | 0.168 |

Takeaway: dynamic pricing lifts revenue **and** flattens utilisation peaks vs a
flat price. Model 1 earns most (it prices higher) but Model 2 balances revenue
with smoother utilisation — a real trade-off to discuss. **The elasticity is an
assumption, not measured** — with live data you'd estimate it from an A/B test on
price and observed occupancy response. State that up front.

---

## Real-time streaming (optional)

```bash
pip install -r requirements-stream.txt    # Pathway (Linux/macOS, heavy)
python -m src.stream_pipeline             # replays dataset.csv as a live stream
```

Pathway is optional on purpose — the core pipeline, tests, and CI don't depend on
it, so they stay fast and reliable everywhere.

---

## Testing & CI/CD

```bash
ruff check src tests    # lint
pytest -q               # test suite
```

`.github/workflows/ci.yml` runs **lint + full test suite + a pipeline
smoke-test** on every push and PR, across Python 3.10 / 3.11 / 3.12. Nothing
merges to `main` unless it's green — automated testing, validation, and
reliable deployment.

---

## Layout

```
dynamic-pricing-parking/
├── data/dataset.csv          # the real Summer Analytics feed
├── src/                      # config, loader, models, streaming, viz
├── tests/                    # pytest suite
├── run_batch.py              # end-to-end demo
├── requirements.txt / requirements-stream.txt
├── Dockerfile / docker-compose.yml
├── pyproject.toml
└── .github/workflows/ci.yml
```
