"""
Learned demand model — the data-driven upgrade to the hand-tuned coefficients.

Model 2's demand weights are hand-picked. Here we instead *learn* how demand
(proxied by occupancy rate — the observable signal of how wanted a lot is)
responds to conditions, then price off predicted demand. This gives the project
what a data-science interviewer looks for: a real fit, a train/test split, error
metrics, and interpretable feature effects.

Two models are fit:
  * LinearRegression  — interpretable coefficients you can compare to Model 2's
                        hand-tuned weights ("the data agrees queue matters most").
  * GradientBoosting  — a stronger non-linear baseline for accuracy.

Honest note for interviews: the capstone has no ground-truth price, so the
target is occupancy rate as a *demand proxy*, not a labelled demand value.
QueueLength correlates with occupancy (queues form when full), so it is a strong
predictor — call that out rather than hide it; feature importances make it visible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["QueueLength", "traffic", "vehicle_w", "IsSpecialDay", "hour", "dow", "Capacity"]
CATEGORICAL = ["SystemCodeNumber"]
TARGET = "occ_rate"


@dataclass
class FitResult:
    name: str
    mae: float
    rmse: float
    r2: float
    model: Pipeline


def _features(df: pd.DataFrame) -> pd.DataFrame:
    from . import config as C
    f = df.copy()
    f["occ_rate"] = (f["Occupancy"] / f["Capacity"]).clip(0, 1)
    f["traffic"] = f["TrafficConditionNearby"].map(C.TRAFFIC_LEVELS).fillna(0.5)
    f["vehicle_w"] = f["VehicleType"].map(C.VEHICLE_WEIGHTS).fillna(1.0)
    f["hour"] = f["Timestamp"].dt.hour
    f["dow"] = f["Timestamp"].dt.dayofweek
    return f


def _pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


def fit_demand_models(df: pd.DataFrame, test_size: float = 0.25,
                      seed: int = 42) -> dict[str, FitResult]:
    """Fit linear + gradient-boosting demand models; return metrics + fitted pipes."""
    f = _features(df)
    X, y = f[NUMERIC + CATEGORICAL], f[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)

    results: dict[str, FitResult] = {}
    for name, est in [
        ("linear", LinearRegression()),
        ("gboost", GradientBoostingRegressor(random_state=seed)),
    ]:
        pipe = _pipeline(est)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results[name] = FitResult(
            name=name,
            mae=float(mean_absolute_error(yte, pred)),
            rmse=float(root_mean_squared_error(yte, pred)),
            r2=float(r2_score(yte, pred)),
            model=pipe,
        )
    return results


def linear_feature_effects(fit: FitResult) -> pd.DataFrame:
    """Standardised linear coefficients — which signals move demand, and which way."""
    pipe = fit.model
    num_names = NUMERIC
    cat_names = list(pipe.named_steps["pre"]
                     .named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    names = num_names + cat_names
    coefs = pipe.named_steps["model"].coef_
    out = (pd.DataFrame({"feature": names, "coef": coefs})
           .assign(abs=lambda d: d["coef"].abs())
           .sort_values("abs", ascending=False)
           .drop(columns="abs").reset_index(drop=True))
    return out


def gboost_importances(fit: FitResult) -> pd.DataFrame:
    """Gradient-boosting feature importances (numeric features only, readable)."""
    pipe = fit.model
    imp = pipe.named_steps["model"].feature_importances_
    cat_names = list(pipe.named_steps["pre"]
                     .named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    names = NUMERIC + cat_names
    out = (pd.DataFrame({"feature": names, "importance": imp})
           .sort_values("importance", ascending=False)
           .head(8).reset_index(drop=True))
    return out


def predict_demand(fit: FitResult, df: pd.DataFrame) -> np.ndarray:
    """Predicted occupancy-rate (demand proxy) for new rows."""
    return fit.model.predict(_features(df)[NUMERIC + CATEGORICAL])
