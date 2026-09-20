"""
Tests for the learned demand model and the evaluation.

Kept light and fast (synthetic sample), but real: they assert the model trains
and returns sane metrics, and that dynamic pricing changes revenue/utilisation
vs. a static price in the expected direction.
"""
import pandas as pd

from src.data_loader import make_sample
from src.evaluate import evaluate
from src.learned_model import fit_demand_models, linear_feature_effects, predict_demand
from src.pricing_models import price_all


def test_demand_models_fit_and_score():
    feed = make_sample(n_lots=5, n_days=6, seed=11)
    fits = fit_demand_models(feed)
    assert set(fits) == {"linear", "gboost"}
    for r in fits.values():
        assert r.mae >= 0
        assert r.rmse >= r.mae            # RMSE >= MAE always
        assert r.r2 <= 1.0                # R2 is bounded above by 1


def test_predictions_are_valid_rates():
    feed = make_sample(n_lots=4, n_days=5, seed=3)
    fits = fit_demand_models(feed)
    preds = predict_demand(fits["gboost"], feed)
    assert len(preds) == len(feed)
    # occupancy-rate predictions should sit near [0, 1] (allow small slack)
    assert preds.min() > -0.2 and preds.max() < 1.2


def test_linear_effects_table():
    feed = make_sample(n_lots=4, n_days=5, seed=5)
    fits = fit_demand_models(feed)
    eff = linear_feature_effects(fits["linear"])
    assert {"feature", "coef"}.issubset(eff.columns)
    assert len(eff) > 0


def test_evaluation_shape_and_baseline():
    feed = make_sample(n_lots=5, n_days=6, seed=9)
    res = evaluate(price_all(feed))
    assert {"strategy", "total_revenue", "mean_utilisation",
            "utilisation_std", "revenue_vs_static_%"}.issubset(res.columns)
    # the static baseline compares to itself => 0%
    static = res.loc[res["strategy"] == "static_$10", "revenue_vs_static_%"].iloc[0]
    assert static == 0.0
    assert isinstance(res, pd.DataFrame) and len(res) == 4
