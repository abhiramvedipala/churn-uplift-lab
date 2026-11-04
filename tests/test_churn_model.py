import numpy as np
import pytest

from churnlab.data import FEATURES, TARGET, cohort_split, load
from churnlab.explain import shap_by_feature, top_drivers
from churnlab.train import evaluate, make_logistic, make_xgb, top_decile_lift


def test_top_decile_lift_perfect_ranking():
    # 10 positives out of 100, all ranked first: top decile is 100% positive -> lift 10.
    y = np.array([1] * 10 + [0] * 90)
    p = np.linspace(1, 0, 100)
    assert top_decile_lift(y, p) == pytest.approx(10.0)


def test_top_decile_lift_random_scores_is_near_one():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 100_000)
    assert top_decile_lift(y, rng.random(100_000)) == pytest.approx(1.0, abs=0.05)


def test_evaluate_perfect_scores():
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc_roc"] == 1.0 and m["auc_pr"] == 1.0


@pytest.fixture(scope="module")
def fitted():
    train, test = cohort_split(load())
    return make_xgb().fit(train[FEATURES], train[TARGET]), train, test


def test_models_beat_chance_on_test_cohort(fitted):
    _, train, test = fitted
    for make in (make_logistic, make_xgb):
        p = make().fit(train[FEATURES], train[TARGET]).predict_proba(test[FEATURES])[:, 1]
        assert evaluate(test[TARGET].to_numpy(), p)["auc_roc"] > 0.7


def test_shap_values_reproduce_model_output(fitted):
    model, _, test = fitted
    values, base = shap_by_feature(model, test)
    assert list(values.columns) == FEATURES
    p = 1 / (1 + np.exp(-(values.sum(axis=1).to_numpy() + base)))
    np.testing.assert_allclose(p, model.predict_proba(test[FEATURES])[:, 1], atol=1e-5)


def test_top_drivers_sorted_by_magnitude(fitted):
    model, _, test = fitted
    drivers = top_drivers(model, test.iloc[[0]], k=3)
    assert len(drivers) == 3
    mags = [abs(d["shap"]) for d in drivers]
    assert mags == sorted(mags, reverse=True)
    assert {d["feature"] for d in drivers} <= set(FEATURES)
