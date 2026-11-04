import numpy as np
import pytest

from churnlab.data import FEATURES, load
from churnlab.simulate import simulate
from churnlab.uplift import TLearner, top_share_uplift


@pytest.fixture(scope="module")
def sim():
    return simulate(load())


def test_simulation_is_deterministic(sim):
    again = simulate(load())
    assert (again["churn"] == sim["churn"]).all()
    assert (again["treatment"] == sim["treatment"]).all()


def test_treatment_is_randomized_half_half(sim):
    assert sim["treatment"].mean() == pytest.approx(0.5, abs=0.02)
    # Randomization: treated and control look alike on baseline risk.
    by_t = sim.groupby("treatment")["p0"].mean()
    assert abs(by_t[1] - by_t[0]) < 0.01


def test_effect_segments_match_documentation(sim):
    two_year = sim["Contract"] == "Two year"
    lost = (sim["InternetService"] == "Fiber optic") & (sim["PaymentMethod"] == "Electronic check")
    assert (sim.loc[two_year & ~lost, "true_uplift"] < 0).all()  # sleeping dogs
    assert (sim.loc[lost, "true_uplift"] == 0).all()  # lost causes
    assert (sim.loc[~two_year & ~lost, "true_uplift"] > 0).all()


def test_top_share_uplift_known_values():
    true = np.array([0.5, 0.4, 0.1, 0.0, -0.2])
    assert top_share_uplift(true, true, top=0.4) == pytest.approx(0.45)
    assert top_share_uplift(true, -true, top=0.4) == pytest.approx(-0.1)


def test_t_learner_ranks_by_true_uplift(sim):
    train, test = sim.iloc[:5000], sim.iloc[5000:]
    t, y = train["treatment"].to_numpy(), 1 - train["churn"].to_numpy()
    tl = TLearner().fit(train[FEATURES], t, y)
    corr = np.corrcoef(tl.predict(test[FEATURES]), test["true_uplift"])[0, 1]
    assert corr > 0.5
