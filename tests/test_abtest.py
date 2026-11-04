"""A/B toolkit checked against known results.

References: Cohen (1988) power tables; statsmodels and scipy as independent
implementations; Armitage, McPherson & Rowe (1969) for peeking; CUPED
variance identity var(y_adj) = var(y)(1 - rho^2) from Deng et al. (2013).
"""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from abtest.inference import bootstrap_ci, cuped, two_proportion_ztest, welch_ttest
from abtest.power import (
    mde_proportions,
    power_proportions,
    sample_size_means,
    sample_size_proportions,
)
from abtest.sequential import peeking_false_positive_rate, warn_if_peeking

# --- power and sample size --------------------------------------------------


@pytest.mark.parametrize("d, cohen_n", [(0.2, 393), (0.5, 64), (0.8, 26)])
def test_t_test_sample_size_matches_cohen_table(d, cohen_n):
    # Cohen's table for alpha=.05 two-sided, power=.80 (rounded up).
    assert int(np.ceil(sample_size_means(d))) in (cohen_n, cohen_n + 1)
    assert sample_size_means(d) == pytest.approx(
        TTestIndPower().solve_power(d, nobs1=None, alpha=0.05, power=0.8)
    )


@pytest.mark.parametrize("p1, p2", [(0.10, 0.12), (0.30, 0.25), (0.05, 0.06)])
def test_proportion_sample_size_matches_statsmodels(p1, p2):
    ref = NormalIndPower().solve_power(
        abs(proportion_effectsize(p2, p1)), nobs1=None, alpha=0.05, power=0.8
    )
    assert sample_size_proportions(p1, p2) == pytest.approx(ref, rel=1e-4)


def test_power_matches_statsmodels_and_round_trips():
    ref = NormalIndPower().power(proportion_effectsize(0.12, 0.10), nobs1=3000, alpha=0.05)
    assert power_proportions(0.10, 0.12, 3000) == pytest.approx(ref, rel=1e-9)
    n = sample_size_proportions(0.10, 0.12)
    assert power_proportions(0.10, 0.12, n) == pytest.approx(0.80, abs=1e-3)


def test_mde_inverts_sample_size():
    n = sample_size_proportions(0.10, 0.12)
    assert mde_proportions(0.10, n) == pytest.approx(0.02, abs=1e-6)


# --- hypothesis tests ---------------------------------------------------------


def test_two_proportion_ztest_matches_statsmodels():
    ours = two_proportion_ztest(120, 1000, 150, 1000)
    z, p = proportions_ztest([120, 150], [1000, 1000])
    assert ours.statistic == pytest.approx(z) and ours.p_value == pytest.approx(p)


def test_two_proportion_ztest_textbook_value():
    # 45/100 vs 30/100: pooled p = .375, SE = .06847, z = 2.191, p = .0285
    r = two_proportion_ztest(45, 100, 30, 100)
    assert r.statistic == pytest.approx(2.1909, abs=1e-4)
    assert r.p_value == pytest.approx(0.0285, abs=1e-4)


def test_welch_matches_scipy_with_unequal_variance_and_size():
    rng = np.random.default_rng(1)
    a, b = rng.normal(10, 1, 30), rng.normal(10.8, 4, 80)
    ours, ref = welch_ttest(a, b), stats.ttest_ind(a, b, equal_var=False)
    assert ours.statistic == pytest.approx(ref.statistic)
    assert ours.p_value == pytest.approx(ref.pvalue)
    assert ours.df == pytest.approx(ref.df)


# --- CUPED ----------------------------------------------------------------------


def test_cuped_cuts_variance_by_rho_squared():
    rng = np.random.default_rng(2)
    rho, n = 0.7, 200_000
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)
    ratio = cuped(y, x).var() / y.var()
    assert ratio == pytest.approx(1 - rho**2, abs=0.01)  # 0.51


def test_cuped_keeps_effect_unbiased_and_shrinks_variance():
    rng = np.random.default_rng(3)
    n = 20_000
    x = rng.normal(100, 20, 2 * n)  # pre-period spend
    t = np.repeat([0, 1], n)
    y = 0.8 * x + rng.normal(0, 10, 2 * n) + 2.0 * t  # true effect = 2
    adj = cuped(y, x)
    assert adj[t == 1].mean() - adj[t == 0].mean() == pytest.approx(2.0, abs=0.3)
    rho = np.corrcoef(x, y)[0, 1]
    assert adj.var() / y.var() == pytest.approx(1 - rho**2, abs=0.01)


def test_cuped_with_uncorrelated_covariate_changes_little():
    rng = np.random.default_rng(4)
    y, x = rng.normal(size=50_000), rng.normal(size=50_000)
    assert cuped(y, x).var() / y.var() == pytest.approx(1.0, abs=0.01)


# --- bootstrap ------------------------------------------------------------------


def test_bootstrap_ci_close_to_t_interval_for_normal_mean():
    rng = np.random.default_rng(5)
    a = rng.normal(50, 10, 400)
    lo, hi = bootstrap_ci(a, n_boot=5000)
    t_lo, t_hi = stats.t.interval(0.95, len(a) - 1, a.mean(), stats.sem(a))
    assert lo == pytest.approx(t_lo, abs=0.15) and hi == pytest.approx(t_hi, abs=0.15)


def test_bootstrap_diff_ci_covers_truth_about_95_percent():
    rng = np.random.default_rng(6)
    hits = 0
    for _ in range(200):
        a, b = rng.exponential(2.0, 150), rng.exponential(1.5, 150)  # true diff 0.5
        lo, hi = bootstrap_ci(a, b, n_boot=1000, seed=int(rng.integers(1e9)))
        hits += lo <= 0.5 <= hi
    assert 0.90 <= hits / 200 <= 0.99


# --- peeking --------------------------------------------------------------------


@pytest.mark.parametrize("looks, expected", [(1, 0.05), (5, 0.142), (10, 0.193)])
def test_peeking_inflates_false_positives_like_armitage(looks, expected):
    assert peeking_false_positive_rate(looks) == pytest.approx(expected, abs=0.01)


def test_warn_if_peeking_warns_only_when_looking_more_than_once():
    with pytest.warns(UserWarning, match="false-positive rate"):
        assert warn_if_peeking(5) > 0.1
    assert warn_if_peeking(1) == 0.05
