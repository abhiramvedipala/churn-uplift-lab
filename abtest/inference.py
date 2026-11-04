"""Significance tests, CUPED and bootstrap confidence intervals."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    df: float | None = None


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> TestResult:
    """Two-sided z-test for p1 != p2 with the pooled standard error."""
    p1, p2, pooled = x1 / n1, x2 / n2, (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return TestResult(z, float(2 * stats.norm.sf(abs(z))))


def welch_ttest(a: np.ndarray, b: np.ndarray) -> TestResult:
    """Two-sided Welch t-test (unequal variances, Welch-Satterthwaite df)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return TestResult(t, float(2 * stats.t.sf(abs(t), df)), df)


def cuped(y: np.ndarray, x: np.ndarray, theta: float | None = None) -> np.ndarray:
    """CUPED-adjusted metric: y - theta * (x - mean(x)).

    x is a pre-experiment covariate (e.g. last month's spend). theta =
    cov(y, x) / var(x) minimises variance; the adjusted variance is
    var(y) * (1 - corr(x, y)^2) and the mean is unchanged. Pass the same
    theta (estimated on pooled data) to both arms.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    if theta is None:
        theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean())


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray | None = None,
    stat: Callable[..., float] | None = None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for stat(a) or, with b, stat(a, b) (default mean diff)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    if b is None:
        stat = stat or np.mean
        draws = [stat(rng.choice(a, len(a))) for _ in range(n_boot)]
    else:
        b = np.asarray(b, float)
        stat = stat or (lambda u, v: u.mean() - v.mean())
        draws = [stat(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
