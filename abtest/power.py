"""Sample size and power for two-arm tests (equal allocation, n is per arm)."""

import math

from scipy import optimize, stats


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions on the arcsine scale."""
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def _z(alpha: float, two_sided: bool) -> float:
    return stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)


def sample_size_proportions(
    p1: float, p2: float, alpha: float = 0.05, power: float = 0.8, two_sided: bool = True
) -> float:
    """Per-arm n to detect p1 -> p2 (normal approximation on Cohen's h).

    The estimate of h has variance 2/n, so the test statistic is shifted by
    h * sqrt(n/2); solving h * sqrt(n/2) = z_alpha + z_beta gives n.
    """
    h = abs(cohens_h(p1, p2))
    if h == 0:
        raise ValueError("p1 and p2 must differ")
    return 2 * ((_z(alpha, two_sided) + stats.norm.ppf(power)) / h) ** 2


def power_proportions(
    p1: float, p2: float, n: float, alpha: float = 0.05, two_sided: bool = True
) -> float:
    """Power of a two-proportion test with n per arm."""
    shift = abs(cohens_h(p1, p2)) * math.sqrt(n / 2)
    z = _z(alpha, two_sided)
    pw = stats.norm.cdf(shift - z)
    if two_sided:
        pw += stats.norm.cdf(-shift - z)
    return float(pw)


def mde_proportions(p1: float, n: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute lift over p1 detectable with n per arm."""
    p2 = optimize.brentq(
        lambda p: sample_size_proportions(p1, p, alpha, power) - n, p1 + 1e-9, 1 - 1e-9
    )
    return p2 - p1


def power_means(d: float, n: float, alpha: float = 0.05) -> float:
    """Exact two-sided power of a two-sample t-test (standardized effect d, n per arm)."""
    df = 2 * n - 2
    nc = abs(d) * math.sqrt(n / 2)
    t = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(t, df, nc) + stats.nct.cdf(-t, df, nc))


def sample_size_means(d: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Per-arm n for a two-sample t-test, solved exactly with the noncentral t."""
    if d == 0:
        raise ValueError("d must be non-zero")
    # Bracket around the normal-approximation answer; the noncentral t is
    # numerically unstable (NaN) for very large n, so never search there.
    approx = 2 * ((_z(alpha, True) + stats.norm.ppf(power)) / d) ** 2
    return optimize.brentq(lambda n: power_means(d, n, alpha) - power, 2, 2 * approx + 10)
