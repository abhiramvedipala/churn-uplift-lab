"""Why peeking breaks a fixed-horizon test: an A/A simulation.

Checking a z-test at several interim looks and stopping at the first
p < alpha inflates the false-positive rate well above alpha. Armitage,
McPherson and Rowe (1969) give ~14% for 5 looks and ~19% for 10 looks at
alpha = 0.05. Use a fixed horizon, or a sequential design (alpha spending,
always-valid p-values), if you need to look early.
"""

import warnings

import numpy as np
from scipy import stats


def peeking_false_positive_rate(
    looks: int, n_per_look: int = 200, n_sims: int = 20_000, alpha: float = 0.05, seed: int = 0
) -> float:
    """Share of A/A tests declared significant at any of `looks` equally spaced looks."""
    rng = np.random.default_rng(seed)
    # Each pair (a_i - b_i) of N(0,1) arms is N(0, 2); summing a batch of
    # n_per_look pairs gives N(0, 2 * n_per_look). Cumulative sums give the
    # running z-statistic at every look without storing each sample.
    batch = rng.normal(0, np.sqrt(2 * n_per_look), (n_sims, looks))
    n = n_per_look * np.arange(1, looks + 1)
    z = np.cumsum(batch, axis=1) / np.sqrt(2 * n)
    crit = stats.norm.ppf(1 - alpha / 2)
    return float((np.abs(z) > crit).any(axis=1).mean())


def warn_if_peeking(looks: int, alpha: float = 0.05) -> float:
    """Warn with the simulated false-positive rate when a fixed test is checked > 1 time."""
    if looks <= 1:
        return alpha
    fpr = peeking_false_positive_rate(looks, alpha=alpha, n_sims=5_000)
    warnings.warn(
        f"Checking a fixed-horizon test {looks} times raises the false-positive rate "
        f"from {alpha:.0%} to about {fpr:.0%}. Use a fixed horizon or a sequential design.",
        stacklevel=2,
    )
    return fpr
