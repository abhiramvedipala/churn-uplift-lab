"""Simulate a randomized retention-offer experiment on top of Telco customers.

Telco has no treatment column, so this script invents one, openly:

1. Baseline churn probability p0(x): out-of-fold logistic predictions on the
   real churn labels. This keeps the baseline realistic (contract, charges and
   services drive it the way they do in the data).
2. Treatment T ~ Bernoulli(0.5): a randomized trial, so there is no confounding.
3. The offer (a discount) cuts churn by a relative amount r(x):
     r = 0.10                                  everyone a little
       + 0.30 if Contract == Month-to-month    no lock-in, easiest to sway
       + 0.15 if MonthlyCharges > 70           price-sensitive
       + 0.10 if TechSupport == No             unhappy-with-service segment
       - 0.60 if Contract == Two year          "sleeping dogs": the offer reminds
                                               locked-in customers to shop around
   r = 0 if Fiber optic and Electronic check  "lost causes": the highest-risk segment
                                               leaves over service quality, so a
                                               discount does nothing
   p1 = clip(p0 * (1 - r), 0, 1)

   Without the lost-causes rule the effect is proportional to p0, and plain
   churn-risk targeting already captures ~95% of the best possible uplift
   (see DECISIONS.md). Real retention data has lost causes, so v2 adds them.
4. Outcomes: churn ~ Bernoulli(p1 if T else p0).

true_uplift = p0 - p1 is saved per customer (the reduction in churn
probability the offer causes), so models can be scored against ground truth,
which never exists in a real experiment.

Run: python -m churnlab.simulate
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from churnlab.data import FEATURES, TARGET, load
from churnlab.train import make_logistic

SEED = 7
OUT = Path("data/synthetic/treatment.csv")


def relative_effect(df: pd.DataFrame) -> np.ndarray:
    r = (
        0.10
        + 0.30 * (df["Contract"] == "Month-to-month")
        + 0.15 * (df["MonthlyCharges"] > 70)
        + 0.10 * (df["TechSupport"] == "No")
        - 0.60 * (df["Contract"] == "Two year")
    )
    lost_cause = (df["InternetService"] == "Fiber optic") & (
        df["PaymentMethod"] == "Electronic check"
    )
    return r.where(~lost_cause, 0.0).to_numpy()


def simulate(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p0 = cross_val_predict(
        make_logistic(), df[FEATURES], df[TARGET], cv=5, method="predict_proba"
    )[:, 1]
    p1 = np.clip(p0 * (1 - relative_effect(df)), 0, 1)
    t = rng.binomial(1, 0.5, len(df))
    churn = rng.binomial(1, np.where(t == 1, p1, p0))
    return df[FEATURES].assign(treatment=t, churn=churn, p0=p0, p1=p1, true_uplift=p0 - p1)


def main() -> pd.DataFrame:
    sim = simulate(load())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sim.to_csv(OUT, index=False)
    by_t = sim.groupby("treatment")["churn"].mean()
    print(
        f"rows={len(sim)} treated={sim.treatment.mean():.3f} "
        f"churn control={by_t[0]:.3f} treated={by_t[1]:.3f} "
        f"true ATE={sim.true_uplift.mean():.4f} "
        f"share harmed={(sim.true_uplift < 0).mean():.3f}"
    )
    return sim


if __name__ == "__main__":
    main()
