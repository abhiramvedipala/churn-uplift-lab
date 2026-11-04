"""Fail if any single feature predicts churn suspiciously well.

A real churn driver (contract type, tenure) tops out around 0.75 AUC on its
own. A column at 0.80+ alone almost always means the outcome leaked in
(a churn score, a cancellation date, a "reason" field).

Run: python -m churnlab.leakage_check   (exit code 1 on leakage)
"""

import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

from churnlab.data import FEATURES, TARGET, cohort_split, load

MAX_SINGLE_FEATURE_AUC = 0.80


def single_feature_auc(train: pd.DataFrame, test: pd.DataFrame, col: str) -> float:
    """Out-of-sample AUC of one feature: target-rate encoded on train, scored on test."""
    if pd.api.types.is_numeric_dtype(train[col]):
        score = test[col]
    else:
        rates = train.groupby(col)[TARGET].mean()
        score = test[col].map(rates).fillna(train[TARGET].mean())
    auc = roc_auc_score(test[TARGET], score)
    return max(auc, 1 - auc)  # direction does not matter


def check(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Return {feature: auc} for every feature over the threshold."""
    aucs = {c: single_feature_auc(train, test, c) for c in cols}
    return {c: a for c, a in aucs.items() if a > MAX_SINGLE_FEATURE_AUC}


def main() -> int:
    train, test = cohort_split(load())
    aucs = {c: single_feature_auc(train, test, c) for c in FEATURES}
    for c, a in sorted(aucs.items(), key=lambda kv: -kv[1]):
        print(f"{c:20s} {a:.3f}")
    bad = {c: a for c, a in aucs.items() if a > MAX_SINGLE_FEATURE_AUC}
    if bad:
        print(f"LEAKAGE: {bad}")
        return 1
    print(f"ok: no feature above {MAX_SINGLE_FEATURE_AUC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
