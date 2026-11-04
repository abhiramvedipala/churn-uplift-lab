import numpy as np
import pytest

from churnlab.data import FEATURES, LEAKY_COLUMNS, TARGET, cohort_split, load, random_split
from churnlab.leakage_check import check


@pytest.fixture(scope="module")
def df():
    return load()


def test_load_shape_and_clean(df):
    assert len(df) == 7043
    assert not set(LEAKY_COLUMNS) & set(df.columns)
    assert df.isna().sum().sum() == 0
    assert set(df[TARGET].unique()) == {0, 1}


def test_cohort_split_is_disjoint_by_tenure(df):
    train, test = cohort_split(df)
    assert len(train) + len(test) == len(df)
    assert test["tenure"].max() < train["tenure"].min()


def test_random_split_matches_test_size_and_churn_rate(df):
    _, cohort_test = cohort_split(df)
    _, test = random_split(df)
    assert abs(len(test) - len(cohort_test)) <= 1
    assert abs(test[TARGET].mean() - df[TARGET].mean()) < 0.005


def test_real_features_pass_leakage_check(df):
    train, test = cohort_split(df)
    assert check(train, test, FEATURES) == {}


def test_leakage_check_flags_outcome_copy_and_noisy_score(df):
    rng = np.random.default_rng(0)
    df = df.assign(
        churn_copy=df[TARGET].map({0: "stay", 1: "left"}),
        churn_score=df[TARGET] * 40 + rng.normal(50, 10, len(df)),
    )
    train, test = cohort_split(df)
    assert set(check(train, test, FEATURES + ["churn_copy", "churn_score"])) == {
        "churn_copy",
        "churn_score",
    }
