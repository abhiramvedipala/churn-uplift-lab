"""Download, verify and clean the IBM Telco Customer Churn dataset.

Run: python -m churnlab.data
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
RAW_PATH = Path("data/raw/telco.csv")

TARGET = "Churn"

# Columns that must never reach a model, and why.
# - customerID: a row identifier; lets a model memorise rows, carries no signal.
# - Churn Label / Churn Value / Churn Score / Churn Reason / Churn Category /
#   CLTV / Customer Status: present in IBM's extended Telco release. They are
#   either the outcome itself, written after the customer left (reason,
#   category, status), or a score IBM fitted on the outcome (Churn Score, CLTV).
#   This CSV does not ship them; we still drop them if they ever appear.
LEAKY_COLUMNS = [
    "customerID",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
    "Churn Category",
    "CLTV",
    "Customer Status",
]

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL

# Tenure stands in for time: a customer with tenure t signed up t months
# before the snapshot. The newest signup cohort (tenure < TEST_TENURE_MAX)
# is held out, so the model is scored on customers "from the future".
TEST_TENURE_MAX = 12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(path: Path = RAW_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    got = sha256(path)
    if got != SHA256:
        raise ValueError(f"hash mismatch for {path}: {got} != {SHA256}")
    return path


def load(path: Path = RAW_PATH) -> pd.DataFrame:
    """Return cleaned features plus a 0/1 Churn column."""
    df = pd.read_csv(download(path))
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
    # 11 customers with tenure 0 have a blank TotalCharges: they have not been
    # billed yet, so 0 is the true value, not a missing one.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df[FEATURES + [TARGET]]


def cohort_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on older signup cohorts, test on the newest one."""
    test = df["tenure"] < TEST_TENURE_MAX
    return df[~test].reset_index(drop=True), df[test].reset_index(drop=True)


def random_split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random split with the same test size, for comparison only."""
    rng = np.random.default_rng(seed)
    n_test = int((df["tenure"] < TEST_TENURE_MAX).sum())
    idx = np.concatenate(
        [
            rng.permutation(np.flatnonzero(df[TARGET] == k))[
                : round(n_test * (df[TARGET] == k).mean())
            ]
            for k in (0, 1)
        ]
    )
    mask = np.zeros(len(df), bool)
    mask[idx] = True
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


if __name__ == "__main__":
    data = load()
    tr, te = cohort_split(data)
    print(f"rows={len(data)} churn_rate={data[TARGET].mean():.3f}")
    print(
        f"train={len(tr)} (churn {tr[TARGET].mean():.3f}) "
        f"test={len(te)} (churn {te[TARGET].mean():.3f})"
    )
