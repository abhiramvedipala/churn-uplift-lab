"""API tests. Need a trained model: run `python -m churnlab.train` first."""

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churnlab.api import EXAMPLE, app
from churnlab.data import FEATURES

LOW_RISK = EXAMPLE | {
    "tenure": 70,
    "MonthlyCharges": 25.0,
    "TotalCharges": 1750.0,
    "Contract": "Two year",
    "InternetService": "DSL",
    "PaymentMethod": "Credit card (automatic)",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "Partner": "Yes",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` runs the lifespan, which loads the model
        yield c


def test_health_reports_model_loaded(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_probability_decile_and_three_drivers(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert 1 <= body["decile"] <= 10
    assert len(body["top_drivers"]) == 3
    assert {d["feature"] for d in body["top_drivers"]} <= set(FEATURES)


def test_prediction_matches_model_directly(client):
    model = joblib.load("models/churn_xgb.joblib")
    expected = model.predict_proba(pd.DataFrame([EXAMPLE])[FEATURES])[0, 1]
    got = client.post("/predict", json=EXAMPLE).json()["churn_probability"]
    assert got == pytest.approx(expected, abs=1e-4)


def test_high_risk_customer_scores_above_low_risk(client):
    hi = client.post("/predict", json=EXAMPLE).json()
    lo = client.post("/predict", json=LOW_RISK).json()
    assert hi["churn_probability"] > lo["churn_probability"]
    assert hi["decile"] > lo["decile"]


def test_drivers_sorted_and_directions_match_sign(client):
    drivers = client.post("/predict", json=EXAMPLE).json()["top_drivers"]
    mags = [abs(d["shap"]) for d in drivers]
    assert mags == sorted(mags, reverse=True)
    for d in drivers:
        assert d["direction"] == ("raises churn risk" if d["shap"] > 0 else "lowers churn risk")


def test_predict_is_deterministic(client):
    a = client.post("/predict", json=EXAMPLE).json()
    assert a == client.post("/predict", json=EXAMPLE).json()


def test_tenure_zero_new_customer_is_accepted(client):
    r = client.post("/predict", json=EXAMPLE | {"tenure": 0, "TotalCharges": 0})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "field, value",
    [
        ("Contract", "Monthly"),
        ("tenure", -1),
        ("tenure", 500),
        ("tenure", "abc"),
        ("MonthlyCharges", 0),
        ("TotalCharges", -5),
        ("gender", "Other"),
        ("PaymentMethod", "Cash"),
        ("OnlineSecurity", "Maybe"),
        ("InternetService", None),
    ],
)
def test_invalid_values_are_rejected(client, field, value):
    r = client.post("/predict", json=EXAMPLE | {field: value})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][-1] == field


def test_missing_field_is_rejected(client):
    body = {k: v for k, v in EXAMPLE.items() if k != "Contract"}
    assert client.post("/predict", json=body).status_code == 422


def test_extra_field_like_customer_id_is_rejected(client):
    # Unknown fields are refused so leaky columns can never be smuggled in.
    r = client.post("/predict", json=EXAMPLE | {"customerID": "1234-ABCDE"})
    assert r.status_code == 422


def test_empty_body_is_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


def test_openapi_documents_predict_with_example(client):
    spec = client.get("/openapi.json").json()
    assert "/predict" in spec["paths"]
    assert spec["components"]["schemas"]["Customer"]["example"]["Contract"] == "Month-to-month"
