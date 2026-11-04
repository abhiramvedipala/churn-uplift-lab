"""FastAPI service for the XGBoost churn model.

Run: uvicorn churnlab.api:app --port 8000
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from churnlab.data import FEATURES
from churnlab.explain import top_drivers

YesNo = Literal["No", "Yes"]
InternetAddon = Literal["No", "Yes", "No internet service"]

EXAMPLE = {
    "tenure": 3,
    "MonthlyCharges": 89.5,
    "TotalCharges": 268.5,
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    MonthlyCharges: float = Field(gt=0, le=500)
    TotalCharges: float = Field(ge=0, le=100_000)
    gender: Literal["Female", "Male"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: Literal["No", "Yes", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class Driver(BaseModel):
    feature: str
    value: str
    shap: float
    direction: str


class Prediction(BaseModel):
    churn_probability: float
    decile: int = Field(description="1 = lowest risk, 10 = highest, vs the test cohort")
    top_drivers: list[Driver]


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    state["model"] = joblib.load(model_dir / "churn_xgb.joblib")
    state["edges"] = np.array(json.loads((model_dir / "decile_edges.json").read_text()))
    yield
    state.clear()


app = FastAPI(title="Churn model", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    row = pd.DataFrame([customer.model_dump()])[FEATURES]
    p = float(state["model"].predict_proba(row)[0, 1])
    decile = int(np.searchsorted(state["edges"], p, side="right")) + 1
    return Prediction(
        churn_probability=round(p, 4),
        decile=decile,
        top_drivers=top_drivers(state["model"], row, k=3),
    )
