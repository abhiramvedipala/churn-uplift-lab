"""Train logistic and XGBoost churn models, evaluate, track in MLflow.

Run: python -m churnlab.train
"""

import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from churnlab.data import (  # noqa: E402
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    TARGET,
    cohort_split,
    load,
    random_split,
)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
METRICS_DIR = Path("reports/metrics")
DOCS = Path("docs")
SEED = 42

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=SEED,
)


def onehot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def make_logistic() -> Pipeline:
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", onehot(), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


def make_xgb() -> Pipeline:
    # Trees do not need scaling; numeric columns pass straight through.
    pre = ColumnTransformer([("num", "passthrough", NUMERIC), ("cat", onehot(), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", XGBClassifier(**XGB_PARAMS))])


def top_decile_lift(y: np.ndarray, p: np.ndarray) -> float:
    """Churn rate in the 10% highest-scored customers divided by the overall rate."""
    k = max(1, int(np.ceil(0.1 * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    return float(np.mean(y[top]) / np.mean(y))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc_pr": float(average_precision_score(y, p)),
        "auc_roc": float(roc_auc_score(y, p)),
        "top_decile_lift": top_decile_lift(y, p),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(np.mean(y)),
    }


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, tuple]:
    out = {}
    for name, make in [("logistic", make_logistic), ("xgboost", make_xgb)]:
        model = make().fit(train[FEATURES], train[TARGET])
        p = model.predict_proba(test[FEATURES])[:, 1]
        out[name] = (model, p, evaluate(test[TARGET].to_numpy(), p))
    return out


def plot_calibration(y: np.ndarray, preds: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, p in preds.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", label=name)
    ax.set(
        xlabel="predicted churn probability",
        ylabel="observed churn rate",
        title="Calibration on newest cohort (tenure < 12)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def log_mlflow(results: dict, split: str) -> None:
    import mlflow  # lab-only dependency, not needed to serve

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("churn")
    for name, metrics in results.items():
        with mlflow.start_run(run_name=f"{name}-{split}"):
            mlflow.log_params({"model": name, "split": split, "n_features": len(FEATURES)})
            if name == "xgboost":
                mlflow.log_params(XGB_PARAMS)
            mlflow.log_metrics(metrics)


def main() -> dict:
    df = load()
    report = {}
    for split, (train, test) in [("cohort", cohort_split(df)), ("random", random_split(df))]:
        scored = fit_and_score(train, test)
        metrics = {name: m for name, (_, _, m) in scored.items()}
        report[split] = {**metrics, "n_train": len(train), "n_test": len(test)}
        log_mlflow(metrics, split)
        if split == "cohort":
            y = test[TARGET].to_numpy()
            DOCS.mkdir(exist_ok=True)
            plot_calibration(y, {n: p for n, (_, p, _) in scored.items()}, DOCS / "calibration.png")
            xgb, p_xgb, _ = scored["xgboost"]
            MODEL_DIR.mkdir(exist_ok=True)
            joblib.dump(xgb, MODEL_DIR / "churn_xgb.joblib")
            # Decile cut points come from out-of-sample scores on the test cohort.
            edges = np.quantile(p_xgb, np.linspace(0.1, 0.9, 9)).tolist()
            (MODEL_DIR / "decile_edges.json").write_text(json.dumps(edges))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "churn.json").write_text(json.dumps(report, indent=2))
    for split, models in report.items():
        for name in ("logistic", "xgboost"):
            m = models[name]
            print(
                f"{split:7s} {name:9s} AUC-PR={m['auc_pr']:.3f} AUC-ROC={m['auc_roc']:.3f} "
                f"lift@10%={m['top_decile_lift']:.2f} brier={m['brier']:.3f}"
            )
    return report


if __name__ == "__main__":
    main()
