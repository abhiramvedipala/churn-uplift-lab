"""Uplift models on the simulated retention offer, scored against ground truth.

Outcome for the models is retention (1 - churn), so a positive uplift score
means "the offer makes this customer more likely to stay".

- T-learner: one XGBoost classifier on treated rows, one on control rows;
  uplift = P(stay | offer) - P(stay | no offer).
- Uplift random forest (causalml): trees split directly on the difference in
  outcome between treated and control (KL-divergence criterion).

Run: python -m churnlab.uplift
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from causalml.inference.tree import UpliftRandomForestClassifier  # noqa: E402
from causalml.metrics import auuc_score, get_qini, qini_score  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

from churnlab.data import CATEGORICAL, FEATURES, NUMERIC, load  # noqa: E402
from churnlab.explain import shap_by_feature, summary_plot  # noqa: E402
from churnlab.simulate import simulate  # noqa: E402
from churnlab.train import XGB_PARAMS, onehot  # noqa: E402

SEED = 11
TOP = 0.20
BUDGETS = (0.10, 0.20, 0.30, 0.50, 1.00)
DOCS = Path("docs")
METRICS = Path("reports/metrics/uplift.json")
# causalml uses the true-effect column for its curves if present; we point it
# at a missing column so Qini/AUUC use observed outcomes, as in a real test.
NO_TAU = "none"


def encoder() -> ColumnTransformer:
    return ColumnTransformer([("num", "passthrough", NUMERIC), ("cat", onehot(), CATEGORICAL)])


class TLearner:
    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "TLearner":
        self.pre = encoder().fit(X)
        Xt = self.pre.transform(X)
        self.m1 = XGBClassifier(**XGB_PARAMS).fit(Xt[t == 1], y[t == 1])
        self.m0 = XGBClassifier(**XGB_PARAMS).fit(Xt[t == 0], y[t == 0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.pre.transform(X)
        return self.m1.predict_proba(Xt)[:, 1] - self.m0.predict_proba(Xt)[:, 1]

    def control_churn(self, X: pd.DataFrame) -> np.ndarray:
        """P(churn | no offer): the classic churn-risk score, used as a targeting baseline."""
        return 1 - self.m0.predict_proba(self.pre.transform(X))[:, 1]


class UpliftForest:
    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "UpliftForest":
        self.pre = encoder().fit(X)
        self.model = UpliftRandomForestClassifier(
            control_name="control",
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=100,
            max_features=10,
            random_state=SEED,
        )
        self.model.fit(self.pre.transform(X), np.where(t == 1, "treatment", "control"), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self.pre.transform(X))).ravel()


def top_share_uplift(true_uplift: np.ndarray, score: np.ndarray, top: float = TOP) -> float:
    """Mean true uplift among the `top` fraction of customers ranked by score."""
    k = int(round(top * len(score)))
    return float(true_uplift[np.argsort(-score, kind="stable")[:k]].mean())


def plot_qini(curve_df: pd.DataFrame, path: Path) -> None:
    q = get_qini(curve_df, outcome_col="y", treatment_col="w", treatment_effect_col=NO_TAU)
    x = np.linspace(0, 100, len(q))
    plt.style.use("default")  # causalml switches matplotlib to a global style on import
    fig, ax = plt.subplots(figsize=(7, 5))
    for col, label in [("t_learner", "T-learner (2x XGBoost)"),
                       ("uplift_forest", "Uplift random forest")]:
        ax.plot(x, q[col], label=label)
    # Random targeting gains the same total linearly: a straight line to the endpoint.
    ax.plot([0, 100], [0, q["t_learner"].iloc[-1]], "k--", label="Random targeting")
    ax.set(
        xlabel="% of customers targeted (ranked by model)",
        ylabel="incremental customers retained (Qini)",
        title="Qini curves on held-out 30% (observed outcomes)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def explain_uplift(X: pd.DataFrame, uplift: np.ndarray, path: Path) -> None:
    """SHAP on a surrogate tree model fit to the T-learner's uplift scores."""
    reg = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED)
    surrogate = Pipeline([("pre", encoder()), ("clf", reg)]).fit(X, uplift)
    values, _ = shap_by_feature(surrogate, X)
    summary_plot(values, X, "SHAP: what drives predicted uplift (T-learner surrogate)", path)


def main() -> dict:
    sim = simulate(load())
    train, test = train_test_split(
        sim, test_size=0.3, random_state=SEED, stratify=sim["treatment"]
    )
    y_tr, t_tr = 1 - train["churn"].to_numpy(), train["treatment"].to_numpy()

    tl = TLearner().fit(train[FEATURES], t_tr, y_tr)
    uf = UpliftForest().fit(train[FEATURES], t_tr, y_tr)
    scores = {
        "t_learner": tl.predict(test[FEATURES]),
        "uplift_forest": uf.predict(test[FEATURES]),
    }
    baselines = {
        "churn_risk": tl.control_churn(test[FEATURES]),
        "random": np.random.default_rng(SEED).random(len(test)),
    }
    true = test["true_uplift"].to_numpy()
    oracle = top_share_uplift(true, true)

    curve_df = pd.DataFrame(
        {"y": 1 - test["churn"].to_numpy(), "w": test["treatment"].to_numpy(), **scores}
    )
    qini = qini_score(curve_df, outcome_col="y", treatment_col="w", treatment_effect_col=NO_TAU)
    auuc = auuc_score(curve_df, outcome_col="y", treatment_col="w", treatment_effect_col=NO_TAU)

    report = {
        "n_train": len(train),
        "n_test": len(test),
        "true_ate_test": float(true.mean()),
        "oracle_top20_true_uplift": oracle,
    }
    for name, s in {**scores, **baselines}.items():
        captured = top_share_uplift(true, s)
        report[name] = {
            "top20_true_uplift": captured,
            "top20_share_of_oracle": captured / oracle,
            "corr_with_true_uplift": float(np.corrcoef(s, true)[0, 1]),
        }
        if name in scores:
            report[name] |= {"qini": float(qini[name]), "auuc": float(auuc[name])}
        # Expected customers retained if offers go to the top `frac` of the full
        # 7,043-customer base, using mean true uplift of that slice of the test set.
        report[name]["retained_by_budget"] = {
            f"{int(frac * 100)}%": round(top_share_uplift(true, s, frac) * frac * len(sim), 1)
            for frac in BUDGETS
        }

    DOCS.mkdir(exist_ok=True)
    plot_qini(curve_df, DOCS / "qini.png")
    explain_uplift(test[FEATURES], scores["t_learner"], DOCS / "uplift_shap.png")
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
