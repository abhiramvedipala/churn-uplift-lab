# Progress

## Done
- **Phase 1: churn model.** Hash-verified data, leaky columns documented and dropped, tenure-cohort split, leakage gate, logistic + XGBoost with MLflow, calibration plot, SHAP summary and 3 customer explanations.
- **Phase 2: uplift.** Seeded simulated offer with saved ground truth (v1 → v2 adds lost causes, both in DECISIONS.md), T-learner and uplift forest, Qini/AUUC, top-20% true-uplift capture, budget table, uplift SHAP.
- **Phase 3: A/B toolkit.** Power/sample size/MDE, z-test, Welch, CUPED, bootstrap, peeking simulation and warning; 20 known-result tests.
- **Phase 4: serving.** FastAPI `/predict` with strict Pydantic schema, Docker image (built and smoke-tested locally), 21 API tests, targeting memo.
- `scripts/check.sh` passes: lint, leakage gate, all pipelines, 57 tests, RESULTS.md regenerated.
- README, DECISIONS.md, docs/INTERVIEW-NOTES.md, CI workflow.

## What needs me
- Create the GitHub repo if not done: `gh repo create abhiramvedipala/churn-uplift-lab --private`, then push `main` (not pushed yet). After the first push, confirm the CI badge goes green.
- Update the resume bullets to the measured numbers below; two differ from the targets:
  - XGBoost does **not** beat the logistic baseline on the cohort split (AUC-PR 0.728 vs 0.746).
  - Top-decile lift is 1.72x, capped at 2.07x by the test cohort's 48% churn rate.

## Next (optional)
- Recalibrate the churn model on the newest cohort (prior shift).
- Shrink the Docker image with `xgboost-cpu`.
- Setup step 6 (LLM SDK with token/cost logging) is not used: no phase needed an LLM call, so I didn't add one.
