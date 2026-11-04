## Phase 1: churn model with leakage-safe evaluation

### What was built
- Hash-verified download of IBM Telco (7,043 rows); leaky columns documented and dropped (`churnlab/data.py`).
- Tenure-cohort split: train on older signups (n=4,974), test on the newest year (n=2,069, 48.3% churn). A random split is kept for comparison.
- `churnlab/leakage_check.py`: fails if any single feature reaches 0.80 out-of-sample AUC.
- Logistic and XGBoost pipelines with MLflow runs, calibration plot (`churnlab/train.py`).
- Exact TreeSHAP via XGBoost `pred_contribs`, summary plot and three customer explanations (`churnlab/explain.py`).

### How it was tested
- 11 tests: loading, both splits, leakage gate catching an injected outcome copy and a noisy score, lift math, metrics, SHAP additivity.
- `python -m churnlab.leakage_check`: best real feature 0.72 AUC, gate passes.

### Numbers
| Split | Model | AUC-PR | AUC-ROC | Top-decile lift |
|---|---|---|---|---|
| cohort | logistic | 0.746 | 0.774 | 1.72x |
| cohort | XGBoost | 0.728 | 0.763 | 1.68x |
| random | logistic | 0.661 | 0.840 | 2.93x |
| random | XGBoost | 0.663 | 0.844 | 3.00x |

### Open questions
- Logistic beats XGBoost on the cohort split; XGBoost is kept for exact SHAP drivers. Worth revisiting.
- Probabilities are miscalibrated on the newest cohort (prior shift, 17.5% vs 48.3% churn).
