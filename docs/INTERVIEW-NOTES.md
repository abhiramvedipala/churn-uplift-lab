# Interview notes

## Phase 1: Churn model

**What it built**
- Hash-verified download of IBM Telco (7,043 rows) with every leaky column documented and dropped.
- Tenure-cohort split: train on older signups, test on the newest year (n=2,069, 48% churn).
- `leakage_check.py`: fails the build if any single feature reaches 0.80 AUC on its own.
- Logistic and XGBoost pipelines scored on AUC-PR, AUC-ROC, top-decile lift and Brier, with MLflow runs and a calibration plot.
- TreeSHAP summary plot plus three per-customer explanations.

**Key decisions**
- Cohort split over random. *Why not random?* It mixes the same kinds of customers into train and test, so it reports 0.84 AUC-ROC where the honest number is 0.76.
- Logistic actually wins (AUC-PR 0.746 vs 0.728). I report that instead of hiding it.

**Hardest problem:** making sense of AUC-PR going *up* on the harder split. AUC-PR's floor is the base rate, and the cohort test set churns at 48% vs 26.5% for the random split, so it isn't comparable across splits. AUC-ROC and lift tell the real story.

**Likely questions**
1. *How did you check for leakage?* I listed every column that is the outcome, was written after it, or was fit on it, and dropped them. Then a gate scores each remaining feature alone out of sample and fails over 0.80. A test injects a copy of the outcome and a noisy churn score to prove the gate catches both.
2. *Why is top-decile lift only 1.7x?* With a 48% base rate the maximum possible lift is 1/0.483 = 2.07x, so 1.72x is 83% of the ceiling. On the random split it is 3.0x because the ceiling is 3.8x.
3. *Why is the model miscalibrated?* The training cohorts churn at 17.5% and the test cohort at 48%, so predicted probabilities are too low. That's prior shift. The fix is recalibrating on recent data (Platt or isotonic on the newest cohort), or using the scores only for ranking.
4. *Why serve XGBoost if logistic wins?* The gap is small, and TreeSHAP gives exact, additive per-customer drivers that the API returns. I'd revisit with logistic plus coefficient-based explanations.
5. *What does MLflow give you here?* Every run's params and metrics for both splits sit in one store. When I change a feature or a parameter I can compare runs instead of trusting memory.

**Files to open:** `churnlab/data.py`, `churnlab/leakage_check.py`, `churnlab/train.py`, `churnlab/explain.py`, `docs/calibration.png`, `docs/shap_summary.png`.

## Phase 2: Uplift modeling

**What it built**
- A seeded simulation of a randomized retention offer with persuadables, sleeping dogs and lost causes, with the true per-customer uplift saved.
- T-learner (two XGBoost models) and causalml's uplift random forest.
- Qini/AUUC on observed outcomes, plus true uplift captured in the top 20%, against churn-risk and random baselines.
- Budget table: expected customers kept at 10-100% offer coverage. SHAP on a surrogate model explains what drives uplift.

**Key decisions**
- Saving the ground truth makes the models scoreable. *Why not use real churn as the outcome?* There is no real treatment, so any "effect" would be invented anyway; it's better to invent it openly and score against it.
- Qini uses observed outcomes only, as it would in a real experiment.

**Hardest problem:** v1 of the simulation made uplift proportional to risk, and churn-risk targeting captured 94.7% of the oracle. The uplift models looked pointless. Adding a lost-causes segment (high risk, unaffected by a discount) is what separates "likely to leave" from "persuadable". After that, risk targeting got 45% and the uplift forest 92%. I kept both versions in git and DECISIONS.md.

**Likely questions**
1. *What is uplift modeling?* It predicts the *change* in outcome caused by a treatment for each customer, not the outcome itself. You target the people the offer moves, not the people likely to leave.
2. *T-learner vs uplift forest?* The T-learner fits separate outcome models and subtracts them, so noise in each model adds up in the difference. The uplift forest splits directly on the treatment-control difference. Here the forest captured 92% of the oracle vs 82%.
3. *What is a Qini curve?* Rank customers by predicted uplift. At each cut, it shows incremental retained customers: treated responders minus control responders scaled to the treated group's size. The area above the random line is the Qini coefficient.
4. *How would you do this without simulation?* Run a randomized pilot (an offer to a random half), train the uplift model on it, then validate on a second randomized holdout using Qini. Ground truth never exists, so Qini is the metric.
5. *Who are sleeping dogs?* Customers the treatment makes *worse*. Here, two-year contract customers reminded to shop around. Risk targeting can't find them; uplift models can, and you exclude them.

**Files to open:** `churnlab/simulate.py`, `churnlab/uplift.py`, `docs/qini.png`, `docs/uplift_shap.png`, `DECISIONS.md` section 5.

## Phase 3: A/B testing toolkit

**What it built**
- Power, sample size and MDE for proportions (Cohen's h) and means (exact noncentral t).
- Two-proportion z-test and Welch t-test, written from the formulas.
- CUPED variance reduction and percentile bootstrap CIs.
- An A/A peeking simulation plus `warn_if_peeking`.
- 20 tests against known results: Cohen's tables, statsmodels, scipy, Armitage (1969).

**Key decisions**
- Formula-first code so the reference libraries in the tests are independent.
- Peeking is simulated with cumulative sums of batch differences: 20k simulations in milliseconds.

**Hardest problem:** the noncentral t returns NaN at very large n, which broke the root-finder for small effects. The fix brackets the search near the normal-approximation answer.

**Likely questions**
1. *How does CUPED work?* It subtracts θ·(x − x̄), where x is a pre-period covariate and θ = cov(y,x)/var(x). The mean is unchanged and the variance shrinks by a factor of 1 − ρ². A test checks ρ = 0.7 gives a ratio of 0.51.
2. *Why is peeking bad?* Each look is another chance to cross the threshold by luck. At 10 looks the false-positive rate is 19.4%, not 5%. The fix is a fixed horizon, alpha spending or always-valid inference.
3. *Welch vs Student t?* Welch doesn't assume equal variances and loses almost nothing when they are equal, so it's the safe default.
4. *How big a test for 26.5% → 23.5% churn?* 3,268 per arm at α = 0.05, power 0.8.
5. *When use bootstrap?* For skewed metrics or statistics without a clean variance formula (medians, ratios). The coverage test shows ~95% on exponential data.

**Files to open:** `abtest/power.py`, `abtest/inference.py`, `abtest/sequential.py`, `tests/test_abtest.py`.

## Phase 4: Serving and memo

**What it built**
- FastAPI `POST /predict` returning probability, risk decile and top 3 SHAP drivers.
- Pydantic schema with enums, ranges and `extra="forbid"`.
- A Docker image (core deps only, non-root) and 21 API tests.
- A one-page targeting memo, every number from RESULTS.md.
- `scripts/check.sh` and CI that regenerate every number.

**Key decisions**
- Deciles are cut on the test cohort's out-of-sample scores, not training scores.
- Unknown fields are rejected so leaky columns can't be sent in.

**Hardest problem:** making the memo's numbers traceable. The sample size and peeking rate were first written by hand, then moved into `results.py` so RESULTS.md produces them.

**Likely questions**
1. *How do you validate input?* Every categorical field is a `Literal` of the training categories, numerics have ranges, and extra keys are forbidden. Bad input gets a 422 naming the field.
2. *How is the decile computed?* `searchsorted` against nine cut points from the test cohort's predicted probabilities: 1 = lowest risk, 10 = highest.
3. *How would you deploy and monitor it?* Push the image to a registry, run it behind a load balancer and log inputs and scores. Alert on feature drift (PSI per feature) and on the score distribution shifting between deciles.
4. *Why is the image 1.7 GB?* The Linux xgboost wheel bundles NCCL for GPUs. `xgboost-cpu` or exporting to ONNX would cut it a lot.
5. *How did you write for a non-technical reader?* The decision comes first, then one table, the four customer groups in plain words, then caveats. There's no jargon like AUC except one line, explained.

**Files to open:** `churnlab/api.py`, `tests/test_api.py`, `Dockerfile`, `docs/targeting-memo.md`, `churnlab/results.py`.
