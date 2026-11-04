# Design decisions and trade-offs

## 1. Tenure cohort split instead of a random split
Telco is a single snapshot with no dates, so I use tenure as signup time: a customer
with tenure *t* signed up *t* months ago. The test set is the newest cohort (tenure < 12).
This is harsh on purpose: the model never sees a first-year customer in training, and
the test churn rate (48%) is far above train (17.5%), much like a model meeting a new
customer mix in production.
- **Result:** the random split looks better on AUC-ROC (0.84 vs 0.76-0.77) and lift
  (3.0x vs 1.7x). That gap is what a random split hides.
- **Why not X?** A random split is optimistic. A true time split needs event dates the
  dataset does not have.
- **Caveat:** AUC-PR is *higher* on the cohort split (0.75 vs 0.66) only because the
  base rate is higher there. AUC-PR is not comparable across different base rates.

## 2. Leakage gate by single-feature AUC, threshold 0.80
Each feature is target-rate encoded on train and scored on test. Real drivers top out
at 0.72, and an injected outcome copy or noisy churn score scores ~1.0. Out-of-sample
scoring avoids false alarms from rare categories that look predictive in-sample.

## 3. Logistic regression is the honest winner on churn
On the cohort split logistic beats XGBoost on every metric (AUC-PR 0.746 vs 0.728).
With 19 mostly categorical features and a distribution shift, the extra flexibility
of trees does not pay off. I still serve XGBoost because its TreeSHAP explanations are
exact and fast, so the per-prediction drivers are trustworthy. Swapping to logistic
would mean using its coefficient × deviation as the explanation. That is the first
thing to revisit.

## 4. TreeSHAP via XGBoost `pred_contribs`, not the shap package, at serve time
pred_contribs computes the same exact TreeSHAP values. Using it keeps shap (and its
numba stack) out of the serving image. One-hot columns are summed back to raw features;
SHAP is additive, so the grouped values still add up to the model's output
(tested to 1e-5).

## 5. Simulated treatment, with the ground truth saved
There is no treatment in Telco, so I simulate a randomized retention offer on top of
a realistic baseline (out-of-fold logistic churn probabilities). The effect varies by
segment, the seed is fixed, and the true per-customer uplift is saved, so models
can be scored against the truth. A real experiment can never do that.

**v1 → v2 (kept in git history):** in v1 the effect was a relative cut in baseline
risk, so true uplift was proportional to risk. Result: plain churn-risk targeting
captured **94.7%** of the oracle's top-20% uplift (uplift forest 96.6%, T-learner
85.5%; commit `3c00724`). The uplift models had nothing to add. Real retention data
has *lost causes*, high-risk customers a discount cannot save, so v2 adds that
segment (fiber + electronic check get zero effect). Now risk targeting drops to 45%
and the uplift forest reaches 92%. The lesson for an interview: **uplift modeling only
beats risk targeting when "likely to leave" and "persuadable" are different people.**
Check that on a pilot before building an uplift model.

## 6. Qini/AUUC from observed outcomes, truth only for "top 20%"
causalml uses the true-effect column for its curves if it exists. I point it at a
missing column so Qini and AUUC use observed outcomes, the way they would in a real
test. Ground truth is used only for the separate "true uplift captured in top 20%"
metric.

## 7. A/B toolkit written from formulas, checked against libraries
The tests and power functions are implemented from the textbook formulas rather than
wrapping statsmodels. That way statsmodels/scipy act as *independent* references in
the tests. A wrapper tested against itself proves nothing.

## 8. Pydantic `extra="forbid"` on the API
Unknown fields return 422 instead of being ignored. A client sending `customerID` or
a churn score finds out immediately, so a leaky column can't reach the model.

## 9. Research libraries in an optional extra
mlflow, shap and causalml are only needed to train and analyse. The serving image
installs the core deps only.
