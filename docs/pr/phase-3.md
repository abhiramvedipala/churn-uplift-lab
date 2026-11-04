## Phase 3: A/B testing toolkit

### What was built
- `abtest/power.py`: sample size, power and MDE for proportions (Cohen's h) and means (exact noncentral t).
- `abtest/inference.py`: two-proportion z-test, Welch t-test, CUPED, percentile bootstrap CI.
- `abtest/sequential.py`: A/A peeking simulation and `warn_if_peeking`.

### How it was tested
- 20 tests against independent known results: Cohen's tables, statsmodels, scipy, Armitage et al. (1969), CUPED variance ratio 1 - rho^2, bootstrap coverage near 95%.

### Numbers
- 3,268 customers per arm to detect churn 26.5% -> 23.5% (alpha 0.05, power 0.8).
- 19.4% false-positive rate when an A/A test is checked at 10 looks (nominal 5%).

### Open questions
- Bootstrap is percentile-only; BCa would help for heavily skewed metrics.
