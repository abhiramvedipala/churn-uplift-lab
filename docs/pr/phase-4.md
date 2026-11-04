## Phase 4: serving, results pipeline and docs

### What was built
- `churnlab/api.py`: FastAPI `POST /predict` returning churn probability, risk decile and top 3 SHAP drivers; strict Pydantic schema (enums, ranges, `extra="forbid"`).
- Dockerfile (python:3.12-slim, core deps only, non-root) and CI running `scripts/check.sh` plus the image build.
- `churnlab/results.py` builds RESULTS.md from metrics files and test counts; `scripts/check.sh` lints, regenerates every number and runs all tests.
- Budget table of customers kept by offer coverage, targeting memo, README, DECISIONS.md, interview notes, progress notes.

### How it was tested
- 21 API tests: response shape, decile bounds, driver ordering, validation errors, unknown fields rejected.
- `scripts/check.sh` green: lint, leakage gate, all pipelines, 57 tests.
- Docker image built locally and `/predict` smoke-tested from the container.

### Numbers
- Offers to 20% of customers: uplift forest keeps 354, churn risk 174, random 102.
- 57 tests in total.

### Open questions
- Image is 1.7 GB because the Linux xgboost wheel bundles NCCL; `xgboost-cpu` would shrink it.
