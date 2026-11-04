#!/usr/bin/env bash
# Lint, regenerate every number, run all tests, rebuild RESULTS.md.
# Usage: scripts/check.sh   (from the repo root; uses .venv unless PYTHON is set)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONWARNINGS=ignore

echo "== lint"
"$PY" -m ruff check .
"$PY" -m ruff format --check .

echo "== data + leakage gate"
"$PY" -m churnlab.data
"$PY" -m churnlab.leakage_check

echo "== churn models, SHAP, uplift"
"$PY" -m churnlab.train
"$PY" -m churnlab.explain
"$PY" -m churnlab.simulate
"$PY" -m churnlab.uplift > /dev/null

echo "== tests"
"$PY" -m pytest -q

echo "== results"
"$PY" -m churnlab.results > /dev/null
echo "RESULTS.md updated"
