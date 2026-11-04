# Serving image: core deps only (no mlflow/shap/causalml).
# Build after training: python -m churnlab.train && docker build -t churn-api .
FROM python:3.12-slim

# XGBoost needs the OpenMP runtime on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY churnlab ./churnlab
COPY abtest ./abtest
RUN pip install --no-cache-dir .

COPY models ./models
ENV MODEL_DIR=/app/models
EXPOSE 8000
USER nobody
CMD ["uvicorn", "churnlab.api:app", "--host", "0.0.0.0", "--port", "8000"]
