# Fraud Detection Dashboard

> Streamlit dashboard + FastAPI scoring service + leakage-aware ML pipeline + local
> monitoring, with Docker and CI scaffolding.

**Status:** This is a working engineering scaffold, **not** a certified fraud-control
system. Demo data is synthetic and is only suitable for UI/pipeline smoke tests.
No real-world fraud-performance number is asserted in this README — inspect
`artifacts/metrics.json` produced by your own run instead.

Author: Janmejay Singh Rathore · Python 3.11 · License: MIT

---

## Start here

| Goal | Entry point |
| --- | --- |
| Explore the synthetic transaction dashboard | [app.py](app.py) |
| Train a model and write evaluation artifacts | [scripts/train.py](scripts/train.py) |
| Inspect the scoring API | [api/main.py](api/main.py) |
| Review the ML pipeline | [fraud_ml](fraud_ml) |
| Reproduce the end-to-end checks | [tests](tests) |

## 1. What is here

| Area | Entry point | Purpose |
|------|-------------|---------|
| Dashboard | `app.py` (Streamlit, port 8501) | Home/EDA/training/prediction pages under `pages/` |
| Scoring API | `api.main:app` (FastAPI, port 8000) | Authenticated single/batch scoring, health and model metadata |
| Data & ML | `fraud_ml/` | Explicit ingestion, validation, temporal splits, features, training, metrics, strict inference |
| Ops extras | `fraud_ml/mlops/`, `fraud_ml/models/` | Standalone tuning, tracking, registry, calibration, business metrics and explanation helpers |
| CLIs | `scripts/train.py`, `scripts/serve.py`, `scripts/monitor.py` | Reproducible training, serving and local monitoring |
| Config | `configs/training.yaml`, `configs/api.yaml`, `FRAUD_*` env vars | Explicit, environment-backed configuration |
| Deploy | `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Non-root Linux image and Compose profiles |
| Quality | `.github/workflows/ci.yml`, `tests/` | Pytest, byte-compile and CLI smoke checks |

```text
  IEEE-CIS files ──► ingestion/validation ──► chronological split
                                                   │
                                  train only ──► preprocessor + models
                                                   │
                        validation ──► threshold/metric selection
                                                   │
                               artifacts/candidate.joblib + metrics.json
                                    │                       │
                       FastAPI :8000 (X-API-Key)     Streamlit :8501
                                    │
                          local JSON drift reports (monitoring/)
```

The final test partition is used only for reporting. Joblib artifacts are
executable serialization: load **only** operator-provisioned, trusted files.

---

## 2. Quick start — demo mode (no dataset needed)

Requires Python 3.11 and pip.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

# 1. Train a candidate on reproducible synthetic demo data.
python scripts/train.py --sample-rows 5000

# 2. Establish a monitoring baseline and produce a report.
python scripts/monitor.py --write-baseline

# 3. Preflight and start the API.
python scripts/serve.py --check
python scripts/serve.py
# equivalent: uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. In a second terminal, start the dashboard.
streamlit run app.py --server.port 8501
```

Default outputs (also mirrored by `configs/*.yaml`):

- `artifacts/candidate.joblib` — persisted pipeline, threshold and feature schema
- `artifacts/metrics.json` — validation/test metrics for every candidate
- `artifacts/monitoring_baseline.json`, `artifacts/monitoring_report.json` — monitoring

Every script supports `--help`, e.g. `python scripts/train.py --help`.

### Real IEEE-CIS data

1. Place `train_transaction.csv` (or `.zip`) and optionally
   `train_identity.csv` (or `.zip`) in `data/`.
2. Set `FRAUD_DATA_MODE=real` (or copy `.env.example` to `.env` and edit it).
3. Run `python scripts/train.py` without `--sample-rows` for the full dataset.

Real mode propagates missing/corrupt-file errors. It never silently falls back
to synthetic data. IEEE-CIS `TransactionDT` is relative elapsed time, not a
calendar timestamp — derived hour/day features carry no calendar meaning.

---

## 3. Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `FRAUD_DATA_MODE` | `demo` | Explicit `demo` (synthetic) or `real` (IEEE-CIS) |
| `FRAUD_DATA_DIR` | `data` | Directory containing real dataset files |
| `FRAUD_MODEL_PATH` | `artifacts/candidate.joblib` | Trusted artifact loaded by API/UI |
| `FRAUD_THRESHOLD` | `0.5` | Default decision cutoff (artifact threshold wins at inference) |
| `FRAUD_API_KEY` | empty | `X-API-Key` value for scoring endpoints; empty returns HTTP 503 |
| `FRAUD_MAX_BATCH_SIZE` | `500` | Maximum batch size accepted by the API |
| `FRAUD_RANDOM_SEED` | `42` | Seed used by ingestion and training |
| `FRAUD_TRAIN_FRACTION` | `0.70` | Chronological train share (library setting) |
| `FRAUD_VALIDATION_FRACTION` | `0.15` | Chronological validation share (library setting) |
| `FRAUD_API_HOST` / `FRAUD_API_PORT` | `0.0.0.0` / `8000` | `scripts/serve.py` only |
| `FRAUD_CORS_ORIGINS` | localhost:3000/8501 | Comma-separated API CORS allowlist |
| `FRAUD_MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow 3 database-backed tracking store (file stores are unsupported) |

`configs/training.yaml` and `configs/api.yaml` document YAML-side defaults.
Precedence differs slightly by CLI: `train.py` uses CLI flags > YAML > `FRAUD_*`
settings; `serve.py` uses CLI > `FRAUD_API_*` env > YAML; `monitor.py` uses CLI >
YAML thresholds/paths > `FRAUD_*` settings for data selection.

A local `.env` file is supported (`cp .env.example .env`); it is gitignored.
Never commit real API keys or model binaries.

---

## 4. API

OpenAPI docs: `http://localhost:8000/docs`.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health/live` | none | Process liveness |
| GET | `/health/ready` | none | 200 only when a loadable artifact exists; 503 otherwise |
| GET | `/model/info` | none | Model name/version, threshold, feature count, data mode |
| POST | `/api/v1/predict` | `X-API-Key` | Score one transaction |
| POST | `/api/v1/predict/batch` | `X-API-Key` | Score a bounded list (or `{"transactions": [...]}`) |

Unknown request fields are rejected, `TransactionAmt` must be finite and
non-negative, and the response is a typed `Prediction` (probability, risk band,
`APPROVE`/`REVIEW`, model version). Artifacts produced by `scripts/train.py`
include validation-fitted score calibration (isotonic by default), but these
probabilities remain estimates — not measured loss probabilities. Check the
artifact's `calibration` report before interpreting scores.

---

## 5. Docker

The image uses `python:3.11-slim`, installs `requirements.txt`, creates a
non-root `app` user, and exposes 8000 (API) and 8501 (dashboard). The default
command is `uvicorn api.main:app --host 0.0.0.0 --port 8000`.

```bash
# API only — works with demo mode and no real dataset.
docker compose up --build

# Include the Streamlit profile.
docker compose --profile dashboard up --build

# Include the optional Redis container (present as infrastructure only).
docker compose --profile cache up --build

# Run a one-off training job using the same image and mounted artifacts.
docker compose run --rm api python scripts/train.py --sample-rows 5000
```

Compose mounts `./artifacts:/app/artifacts` (writable) and `./data:/app/data:ro`
(read-only), sets a `unless-stopped` restart policy, and uses non-privileged
containers with TCP/HTTP liveness checks. `env_file` is optional (Compose v2.24+
supports `required: false`), so `docker compose up` works before `.env` exists.
To run only Streamlit with a different command:

```bash
docker compose run --rm --service-ports streamlit \
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

---

## 6. Tests and CI

```bash
python -m pytest tests/ -q
python -m py_compile app.py scripts/train.py scripts/serve.py scripts/monitor.py
```

`.github/workflows/ci.yml` runs on Python 3.11 with `actions/checkout@v4` and
`actions/setup-python@v5`: install `requirements.txt` + pytest, run the test
suite, byte-compile key files, and verify all three CLI `--help` entry points.
No secrets or external services are required.

---

## 7. Optuna tuning and MLflow tracking

`optuna` and `mlflow` are first-class dependencies in `requirements.txt` and
`pyproject.toml`. The remaining extras are optional:

```bash
python -m pip install -e ".[shap,validation]"   # native SHAP / pandera
```

`configs/training.yaml` enables a 12-trial Optuna TPE search by default. Every
trial evaluates **validation data only** (the final test partition stays sealed)
and the study is persisted to `sqlite:///optuna.db` so trials survive restarts.
Override it from the CLI:

```bash
python scripts/train.py --mode demo --tune --trials 12 --artifact-dir artifacts
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs/models locally
```

Each run logs validation/test metrics, configuration and Optuna best parameters
to `sqlite:///mlflow.db`, uploads `candidate.joblib` + `metrics.json` +
`tuning_study.json`, and logs the fitted sklearn pipeline as a model. Gate-passing
candidates are registered in the MLflow Model Registry; `promote()` / `rollback()`
mirror the local aliases to MLflow aliases. The sklearn model uses cloudpickle
because MLflow 3's default skops serializer rejects sklearn's numpy dtypes —
load only models you trained and trust. If MLflow is misconfigured or absent,
training still succeeds and writes a labelled local JSON record to
`artifacts/ml_tracking/`; if Optuna is absent, a bounded random search is used.
Both fallbacks are reported explicitly in the CLI summary and never silently
pretend to be the real backend.

---

## 8. Honest limitations

1. Demo records are synthetic. Metrics from demo mode say nothing about real fraud detection.
2. No performance figures are promised here; run the pipeline against your own data and read `metrics.json`.
3. Joblib deserialization executes code — treat artifacts as executable and trusted-only.
4. Calibration is fitted on validation data by default, but scores are still estimates; business-cost numbers are assumptions, not measured savings.
5. SHAP-style attributions explain model output, not causal reasons for fraud.
6. `scripts/monitor.py` computes descriptive missing/fraud-rate/amount deltas against a baseline; it is **not** a statistical drift test, alerting system, or model-performance monitor.
7. The Redis Compose service is infrastructure scaffolding; application code does not require it.
8. No production secrets, TLS termination, orchestration manifests or external model registry are configured; MLflow/Optuna use local SQLite stores by default, which are for development only.
9. Scoring is artifact-backed, but prediction history is session-only; confirm a trusted artifact is configured before relying on the UI operationally.

See `docs/ENGINEERING_AUDIT.md` for the original remediation rationale and
`docs/IMPLEMENTATION_REPORT.md` for phase-by-phase status.
