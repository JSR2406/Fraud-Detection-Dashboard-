# Linux-oriented image for the FastAPI scoring service.
# Streamlit uses the same image with an overridden command (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create a dedicated, non-root runtime user before copying application code.
RUN groupadd --system app \
 && useradd --system --gid app --create-home --home-dir /home/app app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

COPY . .

# Artifacts are written at runtime; data is normally mounted read-only.
RUN mkdir -p /app/artifacts /app/data \
 && chown -R app:app /app

USER app

# 8000 = FastAPI, 8501 = Streamlit (profile/override only).
EXPOSE 8000 8501

# Demo mode keeps `docker compose up` working without a real dataset.
# Set FRAUD_DATA_MODE=real and mount IEEE-CIS files for production training.
ENV FRAUD_DATA_MODE=demo \
    FRAUD_MODEL_PATH=artifacts/candidate.joblib

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
