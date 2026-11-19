"""API-key authentication dependency for protected scoring endpoints."""
import secrets
from fastapi import Header, HTTPException
from fraud_ml.config import settings

def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject scoring requests unless a configured key matches in constant time."""
    if not settings.api_key:
        raise HTTPException(status_code=503, detail="API authentication is not configured.")
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key.")
