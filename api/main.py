"""Application factory for the Fraud Risk Intelligence API."""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import __version__
from api.dependencies import load_predictor
from api.routes.health import router as health_router
from api.routes.predict import root_info_router, router as predict_router

logger = logging.getLogger("fraud.api")

# Hard cap on the raw HTTP body of mutating requests.
MAX_REQUEST_BYTES = 1_000_000

_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:8501,http://127.0.0.1:8501"
)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list | None = None,
) -> JSONResponse:
    """Build a consistent JSON error body without stack traces or secrets."""
    content = {
        "detail": details if details is not None else message,
        "error": {
            "code": code,
            "message": message,
            "status": status_code,
            "request_id": _request_id(request),
        },
    }
    if details is not None:
        content["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=content)


def _cors_origins() -> list[str]:
    raw = os.getenv("FRAUD_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS)
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    # Never allow a wildcard, with or without credentials.
    origins = [origin for origin in origins if origin != "*"]
    return origins or ["http://localhost:3000"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load the model; a missing artifact must not stop the app."""
    if load_predictor():
        logger.info("Fraud model loaded; service is ready.")
    else:
        logger.warning("Fraud model artifact unavailable; readiness will report 503.")
    yield


def create_app() -> FastAPI:
    """Build a fresh configured FastAPI application."""
    app = FastAPI(
        title="Fraud Risk Intelligence API",
        version=__version__,
        description="Real-time, authenticated scoring for fraud risk intelligence.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Middlewares are wrapped in reverse registration order, so registering the
    # size limit first makes request_context the outer layer. That guarantees
    # every response (including early 413s) carries an X-Request-ID.
    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        if request.method not in {"POST", "PUT", "PATCH"}:
            return await call_next(request)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
            return _error_response(
                request, 413, "REQUEST_TOO_LARGE", "Request body exceeds the allowed size."
            )

        body = b""
        too_large = False
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                break
            body += message.get("body", b"")
            if len(body) > MAX_REQUEST_BYTES:
                too_large = True
                break
            if not message.get("more_body", False):
                break
        if too_large:
            return _error_response(
                request, 413, "REQUEST_TOO_LARGE", "Request body exceeds the allowed size."
            )

        async def replay():
            return {"type": "http.request", "body": body, "more_body": False}

        request._receive = replay  # type: ignore[attr-defined]
        return await call_next(request)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "").strip()
        request_id = (
            incoming[:128] if incoming and incoming.isprintable() and incoming.isascii()
            else uuid.uuid4().hex
        )
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled API error", extra={"request_id": request_id})
            response = _error_response(
                request, 500, "INTERNAL_ERROR", "An unexpected error occurred."
            )
        response.headers["X-Request-ID"] = request_id
        return response

    # Added last so CORS headers are present on error responses as well.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        details = [
            {
                "loc": list(error.get("loc", [])),
                "msg": error.get("msg", "Invalid value"),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        return _error_response(
            request, 422, "VALIDATION_ERROR", "Request validation failed.", details
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        code = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            413: "REQUEST_TOO_LARGE",
            422: "VALIDATION_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }.get(exc.status_code, "HTTP_ERROR")
        return _error_response(request, exc.status_code, code, message)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("Rejected invalid request", extra={"request_id": _request_id(request)})
        return _error_response(
            request, 422, "INVALID_REQUEST", "The request could not be processed."
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled API error", extra={"request_id": _request_id(request)})
        return _error_response(
            request, 500, "INTERNAL_ERROR", "An unexpected error occurred."
        )

    app.include_router(health_router)
    app.include_router(root_info_router)
    app.include_router(predict_router, prefix="/api/v1")
    return app


app = create_app()
