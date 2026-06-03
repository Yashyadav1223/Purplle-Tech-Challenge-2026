"""
Store Intelligence System — Middleware Stack.

Provides:
- RequestIdMiddleware: injects a UUID per request into headers + structlog context
- TimingMiddleware: measures and logs response time, adds X-Response-Time header
- GracefulDegradationMiddleware: catches DB/infra errors → returns 503
- CORS setup via FastAPI CORSMiddleware
- Rate limiting via slowapi (100/minute per IP)
- Prometheus latency histogram
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Histogram
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from config.settings import get_settings

logger = structlog.get_logger()

# ── Prometheus Metrics ──────────────────────────────────────
API_LATENCY = Histogram(
    "api_latency_seconds",
    "API endpoint latency in seconds",
    labelnames=["method", "path", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ── Rate Limiter ────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])


# ═══════════════════════════════════════════════════════════
# Request ID Middleware
# ═══════════════════════════════════════════════════════════

class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique request ID (trace_id) into every request/response cycle.

    - Reads an existing ``X-Request-ID`` header or generates a UUID.
    - Stores it in ``request.state.request_id`` for downstream use.
    - Adds ``X-Request-ID`` to the response headers.
    - Binds it to the structlog context for correlated logging.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # Bind to structlog context for this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=request_id,
            request_id=request_id,
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = request_id
        return response


# ═══════════════════════════════════════════════════════════
# Timing Middleware (with structured logging)
# ═══════════════════════════════════════════════════════════

class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measures request processing time and:
    - Adds ``X-Response-Time`` header (milliseconds).
    - Records the Prometheus ``api_latency_seconds`` histogram.
    - Logs structured request summary with trace_id, store_id, endpoint,
      latency_ms, event_count (for ingest), and status_code.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        duration_ms = round(duration * 1000, 2)

        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        # Prometheus observation
        path = request.url.path
        API_LATENCY.labels(
            method=request.method,
            path=path,
            status=str(response.status_code),
        ).observe(duration)

        # Extract store_id from path if present (e.g., /api/v1/stores/STORE_BLR_002/metrics)
        store_id = None
        path_parts = path.split("/")
        if "stores" in path_parts:
            idx = path_parts.index("stores")
            if idx + 1 < len(path_parts):
                store_id = path_parts[idx + 1]

        # Extract event_count from response header if set by ingest endpoint
        event_count = response.headers.get("X-Event-Count")

        log_data = {
            "method": request.method,
            "path": path,
            "status_code": response.status_code,
            "latency_ms": duration_ms,
            "trace_id": getattr(request.state, "request_id", "unknown"),
        }

        if store_id:
            log_data["store_id"] = store_id
        if event_count is not None:
            log_data["event_count"] = int(event_count)

        # Determine endpoint name for cleaner logs
        endpoint = path.rstrip("/").split("/")[-1] if path else "root"
        log_data["endpoint"] = endpoint

        logger.info("http.request", **log_data)

        return response


# ═══════════════════════════════════════════════════════════
# Graceful Degradation Middleware
# ═══════════════════════════════════════════════════════════

class GracefulDegradationMiddleware(BaseHTTPMiddleware):
    """
    Catches infrastructure exceptions (database errors, connection failures)
    and returns a structured HTTP 503 response instead of raw stack traces.

    This ensures the API never exposes internal errors to clients.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            response = await call_next(request)
            return response
        except ConnectionError as exc:
            logger.error(
                "middleware.connection_error",
                error=str(exc),
                path=request.url.path,
                trace_id=getattr(request.state, "request_id", "unknown"),
            )
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "data": None,
                    "meta": {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "request_id": getattr(request.state, "request_id", "unknown"),
                    },
                    "error": {
                        "code": "SERVICE_UNAVAILABLE",
                        "message": "A required backend service is unavailable. Please retry later.",
                    },
                },
            )
        except OSError as exc:
            # Covers database file errors, socket errors, etc.
            logger.error(
                "middleware.os_error",
                error=str(exc),
                path=request.url.path,
                trace_id=getattr(request.state, "request_id", "unknown"),
            )
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "data": None,
                    "meta": {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "request_id": getattr(request.state, "request_id", "unknown"),
                    },
                    "error": {
                        "code": "SERVICE_UNAVAILABLE",
                        "message": "Database or storage service is unavailable. Please retry later.",
                    },
                },
            )
        except Exception as exc:
            # Catch-all for unexpected errors — never expose raw tracebacks
            logger.exception(
                "middleware.unhandled_error",
                path=request.url.path,
                trace_id=getattr(request.state, "request_id", "unknown"),
            )
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "data": None,
                    "meta": {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "request_id": getattr(request.state, "request_id", "unknown"),
                    },
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred. Please contact support.",
                    },
                },
            )


# ═══════════════════════════════════════════════════════════
# Setup Helpers
# ═══════════════════════════════════════════════════════════

def setup_middleware(app: FastAPI) -> None:
    """
    Register all middleware on the FastAPI application.

    Order matters — middleware executes bottom-to-top for requests,
    top-to-bottom for responses:
    1. CORS (outermost)
    2. Graceful degradation (catches infra errors)
    3. Request-ID injection
    4. Timing / Prometheus
    """
    settings = get_settings()

    # ── CORS ─────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.API_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Custom middleware (added in reverse execution order) ──
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(GracefulDegradationMiddleware)

    # ── Rate Limiter ─────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    logger.info("middleware.configured", cors_origins=settings.API_CORS_ORIGINS)
