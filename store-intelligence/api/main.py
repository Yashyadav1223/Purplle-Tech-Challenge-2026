"""
Store Intelligence System — FastAPI Application Entry Point.

Creates the FastAPI app with lifespan management, mounts all routers
under ``/api/v1``, registers middleware, and exposes a ``/metrics``
endpoint for Prometheus scraping.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from analytics.store import get_state_store
from api.middleware import setup_middleware
from api.routers import analytics, anomalies, cameras, events, health, stores, tracks, video_stream, zones
from config.settings import get_settings
from db.session import close_db, init_db

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════
# Lifespan (startup / shutdown)
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    **Startup**:
    - Initialize the database (create tables if needed).
    - Initialize the state store singleton.
    - Log a startup banner.

    **Shutdown**:
    - Close the database engine.
    - Log a clean shutdown message.
    """
    settings = get_settings()

    # ── Startup ──────────────────────────────────────
    logger.info(
        "app.starting",
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
    )

    await init_db()
    logger.info("db.initialized")

    # Ensure the state store is alive
    store = get_state_store()
    logger.info(
        "state_store.initialized",
        max_event_history=store.event_history.maxlen,
    )

    logger.info(
        "app.ready",
        host=settings.API_HOST,
        port=settings.API_PORT,
    )

    yield  # ← application is running

    # ── Shutdown ─────────────────────────────────────
    await close_db()
    logger.info("db.closed")
    logger.info("app.shutdown_complete")


# ═══════════════════════════════════════════════════════════
# Application Factory
# ═══════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    """Build and return the fully configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Real-time store analytics, footfall tracking, and anomaly detection.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ────────────────────────────────────
    setup_middleware(app)

    # ── API v1 Routers ───────────────────────────────
    api_prefix = "/api/v1"

    app.include_router(health.router,    prefix=api_prefix)
    app.include_router(cameras.router,   prefix=api_prefix)
    app.include_router(analytics.router, prefix=api_prefix)
    app.include_router(events.router,    prefix=api_prefix)
    app.include_router(anomalies.router, prefix=api_prefix)
    app.include_router(tracks.router,    prefix=api_prefix)
    app.include_router(zones.router,     prefix=api_prefix)
    app.include_router(video_stream.router, prefix=api_prefix)
    app.include_router(stores.router,    prefix=api_prefix)

    # ── Prometheus Metrics ───────────────────────────
    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> PlainTextResponse:
        """Expose Prometheus metrics in the standard text format."""
        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    logger.info("app.configured", routers=9, prefix=api_prefix)
    return app


# ── Module-level app instance (used by uvicorn) ─────────
app = create_app()
