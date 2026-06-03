"""
Store Intelligence System — Health Check Router.

Provides a GET /health endpoint returning system status, uptime,
active cameras, and aggregate event/anomaly counts.
"""

from __future__ import annotations

import time
from datetime import datetime

import structlog
from fastapi import APIRouter, Request

from analytics.store import get_state_store
from api.schemas import (
    ApiResponse,
    ErrorDetail,
    HealthResponse,
    PipelineStatus,
    ResponseMeta,
)
from config.settings import get_settings

logger = structlog.get_logger()
router = APIRouter(tags=["Health"])

# Captured at module-load time — used to compute uptime.
_startup_time: float = time.time()


def _get_request_id(request: Request) -> str:
    """Extract request ID injected by RequestIdMiddleware."""
    return getattr(request.state, "request_id", "unknown")


@router.get("/health", response_model=ApiResponse)
async def health_check(request: Request) -> ApiResponse:
    """
    System health probe.

    Returns uptime, version, pipeline status, active camera count,
    and total event / anomaly tallies.
    """
    try:
        settings = get_settings()
        store = get_state_store()

        cameras = store.get_cameras()
        active_cameras = sum(1 for c in cameras if c.get("status") == "active")

        # Determine overall pipeline status from camera states
        if active_cameras > 0:
            pipeline_status = PipelineStatus.RUNNING
        elif cameras:
            pipeline_status = PipelineStatus.STOPPED
        else:
            pipeline_status = PipelineStatus.STOPPED

        health = HealthResponse(
            status="healthy",
            uptime_seconds=round(time.time() - _startup_time, 2),
            version=settings.APP_VERSION,
            pipeline_status=pipeline_status,
            active_cameras=active_cameras,
            total_events=store.total_events_emitted,
            total_anomalies=len(store.anomalies),
        )

        return ApiResponse(
            status="success",
            data=health.model_dump(),
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("health.check_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="HEALTH_CHECK_FAILED", message=str(exc)),
        )
