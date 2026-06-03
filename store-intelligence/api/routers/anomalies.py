"""
Store Intelligence System — Anomalies Router.

Provides endpoints to list, filter, and resolve anomaly records.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog
from fastapi import APIRouter, Query, Request

from analytics.store import get_state_store
from api.schemas import (
    AnomalyType,
    ApiResponse,
    ErrorDetail,
    ResolveRequest,
    ResponseMeta,
    Severity,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/anomalies", tags=["Anomalies"])


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@router.get("", response_model=ApiResponse)
async def list_anomalies(
    request: Request,
    severity: Optional[Severity] = Query(default=None, description="Filter by severity"),
    type: Optional[AnomalyType] = Query(default=None, alias="type", description="Filter by anomaly type"),
    start: Optional[datetime] = Query(default=None, description="Start of time range (ISO 8601)"),
    end: Optional[datetime] = Query(default=None, description="End of time range (ISO 8601)"),
    resolved: Optional[bool] = Query(default=None, description="Filter by resolution status"),
) -> ApiResponse:
    """
    List anomaly records with optional filters.

    Filters:
    - **severity**: LOW / MEDIUM / HIGH
    - **type**: anomaly type enum value
    - **start** / **end**: ISO-8601 time window
    - **resolved**: ``true`` or ``false``
    """
    try:
        store = get_state_store()

        anomalies = store.get_anomalies_filtered(
            severity=severity.value if severity else None,
            anomaly_type=type.value if type else None,
            resolved=resolved,
        )

        # Apply time-range filter on top (store doesn't natively support it)
        if start:
            start_iso = start.isoformat()
            anomalies = [
                a for a in anomalies if a.get("timestamp", "") >= start_iso
            ]
        if end:
            end_iso = end.isoformat()
            anomalies = [
                a for a in anomalies if a.get("timestamp", "") <= end_iso
            ]

        return ApiResponse(
            status="success",
            data=anomalies,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("anomalies.list_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="ANOMALY_LIST_FAILED", message=str(exc)),
        )


@router.get("/active", response_model=ApiResponse)
async def list_active_anomalies(request: Request) -> ApiResponse:
    """Return all currently unresolved anomalies."""
    try:
        store = get_state_store()
        active = store.get_active_anomalies()

        return ApiResponse(
            status="success",
            data=active,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("anomalies.active_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="ACTIVE_ANOMALIES_FAILED", message=str(exc)),
        )


@router.post("/{anomaly_id}/resolve", response_model=ApiResponse)
async def resolve_anomaly(
    anomaly_id: str,
    body: ResolveRequest,
    request: Request,
) -> ApiResponse:
    """
    Mark an anomaly as resolved.

    Accepts optional resolution notes in the request body.
    """
    try:
        store = get_state_store()
        found = store.resolve_anomaly(anomaly_id, notes=body.notes)

        if not found:
            return ApiResponse(
                status="error",
                meta=ResponseMeta(request_id=_get_request_id(request)),
                error=ErrorDetail(
                    code="ANOMALY_NOT_FOUND",
                    message=f"Anomaly '{anomaly_id}' does not exist.",
                ),
            )

        logger.info("anomaly.resolved", anomaly_id=anomaly_id, notes=body.notes)

        # Return the updated record
        record = store.anomalies.get(anomaly_id)
        return ApiResponse(
            status="success",
            data=record,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("anomalies.resolve_failed", anomaly_id=anomaly_id)
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="ANOMALY_RESOLVE_FAILED", message=str(exc)),
        )
