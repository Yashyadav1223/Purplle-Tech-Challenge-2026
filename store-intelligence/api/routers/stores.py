"""
Store Intelligence System — Stores Router.

Implements all spec-required multi-store endpoints:

* ``POST /stores/events/ingest`` — Batch event ingestion with dedup.
* ``GET  /stores/{store_id}/metrics`` — Real-time store KPIs.
* ``GET  /stores/{store_id}/funnel`` — Session-based conversion funnel.
* ``GET  /stores/{store_id}/heatmap`` — Zone heatmap (normalised 0-100).
* ``GET  /stores/{store_id}/anomalies`` — Active anomalies.
* ``GET  /stores/health`` — Per-store feed health & stale-feed warnings.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from analytics.multi_store import get_multi_store_state
from api.schemas import (
    ApiResponse,
    ErrorDetail,
    ResponseMeta,
    SpecEvent,
    SpecEventMetadata,
    IngestResponse,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/stores", tags=["Stores"])


def _get_request_id(request: Request) -> str:
    """Extract request ID injected by RequestIdMiddleware."""
    return getattr(request.state, "request_id", "unknown")


# ═══════════════════════════════════════════════════════════
# Request / Response Schemas (local to this router)
# ═══════════════════════════════════════════════════════════


class IngestError(BaseModel):
    """Per-event error detail returned when an event fails validation."""

    index: int
    event_id: Optional[str] = None
    error: str


# ═══════════════════════════════════════════════════════════
# POST /stores/events/ingest
# ═══════════════════════════════════════════════════════════


@router.post("/events/ingest", response_model=ApiResponse)
async def ingest_events(request: Request) -> ApiResponse:
    """
    Batch-ingest up to 500 spec events.

    The endpoint accepts a JSON body with an ``events`` list.  Each event is
    validated individually so that a partially-valid batch still succeeds
    for the valid subset.  Duplicate ``event_id`` values are silently
    accepted (idempotent).

    **Request body**::

        {"events": [<SpecEvent>, ...]}

    **Response data** is an :class:`IngestResponse` with ``accepted``,
    ``rejected`` counts and per-event ``errors``.
    """
    request_id = _get_request_id(request)

    try:
        body: Dict[str, Any] = await request.json()
    except Exception as exc:
        logger.warning("stores.ingest.invalid_json", error=str(exc))
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=request_id),
            error=ErrorDetail(
                code="INVALID_JSON",
                message=f"Request body is not valid JSON: {exc}",
            ),
        )

    raw_events: Any = body.get("events")
    if not isinstance(raw_events, list):
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=request_id),
            error=ErrorDetail(
                code="INVALID_PAYLOAD",
                message="Request body must contain an 'events' list.",
            ),
        )

    if len(raw_events) > 500:
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=request_id),
            error=ErrorDetail(
                code="BATCH_TOO_LARGE",
                message=f"Maximum batch size is 500, received {len(raw_events)}.",
            ),
        )

    multi_store = get_multi_store_state()
    accepted = 0
    rejected = 0
    errors: List[Dict[str, Any]] = []

    for idx, raw_event in enumerate(raw_events):
        # Step 1: Validate against SpecEvent schema
        try:
            event = SpecEvent.model_validate(raw_event)
        except ValueError as e:
            rejected += 1
            errors.append({
                "index": idx,
                "event_id": raw_event.get("event_id") if isinstance(raw_event, dict) else None,
                "error": str(e)
            })
            continue

        # Step 2: Ingest into multi-store state
        event_dict = event.model_dump(mode="json")
        ok, err_msg = multi_store.ingest_event(event_dict)
        if ok:
            accepted += 1
        else:
            rejected += 1
            errors.append({
                "index": idx,
                "event_id": event.event_id,
                "error": err_msg or "Unknown error"
            })

    result = IngestResponse(accepted=accepted, rejected=rejected, errors=errors)

    logger.info(
        "stores.ingest.complete",
        accepted=accepted,
        rejected=rejected,
        total=len(raw_events),
    )

    return ApiResponse(
        status="success",
        data=result.model_dump(),
        meta=ResponseMeta(request_id=request_id),
    )


# ═══════════════════════════════════════════════════════════
# GET /stores/{store_id}/metrics
# ═══════════════════════════════════════════════════════════


@router.get("/{store_id}/metrics", response_model=ApiResponse)
async def get_store_metrics(store_id: str, request: Request) -> ApiResponse:
    """
    Return today's real-time metrics for *store_id*.

    Metrics include: ``unique_visitors`` (excluding staff),
    ``conversion_rate``, ``avg_dwell_per_zone``, ``queue_depth``,
    ``abandonment_rate``.
    """
    request_id = _get_request_id(request)

    try:
        multi_store = get_multi_store_state()
        metrics = multi_store.get_metrics(store_id)

        return ApiResponse(
            status="success",
            data=metrics,
            meta=ResponseMeta(request_id=request_id),
        )
    except Exception as exc:
        logger.exception("stores.metrics_failed", store_id=store_id)
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=request_id),
            error=ErrorDetail(code="METRICS_FAILED", message=str(exc)),
        )


# ═══════════════════════════════════════════════════════════
# GET /stores/{store_id}/funnel
# ═══════════════════════════════════════════════════════════


@router.get("/{store_id}/funnel", response_model=ApiResponse)
async def get_store_funnel(store_id: str, request: Request) -> ApiResponse:
    """
    Session-based conversion funnel for *store_id*.

    Stages: Entry → Zone Visit → Billing Queue → Purchase.
    Counts unique sessions (``visitor_id``); re-entries do **not**
    double-count.  Returns per-stage counts and drop-off percentages.
    """
    request_id = _get_request_id(request)

    try:
        multi_store = get_multi_store_state()
        funnel = multi_store.get_funnel(store_id)

        return ApiResponse(
            status="success",
            data=funnel,
            meta=ResponseMeta(request_id=request_id),
        )
    except Exception as exc:
        logger.exception("stores.funnel_failed", store_id=store_id)
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=request_id),
            error=ErrorDetail(code="FUNNEL_FAILED", message=str(exc)),
        )


# ═══════════════════════════════════════════════════════════
# GET /stores/{store_id}/heatmap
# ═══════════════════════════════════════════════════════════


@router.get("/{store_id}/heatmap", response_model=ApiResponse)
async def get_store_heatmap(store_id: str, request: Request) -> ApiResponse:
    """
    Zone visit frequency and avg dwell, normalised 0-100.

    Includes ``data_confidence: false`` when fewer than 20 unique sessions
    contribute to the current window.  Data is ready for grid heatmap
    rendering.
    """
    request_id = _get_request_id(request)

    try:
        multi_store = get_multi_store_state()
        heatmap = multi_store.get_heatmap(store_id)

        return ApiResponse(
            status="success",
            data=heatmap,
            meta=ResponseMeta(request_id=request_id),
        )
    except Exception as exc:
        logger.exception("stores.heatmap_failed", store_id=store_id)
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=request_id),
            error=ErrorDetail(code="HEATMAP_FAILED", message=str(exc)),
        )


# ═══════════════════════════════════════════════════════════
# GET /stores/{store_id}/anomalies
# ═══════════════════════════════════════════════════════════


@router.get("/{store_id}/anomalies", response_model=ApiResponse)
async def get_store_anomalies(store_id: str, request: Request) -> ApiResponse:
    """
    Active anomalies detected for *store_id*.

    Anomaly types: ``queue_spike``, ``conversion_drop``, ``dead_zone``.
    Each includes a ``severity`` (INFO / WARN / CRITICAL) and a
    ``suggested_action`` string.
    """
    request_id = _get_request_id(request)

    try:
        multi_store = get_multi_store_state()
        anomalies = multi_store.get_anomalies(store_id)

        return ApiResponse(
            status="success",
            data=anomalies,
            meta=ResponseMeta(request_id=request_id),
        )
    except Exception as exc:
        logger.exception("stores.anomalies_failed", store_id=store_id)
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=request_id),
            error=ErrorDetail(code="STORE_ANOMALIES_FAILED", message=str(exc)),
        )


# ═══════════════════════════════════════════════════════════
# GET /stores/health
# ═══════════════════════════════════════════════════════════


@router.get("/health", response_model=ApiResponse)
async def stores_health(request: Request) -> ApiResponse:
    """
    Multi-store health probe.

    Returns per-store status with ``last_event_timestamp`` and a
    ``STALE_FEED`` warning when a store's last event is more than
    10 minutes old.
    """
    request_id = _get_request_id(request)

    try:
        multi_store = get_multi_store_state()
        stores_health_data = multi_store.get_health_stores()

        overall_status = "healthy"
        for sh in stores_health_data:
            if sh.get("status") == "STALE_FEED":
                overall_status = "degraded"
                break

        return ApiResponse(
            status="success",
            data={
                "status": overall_status,
                "stores": stores_health_data,
            },
            meta=ResponseMeta(request_id=request_id),
        )
    except Exception as exc:
        logger.exception("stores.health_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=request_id),
            error=ErrorDetail(code="STORES_HEALTH_FAILED", message=str(exc)),
        )
