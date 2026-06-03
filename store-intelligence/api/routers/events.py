"""
Store Intelligence System — Events Router.

Provides a paginated event log endpoint and a WebSocket stream that
delegates to the ConnectionManager for real-time event delivery.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import structlog
from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect

from analytics.store import get_state_store
from api.schemas import (
    ApiResponse,
    ErrorDetail,
    EventType,
    ResponseMeta,
)
from api.websocket import get_connection_manager

logger = structlog.get_logger()
router = APIRouter(prefix="/events", tags=["Events"])


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@router.get("", response_model=ApiResponse)
async def list_events(
    request: Request,
    type: Optional[List[EventType]] = Query(default=None, alias="type", description="Filter by event types"),
    zone: Optional[str] = Query(default=None, description="Filter by zone ID"),
    camera_id: Optional[str] = Query(default=None, description="Filter by camera ID"),
    start: Optional[datetime] = Query(default=None, description="Start of time range (ISO 8601)"),
    end: Optional[datetime] = Query(default=None, description="End of time range (ISO 8601)"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page"),
) -> ApiResponse:
    """
    Paginated event log with optional filtering.

    Filters:
    - **type**: one or more ``EventType`` values (repeated query param)
    - **zone**: zone ID
    - **camera_id**: camera ID
    - **start** / **end**: ISO-8601 time range
    """
    try:
        store = get_state_store()

        # Convert enum list to string values for the store filter
        event_type_strs = [t.value for t in type] if type else None

        events, total = store.get_filtered_events(
            event_types=event_type_strs,
            zone_id=zone,
            camera_id=camera_id,
            start_time=start,
            end_time=end,
            page=page,
            page_size=page_size,
        )

        return ApiResponse(
            status="success",
            data={
                "events": events,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size if page_size else 1,
            },
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("events.list_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="EVENTS_LIST_FAILED", message=str(exc)),
        )


@router.websocket("/stream")
async def event_stream(
    websocket: WebSocket,
    types: Optional[str] = None,
    zone: Optional[str] = None,
) -> None:
    """
    Real-time event stream over WebSocket.

    Query params:
    - **types**: comma-separated event types (e.g. ``zone_enter,zone_exit``)
    - **zone**: zone ID filter

    On connect the client receives a ``backfill`` message with the last 50
    matching events. Subsequent events arrive as ``event`` messages.
    """
    manager = get_connection_manager()
    client = await manager.connect(websocket, types=types, zone=zone)

    try:
        while True:
            # Keep the connection alive; the client can send pings / control messages
            data = await websocket.receive_text()
            # Echo back as a heartbeat acknowledgement
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(client)
    except Exception:
        manager.disconnect(client)
        logger.exception("events.stream_error")
