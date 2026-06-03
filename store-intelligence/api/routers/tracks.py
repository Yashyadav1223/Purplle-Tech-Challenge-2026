"""
Store Intelligence System — Tracks Router.

Provides a single endpoint to retrieve the full journey of a tracked visitor.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from analytics.store import get_state_store
from api.schemas import (
    ApiResponse,
    ErrorDetail,
    ResponseMeta,
    TrackJourney,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/tracks", tags=["Tracks"])


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@router.get("/{track_id}", response_model=ApiResponse)
async def get_track_journey(track_id: int, request: Request) -> ApiResponse:
    """
    Retrieve the full journey of a single tracked visitor.

    Returns the track's first/last seen timestamps, total duration,
    zones visited with dwell times, and bounding-box history.
    """
    try:
        store = get_state_store()
        track = store.get_track(track_id)

        if track is None:
            return ApiResponse(
                status="error",
                meta=ResponseMeta(request_id=_get_request_id(request)),
                error=ErrorDetail(
                    code="TRACK_NOT_FOUND",
                    message=f"Track ID {track_id} not found.",
                ),
            )

        # Compute zones visited and per-zone dwell times from zone_history
        zones_visited: list[str] = []
        zone_dwell_times: dict[str, float] = {}

        for entry in track.zone_history:
            zid = entry.get("zone_id", "unknown")
            if zid not in zones_visited:
                zones_visited.append(zid)
            dwell = entry.get("dwell_seconds", 0.0)
            zone_dwell_times[zid] = zone_dwell_times.get(zid, 0.0) + dwell

        total_duration = (track.last_seen - track.first_seen).total_seconds()

        journey = TrackJourney(
            track_id=track.track_id,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            total_duration_seconds=round(total_duration, 2),
            zones_visited=zones_visited,
            zone_dwell_times=zone_dwell_times,
            bbox_history=track.bbox_history[-100:],  # Cap to last 100 entries
        )

        return ApiResponse(
            status="success",
            data=journey.model_dump(mode="json"),
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("tracks.get_failed", track_id=track_id)
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="TRACK_LOOKUP_FAILED", message=str(exc)),
        )
