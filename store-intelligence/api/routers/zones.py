"""
Store Intelligence System — Zones Router.

Provides a live-data endpoint for individual zones showing
current occupancy, active track IDs, and average dwell time.
"""

from __future__ import annotations

import json
import os

import structlog
from fastapi import APIRouter, Request

from analytics.store import get_state_store
from api.schemas import (
    ApiResponse,
    ErrorDetail,
    ResponseMeta,
    ZoneLiveData,
)
from config.settings import get_settings

logger = structlog.get_logger()
router = APIRouter(prefix="/zones", tags=["Zones"])


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _get_zone_name(zone_id: str) -> str:
    """Look up the human-readable zone name from config/zones.json."""
    settings = get_settings()
    zones_path = settings.ZONES_CONFIG_PATH
    if not os.path.isabs(zones_path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        zones_path = os.path.join(base, zones_path)
    try:
        with open(zones_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for z in data.get("zones", []):
            if z["id"] == zone_id:
                return z.get("name", zone_id)
    except Exception:
        pass
    return zone_id


@router.get("/{zone_id}/live", response_model=ApiResponse)
async def get_zone_live(zone_id: str, request: Request) -> ApiResponse:
    """
    Return real-time occupancy data for a single zone.

    Includes:
    - Current occupancy count
    - List of active track IDs in the zone
    - Average dwell time of people currently in the zone
    """
    try:
        store = get_state_store()
        zone_occ = store.get_zone_occupancy(zone_id)

        if zone_occ is None:
            return ApiResponse(
                status="error",
                meta=ResponseMeta(request_id=_get_request_id(request)),
                error=ErrorDetail(
                    code="ZONE_NOT_FOUND",
                    message=f"Zone '{zone_id}' not found or has no data.",
                ),
            )

        # Compute average dwell for currently active tracks in this zone
        from datetime import datetime as dt

        now = dt.utcnow()
        dwell_sum = 0.0
        active_ids = sorted(zone_occ.active_track_ids)

        for tid in zone_occ.active_track_ids:
            track = store.get_track(tid)
            if track and track.zone_enter_time:
                dwell_sum += (now - track.zone_enter_time).total_seconds()

        avg_dwell = (
            round(dwell_sum / len(active_ids), 1) if active_ids else 0.0
        )

        zone_name = _get_zone_name(zone_id)

        live = ZoneLiveData(
            zone_id=zone_id,
            zone_name=zone_name,
            current_occupancy=zone_occ.current_count,
            active_track_ids=active_ids,
            avg_dwell_current=avg_dwell,
        )

        return ApiResponse(
            status="success",
            data=live.model_dump(),
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("zones.live_failed", zone_id=zone_id)
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="ZONE_LIVE_FAILED", message=str(exc)),
        )
