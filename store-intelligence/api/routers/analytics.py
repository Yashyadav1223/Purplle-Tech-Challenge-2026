"""
Store Intelligence System — Analytics Router.

Provides aggregate analytics endpoints: footfall, zone stats, heatmap,
conversion funnel, and peak-hours distribution.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional

import structlog
from fastapi import APIRouter, Query, Request

from analytics.store import get_state_store
from api.schemas import (
    ApiResponse,
    ErrorDetail,
    FootfallData,
    FunnelStage,
    HeatmapZone,
    PeakHourData,
    ResponseMeta,
    ZoneStats,
)
from config.settings import get_settings

logger = structlog.get_logger()
router = APIRouter(prefix="/analytics", tags=["Analytics"])


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _load_zone_configs() -> list[dict]:
    """Load zone polygon and color data from config/zones.json."""
    settings = get_settings()
    zones_path = settings.ZONES_CONFIG_PATH
    # Resolve relative paths against the project root
    if not os.path.isabs(zones_path):
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        zones_path = os.path.join(base, zones_path)
    try:
        with open(zones_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("zones", [])
    except Exception:
        logger.exception("analytics.zones_config_load_failed", path=zones_path)
        return []


# ═══════════════════════════════════════════════════════════
# GET /analytics/footfall
# ═══════════════════════════════════════════════════════════

@router.get("/footfall", response_model=ApiResponse)
async def get_footfall(
    request: Request,
    start: Optional[datetime] = Query(default=None, description="Start of range (ISO 8601)"),
    end: Optional[datetime] = Query(default=None, description="End of range (ISO 8601)"),
    granularity: str = Query(default="1h", description="Aggregation granularity: '1h' or '1d'"),
) -> ApiResponse:
    """
    Return footfall data aggregated by hour or day.

    Uses the hourly footfall counters maintained by the state store.
    """
    try:
        store = get_state_store()
        hourly = store.get_hourly_footfall()  # Dict[int, int]

        now = datetime.utcnow()
        records: list[dict] = []

        if granularity == "1d":
            # Collapse all hours into a single daily total
            total = sum(hourly.values())
            records.append(
                FootfallData(
                    period="today",
                    count=total,
                    start_time=now.replace(hour=0, minute=0, second=0, microsecond=0),
                    end_time=now,
                ).model_dump(mode="json")
            )
        else:
            # Hourly granularity (default)
            for hour in range(24):
                count = hourly.get(hour, 0)
                start_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                end_time = now.replace(hour=hour, minute=59, second=59, microsecond=0)
                records.append(
                    FootfallData(
                        period=f"{hour:02d}:00",
                        count=count,
                        start_time=start_time,
                        end_time=end_time,
                    ).model_dump(mode="json")
                )

        return ApiResponse(
            status="success",
            data=records,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("analytics.footfall_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="FOOTFALL_FAILED", message=str(exc)),
        )


# ═══════════════════════════════════════════════════════════
# GET /analytics/zones
# ═══════════════════════════════════════════════════════════

@router.get("/zones", response_model=ApiResponse)
async def get_zone_stats(request: Request) -> ApiResponse:
    """Return per-zone visit counts, average dwell, peak count, and current occupancy."""
    try:
        store = get_state_store()
        zone_data = store.get_all_zone_stats()  # Dict[str, ZoneOccupancy]
        zone_configs = {z["id"]: z for z in _load_zone_configs()}

        result: list[dict] = []
        for zone_id, occ in zone_data.items():
            cfg = zone_configs.get(zone_id, {})
            avg_dwell = (
                occ.total_dwell_seconds / len(occ.dwell_times)
                if occ.dwell_times
                else 0.0
            )
            result.append(
                ZoneStats(
                    zone_id=zone_id,
                    zone_name=cfg.get("name", zone_id),
                    zone_type=cfg.get("type", "unknown"),
                    visit_count=occ.total_visits,
                    avg_dwell_seconds=round(avg_dwell, 1),
                    peak_count=occ.peak_count,
                    current_occupancy=occ.current_count,
                ).model_dump()
            )

        return ApiResponse(
            status="success",
            data=result,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("analytics.zones_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="ZONE_STATS_FAILED", message=str(exc)),
        )


# ═══════════════════════════════════════════════════════════
# GET /analytics/heatmap
# ═══════════════════════════════════════════════════════════

@router.get("/heatmap", response_model=ApiResponse)
async def get_heatmap(request: Request) -> ApiResponse:
    """
    Return zone heatmap data with normalized 0-1 intensity.

    Intensity is computed as ``zone_visits / max_visits`` across all zones.
    Polygon and color data are sourced from ``config/zones.json``.
    """
    try:
        store = get_state_store()
        zone_data = store.get_all_zone_stats()
        zone_configs = {z["id"]: z for z in _load_zone_configs()}

        # Compute max visits for normalisation
        max_visits = max(
            (occ.total_visits for occ in zone_data.values()), default=1
        ) or 1  # avoid division by zero

        result: list[dict] = []
        for zone_id, occ in zone_data.items():
            cfg = zone_configs.get(zone_id, {})
            avg_dwell = (
                occ.total_dwell_seconds / len(occ.dwell_times)
                if occ.dwell_times
                else 0.0
            )
            result.append(
                HeatmapZone(
                    zone_id=zone_id,
                    zone_name=cfg.get("name", zone_id),
                    intensity=round(occ.total_visits / max_visits, 3),
                    visit_count=occ.total_visits,
                    avg_dwell_seconds=round(avg_dwell, 1),
                    polygon=cfg.get("polygon", []),
                    color=cfg.get("color", "#6366F1"),
                ).model_dump()
            )

        return ApiResponse(
            status="success",
            data=result,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("analytics.heatmap_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="HEATMAP_FAILED", message=str(exc)),
        )


# ═══════════════════════════════════════════════════════════
# GET /analytics/funnel
# ═══════════════════════════════════════════════════════════

@router.get("/funnel", response_model=ApiResponse)
async def get_funnel(request: Request) -> ApiResponse:
    """
    Return the conversion funnel: entrance → product → engagement → checkout.

    Percentages are relative to the entrance (top-of-funnel) count.
    """
    try:
        store = get_state_store()
        funnel = store.get_funnel_data()  # Dict[str, int]

        stages_order = ["entrance", "product", "engagement", "checkout"]
        top = funnel.get("entrance", 0) or 1  # avoid div-by-zero

        result: list[dict] = []
        for stage_name in stages_order:
            count = funnel.get(stage_name, 0)
            result.append(
                FunnelStage(
                    stage=stage_name,
                    count=count,
                    percentage=round(count / top * 100, 1),
                ).model_dump()
            )

        return ApiResponse(
            status="success",
            data=result,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("analytics.funnel_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="FUNNEL_FAILED", message=str(exc)),
        )


# ═══════════════════════════════════════════════════════════
# GET /analytics/peak-hours
# ═══════════════════════════════════════════════════════════

@router.get("/peak-hours", response_model=ApiResponse)
async def get_peak_hours(request: Request) -> ApiResponse:
    """Return hourly footfall distribution (0-23)."""
    try:
        store = get_state_store()
        hourly = store.get_hourly_footfall()

        result: list[dict] = [
            PeakHourData(hour=h, count=hourly.get(h, 0)).model_dump()
            for h in range(24)
        ]

        return ApiResponse(
            status="success",
            data=result,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("analytics.peak_hours_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="PEAK_HOURS_FAILED", message=str(exc)),
        )
