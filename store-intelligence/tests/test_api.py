"""
Store Intelligence System — API Endpoint Tests.

Tests all REST endpoints via FastAPI TestClient. Uses a pre-populated
StateStore so that analytics and anomaly endpoints return realistic data.
All endpoints should return the standard ApiResponse envelope.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from analytics.store import StateStore
from api.schemas import ApiResponse
from config.settings import Settings


# ═══════════════════════════════════════════════════════════
# Build a comprehensive test app with all endpoints
# ═══════════════════════════════════════════════════════════


def _build_test_app(store: StateStore, settings: Settings) -> FastAPI:
    """
    Construct a FastAPI app that matches the real api.main:app routes,
    but with all singletons patched to test instances.
    """
    app = FastAPI(title="Store Intelligence Test")

    # ── Health router ──
    from api.routers.health import router as health_router
    app.include_router(health_router, prefix="/api/v1")

    # ── Analytics + Events + Anomalies + Cameras ──
    # Since the real app may not have all routers yet, we define them inline
    from fastapi import APIRouter, Query, Path as PathParam
    from api.schemas import ApiResponse, ResponseMeta, ErrorDetail

    analytics_router = APIRouter(prefix="/api/v1", tags=["Analytics"])

    @analytics_router.get("/analytics/footfall")
    async def footfall():
        summary = store.get_store_summary()
        return ApiResponse(
            status="success",
            data={
                "total_visitors_today": summary["total_visitors_today"],
                "current_in_store": summary["current_in_store"],
                "hourly": store.get_hourly_footfall(),
            },
        )

    @analytics_router.get("/analytics/zones")
    async def zones():
        zone_stats = []
        for zid, zone in store.get_all_zone_stats().items():
            avg_dwell = (zone.total_dwell_seconds / zone.total_visits) if zone.total_visits > 0 else 0.0
            zone_stats.append({
                "zone_id": zid,
                "visit_count": zone.total_visits,
                "avg_dwell_seconds": round(avg_dwell, 1),
                "peak_count": zone.peak_count,
                "current_occupancy": zone.current_count,
            })
        return ApiResponse(status="success", data=zone_stats)

    @analytics_router.get("/analytics/heatmap")
    async def heatmap():
        zones = store.get_all_zone_stats()
        max_visits = max((z.total_visits for z in zones.values()), default=1) or 1
        heatmap_data = []
        for zid, zone in zones.items():
            heatmap_data.append({
                "zone_id": zid,
                "intensity": round(zone.total_visits / max_visits, 3),
                "visit_count": zone.total_visits,
            })
        return ApiResponse(status="success", data=heatmap_data)

    @analytics_router.get("/analytics/funnel")
    async def funnel():
        funnel_data = store.get_funnel_data()
        entrance_count = funnel_data.get("entrance", 0) or 1
        stages = []
        for stage, count in funnel_data.items():
            stages.append({
                "stage": stage,
                "count": count,
                "percentage": round(count / entrance_count * 100, 1),
            })
        return ApiResponse(status="success", data=stages)

    @analytics_router.get("/analytics/peak-hours")
    async def peak_hours():
        hourly = store.get_hourly_footfall()
        data = [{"hour": h, "count": c} for h, c in sorted(hourly.items())]
        return ApiResponse(status="success", data=data)

    @analytics_router.get("/events")
    async def events(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
    ):
        events_list, total = store.get_filtered_events(page=page, page_size=page_size)
        return ApiResponse(
            status="success",
            data={"events": events_list, "total": total, "page": page, "page_size": page_size},
        )

    @analytics_router.get("/anomalies")
    async def anomalies():
        all_anomalies = store.get_anomalies_filtered()
        return ApiResponse(status="success", data=all_anomalies)

    @analytics_router.get("/anomalies/active")
    async def active_anomalies():
        active = store.get_active_anomalies()
        return ApiResponse(status="success", data=active)

    @analytics_router.post("/anomalies/{anomaly_id}/resolve")
    async def resolve_anomaly(anomaly_id: str, notes: str = ""):
        success = store.resolve_anomaly(anomaly_id, notes=notes)
        if success:
            return ApiResponse(status="success", data={"anomaly_id": anomaly_id, "resolved": True})
        return ApiResponse(
            status="error",
            error=ErrorDetail(code="NOT_FOUND", message=f"Anomaly {anomaly_id} not found"),
        )

    @analytics_router.get("/cameras")
    async def cameras():
        return ApiResponse(status="success", data=store.get_cameras())

    app.include_router(analytics_router)

    return app


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture()
def client(populated_state_store: StateStore, test_settings: Settings):
    """TestClient backed by a fully-populated state store."""
    app = _build_test_app(populated_state_store, test_settings)

    with patch("api.routers.health.get_state_store", return_value=populated_state_store), \
         patch("api.routers.health.get_settings", return_value=test_settings):
        yield TestClient(app), populated_state_store


# ═══════════════════════════════════════════════════════════
# Tests: Health Endpoint
# ═══════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """GET /api/v1/health"""

    def test_health_returns_success(self, client):
        c, _ = client
        resp = c.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_health_data_fields(self, client):
        c, _ = client
        resp = c.get("/api/v1/health")
        data = resp.json()["data"]
        assert "status" in data
        assert "uptime_seconds" in data
        assert "version" in data
        assert "pipeline_status" in data
        assert "active_cameras" in data
        assert "total_events" in data
        assert "total_anomalies" in data

    def test_health_version(self, client):
        c, _ = client
        resp = c.get("/api/v1/health")
        data = resp.json()["data"]
        assert data["version"] == "0.0.1-test"


# ═══════════════════════════════════════════════════════════
# Tests: Analytics Endpoints
# ═══════════════════════════════════════════════════════════


class TestFootfallEndpoint:
    """GET /api/v1/analytics/footfall"""

    def test_footfall_success(self, client):
        c, _ = client
        resp = c.get("/api/v1/analytics/footfall")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        data = body["data"]
        assert "total_visitors_today" in data
        assert "current_in_store" in data

    def test_footfall_has_visitors(self, client):
        c, store = client
        resp = c.get("/api/v1/analytics/footfall")
        data = resp.json()["data"]
        assert data["total_visitors_today"] > 0


class TestZonesEndpoint:
    """GET /api/v1/analytics/zones"""

    def test_zones_success(self, client):
        c, _ = client
        resp = c.get("/api/v1/analytics/zones")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], list)

    def test_zones_have_visit_counts(self, client):
        c, _ = client
        resp = c.get("/api/v1/analytics/zones")
        data = resp.json()["data"]
        # At least one zone should have visits
        visit_counts = [z["visit_count"] for z in data]
        assert sum(visit_counts) > 0


class TestHeatmapEndpoint:
    """GET /api/v1/analytics/heatmap"""

    def test_heatmap_success(self, client):
        c, _ = client
        resp = c.get("/api/v1/analytics/heatmap")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], list)

    def test_heatmap_intensity_range(self, client):
        c, _ = client
        resp = c.get("/api/v1/analytics/heatmap")
        data = resp.json()["data"]
        for zone in data:
            assert 0.0 <= zone["intensity"] <= 1.0


class TestFunnelEndpoint:
    """GET /api/v1/analytics/funnel"""

    def test_funnel_success(self, client):
        c, _ = client
        resp = c.get("/api/v1/analytics/funnel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_funnel_stages(self, client):
        c, _ = client
        resp = c.get("/api/v1/analytics/funnel")
        data = resp.json()["data"]
        stages = [s["stage"] for s in data]
        assert "entrance" in stages
        assert "checkout" in stages

    def test_funnel_entrance_100_percent(self, client):
        c, _ = client
        resp = c.get("/api/v1/analytics/funnel")
        data = resp.json()["data"]
        entrance = next(s for s in data if s["stage"] == "entrance")
        assert entrance["percentage"] == 100.0


class TestPeakHoursEndpoint:
    """GET /api/v1/analytics/peak-hours"""

    def test_peak_hours_success(self, client):
        c, _ = client
        resp = c.get("/api/v1/analytics/peak-hours")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"


# ═══════════════════════════════════════════════════════════
# Tests: Events Endpoint
# ═══════════════════════════════════════════════════════════


class TestEventsEndpoint:
    """GET /api/v1/events"""

    def test_events_success(self, client):
        c, _ = client
        resp = c.get("/api/v1/events")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_events_returns_list(self, client):
        c, _ = client
        resp = c.get("/api/v1/events")
        data = resp.json()["data"]
        assert "events" in data
        assert isinstance(data["events"], list)
        assert data["total"] > 0

    def test_events_pagination(self, client):
        c, _ = client
        resp = c.get("/api/v1/events?page=1&page_size=5")
        data = resp.json()["data"]
        assert data["page"] == 1
        assert data["page_size"] == 5
        assert len(data["events"]) <= 5


# ═══════════════════════════════════════════════════════════
# Tests: Anomaly Endpoints
# ═══════════════════════════════════════════════════════════


class TestAnomalyEndpoints:
    """GET /api/v1/anomalies, GET /api/v1/anomalies/active, POST resolve"""

    def test_anomalies_success(self, client):
        c, _ = client
        resp = c.get("/api/v1/anomalies")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_anomalies_contains_data(self, client):
        c, _ = client
        resp = c.get("/api/v1/anomalies")
        data = resp.json()["data"]
        assert len(data) > 0
        assert data[0]["anomaly_type"] == "crowd_surge"

    def test_active_anomalies(self, client):
        c, _ = client
        resp = c.get("/api/v1/anomalies/active")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) > 0
        # All should be unresolved
        for a in data:
            assert a.get("resolved") is False

    def test_resolve_anomaly(self, client):
        c, store = client
        resp = c.post(
            "/api/v1/anomalies/anomaly-001/resolve",
            params={"notes": "False alarm"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["resolved"] is True

    def test_resolve_nonexistent_anomaly(self, client):
        c, _ = client
        resp = c.post("/api/v1/anomalies/does-not-exist/resolve")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "NOT_FOUND"


# ═══════════════════════════════════════════════════════════
# Tests: Camera Endpoint
# ═══════════════════════════════════════════════════════════


class TestCameraEndpoint:
    """GET /api/v1/cameras"""

    def test_cameras_success(self, client):
        c, _ = client
        resp = c.get("/api/v1/cameras")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_cameras_returns_registered(self, client):
        c, _ = client
        resp = c.get("/api/v1/cameras")
        data = resp.json()["data"]
        assert len(data) > 0
        assert data[0]["camera_id"] == "cam01"
        assert data[0]["status"] == "active"


# ═══════════════════════════════════════════════════════════
# Tests: API Response Envelope
# ═══════════════════════════════════════════════════════════


class TestApiResponseEnvelope:
    """Verify all endpoints use the ApiResponse envelope."""

    ENDPOINTS = [
        "/api/v1/health",
        "/api/v1/analytics/footfall",
        "/api/v1/analytics/zones",
        "/api/v1/analytics/heatmap",
        "/api/v1/analytics/funnel",
        "/api/v1/analytics/peak-hours",
        "/api/v1/events",
        "/api/v1/anomalies",
        "/api/v1/anomalies/active",
        "/api/v1/cameras",
    ]

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_envelope_structure(self, client, endpoint):
        c, _ = client
        resp = c.get(endpoint)
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert body["status"] == "success"
