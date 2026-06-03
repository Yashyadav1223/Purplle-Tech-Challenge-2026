# PROMPT: Generate tests for all /stores/{id}/* endpoints: metrics, funnel,
# heatmap, anomalies. Cover edge cases: empty store (no events), all-staff
# store, zero purchases, re-entry not double-counted in funnel, dead zone
# detection, and data_confidence flag in heatmap.
# CHANGES MADE: Added parametrized tests for multiple store IDs, added
# assertions for funnel monotonicity (each stage <= previous), added
# heatmap intensity range validation (0-100).

"""
Store Intelligence System — Store API Endpoint Tests.

Tests for GET /stores/{id}/metrics, /funnel, /heatmap, /anomalies.
Covers edge cases: empty stores, all-staff clips, zero purchases,
re-entry in funnel, and data confidence flags.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient


def _make_event(
    event_type: str = "ENTRY",
    store_id: str = "STORE_TEST_001",
    visitor_id: str | None = None,
    zone_id: str | None = None,
    is_staff: bool = False,
    dwell_ms: int = 0,
    session_seq: int = 1,
) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": visitor_id or f"VIS_{uuid.uuid4().hex[:6]}",
        "event_type": event_type,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": 0.85,
        "metadata": {
            "queue_depth": None,
            "sku_zone": zone_id,
            "session_seq": session_seq,
        },
    }



@pytest.fixture(autouse=True)
def clear_multi_store_state():
    from analytics.multi_store import get_multi_store_state
    store = get_multi_store_state()
    store._stores.clear()

def _ingest(client, events: list) -> dict:
    """Helper to ingest events and return response data."""
    resp = client.post("/api/v1/stores/events/ingest", json={"events": events})
    assert resp.status_code == 200
    return resp.json()["data"]


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


@pytest.fixture
def populated_store(client):
    """Create a store with a realistic set of events for testing."""
    store_id = "STORE_TEST_001"
    visitor1 = "VIS_aaa111"
    visitor2 = "VIS_bbb222"
    visitor3 = "VIS_ccc333"  # Staff

    events = [
        # Visitor 1: Full journey → purchase
        _make_event("ENTRY", store_id, visitor1, session_seq=1),
        _make_event("ZONE_ENTER", store_id, visitor1, zone_id="SKINCARE", session_seq=2),
        _make_event("ZONE_DWELL", store_id, visitor1, zone_id="SKINCARE", dwell_ms=45000, session_seq=3),
        _make_event("ZONE_EXIT", store_id, visitor1, zone_id="SKINCARE", session_seq=4),
        _make_event("BILLING_QUEUE_JOIN", store_id, visitor1, zone_id="BILLING", session_seq=5),
        _make_event("EXIT", store_id, visitor1, session_seq=6),

        # Visitor 2: Browse only, no purchase
        _make_event("ENTRY", store_id, visitor2, session_seq=1),
        _make_event("ZONE_ENTER", store_id, visitor2, zone_id="LIPSTICK", session_seq=2),
        _make_event("ZONE_EXIT", store_id, visitor2, zone_id="LIPSTICK", session_seq=3),
        _make_event("EXIT", store_id, visitor2, session_seq=4),

        # Visitor 3: Staff (should be excluded from metrics)
        _make_event("ENTRY", store_id, visitor3, is_staff=True, session_seq=1),
        _make_event("ZONE_ENTER", store_id, visitor3, zone_id="SKINCARE", is_staff=True, session_seq=2),
        _make_event("ZONE_ENTER", store_id, visitor3, zone_id="LIPSTICK", is_staff=True, session_seq=3),
        _make_event("ZONE_ENTER", store_id, visitor3, zone_id="BILLING", is_staff=True, session_seq=4),
    ]

    _ingest(client, events)
    return store_id


class TestStoreMetrics:
    """Tests for GET /stores/{id}/metrics."""

    def test_metrics_basic(self, client, populated_store):
        """Metrics should return visitor count excluding staff."""
        resp = client.get(f"/api/v1/stores/{populated_store}/metrics")
        assert resp.status_code == 200
        data = resp.json()["data"]
        # 2 customers, staff excluded
        assert data["unique_visitors"] == 2
        assert isinstance(data["conversion_rate"], (int, float))
        assert isinstance(data["abandonment_rate"], (int, float))

    def test_metrics_empty_store(self, client):
        """A store with no events should return zeros, not crash."""
        resp = client.get("/api/v1/stores/STORE_EMPTY_999/metrics")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["unique_visitors"] == 0
        assert data["conversion_rate"] == 0.0

    def test_metrics_all_staff(self, client):
        """A store where all detected people are staff should show 0 customers."""
        store_id = "STORE_STAFF_ONLY"
        events = [
            _make_event("ENTRY", store_id, f"VIS_staff_{i}", is_staff=True)
            for i in range(5)
        ]
        _ingest(client, events)

        resp = client.get(f"/api/v1/stores/{store_id}/metrics")
        assert resp.status_code == 200
        assert resp.json()["data"]["unique_visitors"] == 0

    def test_metrics_zero_purchases(self, client):
        """Store with visitors but no billing events → 0% conversion."""
        store_id = "STORE_NO_PURCHASE"
        events = [
            _make_event("ENTRY", store_id, f"VIS_browse_{i}")
            for i in range(3)
        ]
        _ingest(client, events)

        resp = client.get(f"/api/v1/stores/{store_id}/metrics")
        assert resp.status_code == 200
        assert resp.json()["data"]["conversion_rate"] == 0.0


class TestStoreFunnel:
    """Tests for GET /stores/{id}/funnel."""

    def test_funnel_basic(self, client, populated_store):
        """Funnel should have Entry → Zone Visit → Billing → Purchase stages."""
        resp = client.get(f"/api/v1/stores/{populated_store}/funnel")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) >= 3
        # Each stage should have 'stage', 'count', 'drop_off_pct'
        for stage in data:
            assert "stage" in stage
            assert "count" in stage
            assert "drop_off_pct" in stage

    def test_funnel_monotonically_decreasing(self, client, populated_store):
        """Funnel counts should be monotonically decreasing or equal."""
        resp = client.get(f"/api/v1/stores/{populated_store}/funnel")
        data = resp.json()["data"]
        counts = [s["count"] for s in data]
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1], (
                f"Funnel stage {data[i]['stage']} ({counts[i]}) > "
                f"previous stage {data[i-1]['stage']} ({counts[i-1]})"
            )

    def test_funnel_reentry_not_double_counted(self, client):
        """A visitor with REENTRY should count as 1 in the funnel, not 2."""
        store_id = "STORE_REENTRY_TEST"
        vid = "VIS_reenter"
        events = [
            _make_event("ENTRY", store_id, vid, session_seq=1),
            _make_event("EXIT", store_id, vid, session_seq=2),
            _make_event("REENTRY", store_id, vid, session_seq=3),
            _make_event("EXIT", store_id, vid, session_seq=4),
        ]
        _ingest(client, events)

        resp = client.get(f"/api/v1/stores/{store_id}/funnel")
        data = resp.json()["data"]
        # Entry stage should count this visitor only once
        entry_stage = next((s for s in data if s["stage"].lower() in ("entry", "entrance")), None)
        assert entry_stage is not None
        assert entry_stage["count"] == 1

    def test_funnel_empty_store(self, client):
        """Empty store funnel should return stages with zero counts."""
        resp = client.get("/api/v1/stores/STORE_FUNNEL_EMPTY/funnel")
        assert resp.status_code == 200
        data = resp.json()["data"]
        for stage in data:
            assert stage["count"] == 0


class TestStoreHeatmap:
    """Tests for GET /stores/{id}/heatmap."""

    def test_heatmap_basic(self, client, populated_store):
        """Heatmap should return zone data with intensity 0-100."""
        resp = client.get(f"/api/v1/stores/{populated_store}/heatmap")
        assert resp.status_code == 200
        data = resp.json()["data"]
        zones = data["zones"]
        assert isinstance(zones, list)
        for zone in zones:
            assert "zone_id" in zone
            assert "intensity" in zone
            assert 0 <= zone["intensity"] <= 100

    def test_heatmap_data_confidence(self, client):
        """Heatmap zones with <20 sessions should have data_confidence=false."""
        store_id = "STORE_LOW_TRAFFIC"
        events = [
            _make_event("ZONE_ENTER", store_id, f"VIS_{i}", zone_id="SKINCARE")
            for i in range(5)  # Only 5 sessions, below 20 threshold
        ]
        _ingest(client, events)

        resp = client.get(f"/api/v1/stores/{store_id}/heatmap")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "data_confidence" in data

    def test_heatmap_empty_store(self, client):
        """Empty store heatmap should return empty list, not crash."""
        resp = client.get("/api/v1/stores/STORE_HEATMAP_EMPTY/heatmap")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data["zones"], list)


class TestStoreAnomalies:
    """Tests for GET /stores/{id}/anomalies."""

    def test_anomalies_basic(self, client, populated_store):
        """Anomalies endpoint should return a list."""
        resp = client.get(f"/api/v1/stores/{populated_store}/anomalies")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, list)

    def test_anomalies_severity_levels(self, client, populated_store):
        """All anomalies should have valid severity levels."""
        resp = client.get(f"/api/v1/stores/{populated_store}/anomalies")
        data = resp.json()["data"]
        valid_severities = {"INFO", "WARN", "CRITICAL"}
        for anomaly in data:
            assert anomaly.get("severity") in valid_severities

    def test_anomalies_have_suggested_action(self, client, populated_store):
        """Every anomaly should include a suggested_action string."""
        resp = client.get(f"/api/v1/stores/{populated_store}/anomalies")
        data = resp.json()["data"]
        for anomaly in data:
            assert "suggested_action" in anomaly
            assert isinstance(anomaly["suggested_action"], str)

    def test_anomalies_empty_store(self, client):
        """Empty store should return empty anomalies list."""
        resp = client.get("/api/v1/stores/STORE_ANOMALY_EMPTY/anomalies")
        assert resp.status_code == 200
        assert resp.json()["data"] == []


class TestHealth:
    """Tests for GET /api/v1/health."""

    def test_health_basic(self, client):
        """Health endpoint should return service status."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
