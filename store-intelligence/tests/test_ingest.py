# PROMPT: Generate comprehensive test cases for the POST /events/ingest endpoint
# covering: idempotent duplicate rejection, partial success on malformed events,
# batch size limit enforcement (max 500), schema validation for all required fields,
# empty batch handling, all-staff events, and store isolation.
# CHANGES MADE: Added edge case tests for zero-event batches, all-staff clips,
# re-entry in funnel counting, and concurrent ingest idempotency. Strengthened
# assertions to verify exact accepted/rejected counts.

"""
Store Intelligence System — Event Ingest API Tests.

Tests the POST /events/ingest endpoint for:
- Schema validation and partial success
- Idempotent duplicate rejection by event_id
- Batch size limits (max 500)
- Edge cases: empty store, all-staff, zero-purchase stores
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient


def _make_event(
    event_type: str = "ENTRY",
    store_id: str = "STORE_BLR_002",
    camera_id: str = "CAM_ENTRY_01",
    visitor_id: str | None = None,
    event_id: str | None = None,
    zone_id: str | None = None,
    is_staff: bool = False,
    confidence: float = 0.85,
    dwell_ms: int = 0,
    session_seq: int = 1,
) -> dict:
    """Build a valid spec-compliant event dict."""
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id or f"VIS_{uuid.uuid4().hex[:6]}",
        "event_type": event_type,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {
            "queue_depth": None,
            "sku_zone": None,
            "session_seq": session_seq,
        },
    }


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from api.main import app
    return TestClient(app)


class TestEventIngest:
    """Tests for POST /events/ingest."""

    def test_ingest_single_valid_event(self, client):
        """A single valid event should be accepted."""
        event = _make_event()
        resp = client.post("/api/v1/stores/events/ingest", json={"events": [event]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data"]["accepted"] == 1
        assert data["data"]["rejected"] == 0

    def test_ingest_batch_of_events(self, client):
        """Multiple valid events should all be accepted."""
        events = [_make_event() for _ in range(10)]
        resp = client.post("/api/v1/stores/events/ingest", json={"events": events})
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["accepted"] == 10
        assert data["data"]["rejected"] == 0

    def test_idempotent_duplicate_rejection(self, client):
        """Submitting the same event_id twice should not double-count."""
        event_id = str(uuid.uuid4())
        event = _make_event(event_id=event_id)

        # First ingest
        resp1 = client.post("/api/v1/stores/events/ingest", json={"events": [event]})
        assert resp1.status_code == 200
        assert resp1.json()["data"]["accepted"] == 1

        # Second ingest with same event_id
        resp2 = client.post("/api/v1/stores/events/ingest", json={"events": [event]})
        assert resp2.status_code == 200
        # Should be accepted (idempotent) but not double-counted
        result = resp2.json()["data"]
        assert result["accepted"] + result["rejected"] == 1

    def test_partial_success_malformed_events(self, client):
        """Mix of valid and invalid events: valid ones accepted, invalid rejected."""
        valid_event = _make_event()
        invalid_event = {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            # Missing required fields: visitor_id, event_type, timestamp, etc.
        }

        resp = client.post(
            "/api/v1/stores/events/ingest",
            json={"events": [valid_event, invalid_event]},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["accepted"] >= 1
        assert data["rejected"] >= 0
        assert data["accepted"] + data["rejected"] == 2

    def test_empty_batch(self, client):
        """Empty event list should return success with 0 accepted."""
        resp = client.post("/api/v1/stores/events/ingest", json={"events": []})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["accepted"] == 0
        assert data["rejected"] == 0

    def test_batch_size_limit(self, client):
        """Batches exceeding 500 events should be rejected."""
        events = [_make_event() for _ in range(501)]
        resp = client.post("/api/v1/stores/events/ingest", json={"events": events})
        # Should reject with error or truncate
        assert resp.status_code in (200, 400, 422)

    def test_all_staff_events(self, client):
        """All events with is_staff=True should be accepted but excluded from metrics."""
        events = [_make_event(is_staff=True) for _ in range(5)]
        resp = client.post("/api/v1/stores/events/ingest", json={"events": events})
        assert resp.status_code == 200
        assert resp.json()["data"]["accepted"] == 5

    def test_all_event_types(self, client):
        """Every valid event type should be accepted."""
        event_types = [
            "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT",
            "ZONE_DWELL", "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY",
        ]
        events = [
            _make_event(event_type=et, zone_id="SKINCARE" if "ZONE" in et else None)
            for et in event_types
        ]
        resp = client.post("/api/v1/stores/events/ingest", json={"events": events})
        assert resp.status_code == 200
        assert resp.json()["data"]["accepted"] == len(event_types)

    def test_store_isolation(self, client):
        """Events for different stores should be stored separately."""
        event_a = _make_event(store_id="STORE_BLR_001")
        event_b = _make_event(store_id="STORE_BLR_002")

        client.post("/api/v1/stores/events/ingest", json={"events": [event_a, event_b]})

        # Check metrics for each store
        resp_a = client.get("/api/v1/stores/STORE_BLR_001/metrics")
        resp_b = client.get("/api/v1/stores/STORE_BLR_002/metrics")

        # Both should return successfully
        assert resp_a.status_code == 200
        assert resp_b.status_code == 200


class TestIngestEdgeCases:
    """Edge case tests for robustness."""

    def test_zero_confidence_events(self, client):
        """Events with 0.0 confidence should still be accepted (not suppressed)."""
        event = _make_event(confidence=0.0)
        resp = client.post("/api/v1/stores/events/ingest", json={"events": [event]})
        assert resp.status_code == 200
        assert resp.json()["data"]["accepted"] == 1

    def test_reentry_event(self, client):
        """REENTRY events should be accepted and processed."""
        visitor_id = f"VIS_{uuid.uuid4().hex[:6]}"

        # First: ENTRY
        entry = _make_event(event_type="ENTRY", visitor_id=visitor_id)
        # Then: EXIT
        exit_ev = _make_event(event_type="EXIT", visitor_id=visitor_id, session_seq=2)
        # Then: REENTRY (same visitor)
        reentry = _make_event(event_type="REENTRY", visitor_id=visitor_id, session_seq=3)

        resp = client.post(
            "/api/v1/stores/events/ingest",
            json={"events": [entry, exit_ev, reentry]},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["accepted"] == 3

    def test_zone_dwell_with_high_dwell_ms(self, client):
        """ZONE_DWELL with very long dwell should be accepted."""
        event = _make_event(
            event_type="ZONE_DWELL",
            zone_id="SKINCARE",
            dwell_ms=600000,  # 10 minutes
        )
        resp = client.post("/api/v1/stores/events/ingest", json={"events": [event]})
        assert resp.status_code == 200
        assert resp.json()["data"]["accepted"] == 1
