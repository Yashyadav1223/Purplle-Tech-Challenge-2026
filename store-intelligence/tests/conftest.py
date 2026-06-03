"""
Store Intelligence System — Pytest Configuration & Shared Fixtures.

Provides reusable test fixtures for frames, events, state store,
settings overrides, and FastAPI TestClient.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Project imports ──────────────────────────────────────────
from analytics.store import StateStore, ZoneOccupancy
from config.settings import Settings


# ═══════════════════════════════════════════════════════════
# Settings Override
# ═══════════════════════════════════════════════════════════


@pytest.fixture()
def test_settings() -> Settings:
    """Return a Settings instance with deterministic test values."""
    return Settings(
        APP_NAME="Store Intelligence Test",
        APP_VERSION="0.0.1-test",
        DEBUG=True,
        LOG_LEVEL="DEBUG",
        VIDEO_FPS=5,
        VIDEO_FRAME_WIDTH=640,
        VIDEO_FRAME_HEIGHT=640,
        ENABLE_CLAHE=False,
        CONFIDENCE_THRESHOLD=0.4,
        USE_CUDA=False,
        ZONES_CONFIG_PATH="config/zones.json",
        DWELL_ALERT_THRESHOLD_SECONDS=120,
        CROWD_THRESHOLD=10,
        LOITERING_THRESHOLD_SECONDS=300,
        CHECKOUT_QUEUE_THRESHOLD=8,
        ZSCORE_THRESHOLD=2.5,
        BUSINESS_HOURS_START=9,
        BUSINESS_HOURS_END=21,
        STREAM_BACKEND="memory",
        DATABASE_URL="sqlite+aiosqlite:///./test_store.db",
    )


# ═══════════════════════════════════════════════════════════
# State Store
# ═══════════════════════════════════════════════════════════


@pytest.fixture()
def mock_state_store() -> StateStore:
    """Return a fresh StateStore pre-initialised with standard zones."""
    store = StateStore(max_event_history=500, max_footfall_history=100)
    for zone_id in ("entrance", "lipstick_aisle", "skincare_aisle", "trial_area", "checkout"):
        store.init_zone(zone_id)
    return store


@pytest.fixture()
def populated_state_store(mock_state_store: StateStore) -> StateStore:
    """StateStore pre-populated with sample tracks, events, and anomalies."""
    store = mock_state_store

    # Register a camera
    store.register_camera("cam01", "test_data/sample_store.mp4")

    # Simulate 5 visitors entering and moving through zones
    for tid in range(5):
        store.upsert_track(tid, "cam01", [0.1, 0.2, 0.3, 0.4], zone_id="entrance")
        store.zone_enter("entrance", tid, "entry_exit")

    # Move some to product zones
    for tid in [0, 1, 2]:
        store.zone_exit("entrance", tid, dwell_seconds=15.0)
        store.upsert_track(tid, "cam01", [0.35, 0.1, 0.5, 0.4], zone_id="lipstick_aisle")
        store.zone_enter("lipstick_aisle", tid, "product")

    # Move 2 to checkout
    for tid in [0, 1]:
        store.zone_exit("lipstick_aisle", tid, dwell_seconds=45.0)
        store.upsert_track(tid, "cam01", [0.7, 0.6, 0.9, 0.9], zone_id="checkout")
        store.zone_enter("checkout", tid, "checkout")

    # Add sample events
    for i in range(10):
        store.add_event({
            "event_id": str(uuid.uuid4()),
            "event_type": "zone_enter",
            "camera_id": "cam01",
            "timestamp": datetime.utcnow().isoformat(),
            "track_id": i % 5,
            "zone_id": "entrance",
        })

    # Add an anomaly
    store.add_anomaly({
        "anomaly_id": "anomaly-001",
        "anomaly_type": "crowd_surge",
        "severity": "HIGH",
        "affected_zone": "entrance",
        "camera_id": "cam01",
        "timestamp": datetime.utcnow().isoformat(),
        "evidence": {"count": 15, "threshold": 10},
        "suggested_action": "Send staff to entrance",
        "resolved": False,
    })

    return store


# ═══════════════════════════════════════════════════════════
# Sample Frames (numpy arrays)
# ═══════════════════════════════════════════════════════════


@pytest.fixture()
def sample_frame_640() -> np.ndarray:
    """640×640 BGR frame filled with mid-grey."""
    return np.full((640, 640, 3), 128, dtype=np.uint8)


@pytest.fixture()
def sample_frame_480p() -> np.ndarray:
    """640×480 BGR frame filled with mid-grey."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture()
def sample_frame_1080p() -> np.ndarray:
    """1920×1080 BGR frame for resize tests."""
    return np.full((1080, 1920, 3), 100, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════
# Sample Events
# ═══════════════════════════════════════════════════════════


@pytest.fixture()
def sample_event() -> Dict[str, Any]:
    """A sample zone_enter event dict."""
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "zone_enter",
        "camera_id": "cam01",
        "timestamp": datetime.utcnow().isoformat(),
        "track_id": 42,
        "bbox": [0.3, 0.4, 0.5, 0.6],
        "zone_id": "lipstick_aisle",
        "metadata": {"confidence": 0.87},
    }


@pytest.fixture()
def sample_anomaly_event() -> Dict[str, Any]:
    """A sample anomaly event dict."""
    return {
        "anomaly_id": str(uuid.uuid4()),
        "anomaly_type": "loitering",
        "severity": "MEDIUM",
        "affected_zone": "entrance",
        "camera_id": "cam01",
        "timestamp": datetime.utcnow().isoformat(),
        "evidence": {"track_id": 7, "dwell_seconds": 350},
        "suggested_action": "Review camera feed for entrance",
        "resolved": False,
    }


# ═══════════════════════════════════════════════════════════
# Zone Configuration
# ═══════════════════════════════════════════════════════════


@pytest.fixture()
def zones_config() -> List[Dict[str, Any]]:
    """Zone definitions matching config/zones.json."""
    return [
        {
            "id": "entrance",
            "name": "Store Entrance",
            "polygon": [[0, 400], [200, 400], [200, 640], [0, 640]],
            "type": "entry_exit",
            "color": "#3B82F6",
        },
        {
            "id": "lipstick_aisle",
            "name": "Lipstick Aisle",
            "polygon": [[220, 50], [420, 50], [420, 300], [220, 300]],
            "type": "product",
            "color": "#EC4899",
        },
        {
            "id": "skincare_aisle",
            "name": "Skincare Aisle",
            "polygon": [[440, 50], [640, 50], [640, 300], [440, 300]],
            "type": "product",
            "color": "#8B5CF6",
        },
        {
            "id": "trial_area",
            "name": "Trial / Sampling Area",
            "polygon": [[220, 320], [420, 320], [420, 520], [220, 520]],
            "type": "engagement",
            "color": "#F59E0B",
        },
        {
            "id": "checkout",
            "name": "Checkout Counter",
            "polygon": [[440, 400], [640, 400], [640, 640], [440, 640]],
            "type": "checkout",
            "color": "#10B981",
        },
    ]


# ═══════════════════════════════════════════════════════════
# FastAPI TestClient
# ═══════════════════════════════════════════════════════════


@pytest.fixture()
def app_client(populated_state_store: StateStore, test_settings: Settings):
    """
    Yield a FastAPI TestClient with the state store and settings
    singletons patched to use test instances.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers.health import router as health_router

    # Build a minimal test app with only the health router
    # (tests that need more routers can extend this)
    app = FastAPI(title="Store Intelligence Test")
    app.include_router(health_router, prefix="/api/v1")

    with patch("api.routers.health.get_state_store", return_value=populated_state_store), \
         patch("api.routers.health.get_settings", return_value=test_settings):
        client = TestClient(app)
        yield client, populated_state_store, test_settings
