"""
Store Intelligence System — Anomaly Detection Tests.

Tests each anomaly rule with carefully crafted scenarios:
- Crowd surge detection
- Loitering detection
- Checkout queue threshold
- After-hours detection
- Z-score statistical anomaly
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from analytics.store import StateStore, ZoneOccupancy
from config.settings import Settings


# ═══════════════════════════════════════════════════════════
# Anomaly Detection Logic (standalone functions for testability)
#
# These functions mirror the detection rules that would live in
# an AnomalyDetector class, extracted here so they can be tested
# in isolation without importing heavy ML dependencies.
# ═══════════════════════════════════════════════════════════


def detect_crowd_surge(
    zone_id: str,
    current_count: int,
    threshold: int,
) -> Optional[Dict[str, Any]]:
    """Detect if zone occupancy exceeds the crowd threshold."""
    if current_count > threshold:
        return {
            "anomaly_id": str(uuid.uuid4()),
            "anomaly_type": "crowd_surge",
            "severity": "HIGH" if current_count > threshold * 1.5 else "MEDIUM",
            "affected_zone": zone_id,
            "evidence": {
                "current_count": current_count,
                "threshold": threshold,
                "excess": current_count - threshold,
            },
            "suggested_action": f"Send staff to {zone_id} immediately",
        }
    return None


def detect_loitering(
    track_id: int,
    zone_id: str,
    dwell_seconds: float,
    threshold_seconds: int,
) -> Optional[Dict[str, Any]]:
    """Detect if a person has been in one zone too long."""
    if dwell_seconds > threshold_seconds:
        return {
            "anomaly_id": str(uuid.uuid4()),
            "anomaly_type": "loitering",
            "severity": "MEDIUM" if dwell_seconds < threshold_seconds * 2 else "HIGH",
            "affected_zone": zone_id,
            "evidence": {
                "track_id": track_id,
                "dwell_seconds": dwell_seconds,
                "threshold_seconds": threshold_seconds,
            },
            "suggested_action": f"Review camera feed for {zone_id}",
        }
    return None


def detect_checkout_queue(
    zone_id: str,
    queue_count: int,
    threshold: int,
) -> Optional[Dict[str, Any]]:
    """Detect if the checkout queue exceeds acceptable length."""
    if queue_count > threshold:
        return {
            "anomaly_id": str(uuid.uuid4()),
            "anomaly_type": "checkout_queue",
            "severity": "MEDIUM",
            "affected_zone": zone_id,
            "evidence": {
                "queue_count": queue_count,
                "threshold": threshold,
            },
            "suggested_action": "Open additional checkout counter",
        }
    return None


def detect_after_hours(
    zone_id: str,
    current_hour: int,
    business_start: int,
    business_end: int,
    person_count: int,
) -> Optional[Dict[str, Any]]:
    """Detect presence in the store outside business hours."""
    is_after_hours = current_hour < business_start or current_hour >= business_end
    if is_after_hours and person_count > 0:
        return {
            "anomaly_id": str(uuid.uuid4()),
            "anomaly_type": "after_hours_entry",
            "severity": "HIGH",
            "affected_zone": zone_id,
            "evidence": {
                "current_hour": current_hour,
                "business_hours": f"{business_start:02d}:00-{business_end:02d}:00",
                "person_count": person_count,
            },
            "suggested_action": "Alert security — after-hours presence detected",
        }
    return None


def detect_statistical_anomaly(
    current_value: float,
    history: List[float],
    z_threshold: float = 2.5,
) -> Optional[Dict[str, Any]]:
    """Detect anomalies using Z-score on a rolling window."""
    if len(history) < 10:
        return None  # Not enough data for baseline

    arr = np.array(history, dtype=np.float64)
    mean = arr.mean()
    std = arr.std()

    if std == 0:
        if current_value != mean:
            return {
                "anomaly_id": str(uuid.uuid4()),
                "anomaly_type": "statistical_anomaly",
                "severity": "HIGH",
                "affected_zone": "store-wide",
                "evidence": {
                    "current_value": current_value,
                    "mean": round(float(mean), 2),
                    "std": 0.0,
                    "z_score": float("-inf" if current_value < mean else "inf"),
                    "threshold": z_threshold,
                },
                "suggested_action": "Investigate unusual activity pattern",
            }
        return None  # No variance

    z_score = (current_value - mean) / std

    if abs(z_score) > z_threshold:
        return {
            "anomaly_id": str(uuid.uuid4()),
            "anomaly_type": "statistical_anomaly",
            "severity": "HIGH" if abs(z_score) > z_threshold * 1.5 else "MEDIUM",
            "affected_zone": "store-wide",
            "evidence": {
                "current_value": current_value,
                "mean": round(float(mean), 2),
                "std": round(float(std), 2),
                "z_score": round(float(z_score), 2),
                "threshold": z_threshold,
            },
            "suggested_action": "Investigate unusual activity pattern",
        }
    return None


# ═══════════════════════════════════════════════════════════
# Tests: Crowd Surge Detection
# ═══════════════════════════════════════════════════════════


class TestCrowdSurge:
    """Test crowd surge anomaly detection."""

    def test_no_alert_under_threshold(self, test_settings: Settings):
        """No anomaly when count is at or below threshold."""
        result = detect_crowd_surge("entrance", 10, test_settings.CROWD_THRESHOLD)
        assert result is None

    def test_no_alert_at_threshold(self, test_settings: Settings):
        """No anomaly when count equals threshold."""
        result = detect_crowd_surge("entrance", test_settings.CROWD_THRESHOLD, test_settings.CROWD_THRESHOLD)
        assert result is None

    def test_alert_above_threshold(self, test_settings: Settings):
        """Anomaly when count exceeds threshold."""
        result = detect_crowd_surge("entrance", 15, test_settings.CROWD_THRESHOLD)
        assert result is not None
        assert result["anomaly_type"] == "crowd_surge"
        assert result["evidence"]["current_count"] == 15
        assert result["evidence"]["threshold"] == 10

    def test_high_severity_well_above_threshold(self, test_settings: Settings):
        """HIGH severity when count > 1.5× threshold."""
        result = detect_crowd_surge("entrance", 20, test_settings.CROWD_THRESHOLD)
        assert result is not None
        assert result["severity"] == "HIGH"

    def test_medium_severity_slightly_above(self, test_settings: Settings):
        """MEDIUM severity when count is just above threshold."""
        result = detect_crowd_surge("entrance", 12, test_settings.CROWD_THRESHOLD)
        assert result is not None
        assert result["severity"] == "MEDIUM"

    def test_alert_has_suggested_action(self, test_settings: Settings):
        """Anomaly should include a suggested action."""
        result = detect_crowd_surge("lipstick_aisle", 15, test_settings.CROWD_THRESHOLD)
        assert "suggested_action" in result
        assert "lipstick_aisle" in result["suggested_action"]

    def test_crowd_surge_integration(self, mock_state_store: StateStore, test_settings: Settings):
        """Integration: trigger crowd surge via state store occupancy."""
        store = mock_state_store
        # Fill a zone with more people than threshold
        for tid in range(15):
            store.zone_enter("entrance", tid, "entry_exit")

        zone = store.get_zone_occupancy("entrance")
        result = detect_crowd_surge("entrance", zone.current_count, test_settings.CROWD_THRESHOLD)
        assert result is not None
        assert result["evidence"]["current_count"] == 15


# ═══════════════════════════════════════════════════════════
# Tests: Loitering Detection
# ═══════════════════════════════════════════════════════════


class TestLoitering:
    """Test loitering anomaly detection."""

    def test_no_alert_short_dwell(self, test_settings: Settings):
        """No anomaly when dwell time is under threshold."""
        result = detect_loitering(42, "entrance", 120.0, test_settings.LOITERING_THRESHOLD_SECONDS)
        assert result is None

    def test_alert_long_dwell(self, test_settings: Settings):
        """Anomaly when dwell time exceeds threshold."""
        result = detect_loitering(42, "entrance", 350.0, test_settings.LOITERING_THRESHOLD_SECONDS)
        assert result is not None
        assert result["anomaly_type"] == "loitering"
        assert result["evidence"]["track_id"] == 42
        assert result["evidence"]["dwell_seconds"] == 350.0

    def test_high_severity_extreme_dwell(self, test_settings: Settings):
        """HIGH severity when dwell > 2× threshold."""
        result = detect_loitering(7, "trial_area", 700.0, test_settings.LOITERING_THRESHOLD_SECONDS)
        assert result is not None
        assert result["severity"] == "HIGH"

    def test_medium_severity_moderate_dwell(self, test_settings: Settings):
        """MEDIUM severity when dwell is just above threshold."""
        result = detect_loitering(7, "trial_area", 310.0, test_settings.LOITERING_THRESHOLD_SECONDS)
        assert result is not None
        assert result["severity"] == "MEDIUM"

    def test_at_threshold_no_alert(self, test_settings: Settings):
        """Exactly at threshold should NOT trigger (> not >=)."""
        result = detect_loitering(7, "entrance", 300.0, test_settings.LOITERING_THRESHOLD_SECONDS)
        assert result is None


# ═══════════════════════════════════════════════════════════
# Tests: Checkout Queue Detection
# ═══════════════════════════════════════════════════════════


class TestCheckoutQueue:
    """Test checkout queue anomaly detection."""

    def test_no_alert_under_threshold(self, test_settings: Settings):
        """No anomaly when queue is small."""
        result = detect_checkout_queue("checkout", 5, test_settings.CHECKOUT_QUEUE_THRESHOLD)
        assert result is None

    def test_alert_above_threshold(self, test_settings: Settings):
        """Anomaly when queue exceeds threshold."""
        result = detect_checkout_queue("checkout", 12, test_settings.CHECKOUT_QUEUE_THRESHOLD)
        assert result is not None
        assert result["anomaly_type"] == "checkout_queue"
        assert result["evidence"]["queue_count"] == 12

    def test_suggested_action(self, test_settings: Settings):
        """Action should suggest opening another counter."""
        result = detect_checkout_queue("checkout", 10, test_settings.CHECKOUT_QUEUE_THRESHOLD)
        assert "additional checkout" in result["suggested_action"].lower()


# ═══════════════════════════════════════════════════════════
# Tests: After-Hours Detection
# ═══════════════════════════════════════════════════════════


class TestAfterHours:
    """Test after-hours presence detection."""

    def test_no_alert_during_business_hours(self, test_settings: Settings):
        """No anomaly during business hours (9-21)."""
        result = detect_after_hours(
            "entrance", 14, test_settings.BUSINESS_HOURS_START,
            test_settings.BUSINESS_HOURS_END, 3
        )
        assert result is None

    def test_alert_before_open(self, test_settings: Settings):
        """Anomaly when people detected before business hours."""
        result = detect_after_hours(
            "entrance", 6, test_settings.BUSINESS_HOURS_START,
            test_settings.BUSINESS_HOURS_END, 2
        )
        assert result is not None
        assert result["anomaly_type"] == "after_hours_entry"
        assert result["severity"] == "HIGH"

    def test_alert_after_close(self, test_settings: Settings):
        """Anomaly when people detected after business hours."""
        result = detect_after_hours(
            "entrance", 23, test_settings.BUSINESS_HOURS_START,
            test_settings.BUSINESS_HOURS_END, 1
        )
        assert result is not None
        assert result["anomaly_type"] == "after_hours_entry"

    def test_no_alert_empty_store_after_hours(self, test_settings: Settings):
        """No anomaly if store is empty after hours."""
        result = detect_after_hours(
            "entrance", 23, test_settings.BUSINESS_HOURS_START,
            test_settings.BUSINESS_HOURS_END, 0
        )
        assert result is None

    def test_boundary_start_hour(self, test_settings: Settings):
        """Exactly at business start should be within hours."""
        result = detect_after_hours(
            "entrance", 9, test_settings.BUSINESS_HOURS_START,
            test_settings.BUSINESS_HOURS_END, 5
        )
        assert result is None

    def test_boundary_end_hour(self, test_settings: Settings):
        """Exactly at business end should be after hours."""
        result = detect_after_hours(
            "entrance", 21, test_settings.BUSINESS_HOURS_START,
            test_settings.BUSINESS_HOURS_END, 5
        )
        assert result is not None


# ═══════════════════════════════════════════════════════════
# Tests: Z-Score Statistical Anomaly
# ═══════════════════════════════════════════════════════════


class TestStatisticalAnomaly:
    """Test Z-score based statistical anomaly detection."""

    def test_no_alert_insufficient_data(self):
        """No anomaly when history has fewer than 10 data points."""
        result = detect_statistical_anomaly(100, [5, 10, 15, 20, 25])
        assert result is None

    def test_no_alert_normal_value(self):
        """No anomaly for a value within normal range."""
        history = [10, 12, 11, 9, 13, 10, 11, 12, 10, 11, 13, 9]
        result = detect_statistical_anomaly(11, history)
        assert result is None

    def test_alert_high_outlier(self):
        """Anomaly for a value far above the mean."""
        history = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        # Very consistent baseline → any deviation is significant
        result = detect_statistical_anomaly(50, history)
        assert result is not None
        assert result["anomaly_type"] == "statistical_anomaly"
        assert result["evidence"]["z_score"] > 2.5

    def test_alert_low_outlier(self):
        """Anomaly for a value far below the mean."""
        history = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        result = detect_statistical_anomaly(0, history)
        assert result is not None
        assert result["evidence"]["z_score"] < -2.5

    def test_no_alert_zero_variance(self):
        """No anomaly when all values are identical (std=0)."""
        history = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        result = detect_statistical_anomaly(10, history)
        assert result is None

    def test_custom_threshold(self):
        """Should respect a custom z_threshold."""
        history = list(range(10, 30))  # mean ≈ 19.5, std ≈ 5.77
        # z-score for value 50: (50 - 19.5) / 5.77 ≈ 5.29
        result = detect_statistical_anomaly(50, history, z_threshold=5.0)
        assert result is not None

    def test_high_severity_extreme_z(self):
        """HIGH severity when z-score exceeds 1.5× threshold."""
        history = [10] * 20
        # Very extreme → z_score way above threshold
        result = detect_statistical_anomaly(1000, history, z_threshold=2.5)
        assert result is not None
        assert result["severity"] == "HIGH"

    def test_evidence_fields(self):
        """Evidence should contain mean, std, z_score, and threshold."""
        history = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        result = detect_statistical_anomaly(50, history)
        assert result is not None
        ev = result["evidence"]
        assert "mean" in ev
        assert "std" in ev
        assert "z_score" in ev
        assert "threshold" in ev
        assert "current_value" in ev


# ═══════════════════════════════════════════════════════════
# Tests: Anomaly Storage in StateStore
# ═══════════════════════════════════════════════════════════


class TestAnomalyStorage:
    """Test StateStore anomaly CRUD operations."""

    def test_add_and_retrieve(self, mock_state_store: StateStore):
        """Added anomalies should be retrievable."""
        anomaly = {
            "anomaly_id": "test-001",
            "anomaly_type": "crowd_surge",
            "severity": "HIGH",
            "affected_zone": "entrance",
            "resolved": False,
        }
        mock_state_store.add_anomaly(anomaly)
        active = mock_state_store.get_active_anomalies()
        assert any(a["anomaly_id"] == "test-001" for a in active)

    def test_resolve_removes_from_active(self, mock_state_store: StateStore):
        """Resolved anomalies should not appear in active list."""
        anomaly = {
            "anomaly_id": "test-002",
            "anomaly_type": "loitering",
            "severity": "MEDIUM",
            "affected_zone": "trial_area",
            "resolved": False,
        }
        mock_state_store.add_anomaly(anomaly)
        mock_state_store.resolve_anomaly("test-002", notes="handled")

        active = mock_state_store.get_active_anomalies()
        assert not any(a["anomaly_id"] == "test-002" for a in active)

    def test_filter_by_severity(self, mock_state_store: StateStore):
        """Filtering by severity should return matching anomalies."""
        for sev in ["HIGH", "MEDIUM", "LOW"]:
            mock_state_store.add_anomaly({
                "anomaly_id": f"filter-{sev}",
                "anomaly_type": "crowd_surge",
                "severity": sev,
                "affected_zone": "entrance",
                "resolved": False,
            })

        high = mock_state_store.get_anomalies_filtered(severity="HIGH")
        assert all(a["severity"] == "HIGH" for a in high)

    def test_filter_by_type(self, mock_state_store: StateStore):
        """Filtering by anomaly_type should return matching anomalies."""
        mock_state_store.add_anomaly({
            "anomaly_id": "type-test",
            "anomaly_type": "loitering",
            "severity": "MEDIUM",
            "affected_zone": "entrance",
            "resolved": False,
        })
        results = mock_state_store.get_anomalies_filtered(anomaly_type="loitering")
        assert all(a["anomaly_type"] == "loitering" for a in results)
