# PROMPT: Generate an assertions test suite that validates the pipeline
# output against the spec-defined event schema, funnel monotonicity,
# unique visitor counting, staff exclusion, and heatmap intensity bounds.
# CHANGES MADE: Combined schema validation, funnel ordering, heatmap
# range checks, and staff exclusion into a single assertion module.

"""
Store Intelligence System — Spec Assertions.

Validates that system output conforms to the challenge specification.
Run with: pytest tests/assertions.py -v
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# ═══════════════════════════════════════════════════════════
# Schema Validation
# ═══════════════════════════════════════════════════════════

REQUIRED_EVENT_FIELDS = {
    "event_id", "store_id", "camera_id", "visitor_id",
    "event_type", "timestamp", "zone_id", "dwell_ms",
    "is_staff", "confidence", "metadata",
}

VALID_EVENT_TYPES = {
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
    "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY",
}

VALID_SEVERITIES = {"INFO", "WARN", "CRITICAL"}


def assert_valid_spec_event(event: Dict[str, Any]) -> None:
    """Validate a single event against the spec schema."""
    # Check required fields
    missing = REQUIRED_EVENT_FIELDS - set(event.keys())
    assert not missing, f"Event missing required fields: {missing}"

    # Validate event_id is UUID-like
    assert isinstance(event["event_id"], str), "event_id must be a string"
    assert len(event["event_id"]) >= 32, f"event_id too short: {event['event_id']}"

    # Validate event_type
    assert event["event_type"] in VALID_EVENT_TYPES, (
        f"Invalid event_type: {event['event_type']}"
    )

    # Validate visitor_id format
    vid = event["visitor_id"]
    assert vid is None or (isinstance(vid, str) and vid.startswith("VIS_")), (
        f"visitor_id must start with 'VIS_' or be null, got: {vid}"
    )

    # Validate timestamp is ISO-8601
    ts = event["timestamp"]
    assert isinstance(ts, str), "timestamp must be a string"
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts), (
        f"timestamp not ISO-8601: {ts}"
    )

    # Validate dwell_ms is non-negative integer
    assert isinstance(event["dwell_ms"], int), "dwell_ms must be int"
    assert event["dwell_ms"] >= 0, f"dwell_ms cannot be negative: {event['dwell_ms']}"

    # Validate is_staff is boolean
    assert isinstance(event["is_staff"], bool), "is_staff must be boolean"

    # Validate confidence is float in [0, 1]
    assert isinstance(event["confidence"], (int, float)), "confidence must be numeric"
    assert 0.0 <= event["confidence"] <= 1.0, (
        f"confidence out of range: {event['confidence']}"
    )

    # Validate metadata
    meta = event["metadata"]
    assert isinstance(meta, dict), "metadata must be a dict"
    assert "session_seq" in meta, "metadata.session_seq is required"


def assert_valid_funnel(funnel: List[Dict[str, Any]]) -> None:
    """Validate funnel is monotonically decreasing."""
    assert len(funnel) >= 3, f"Funnel must have at least 3 stages, got {len(funnel)}"

    for stage in funnel:
        assert "stage" in stage, "Each funnel stage must have 'stage'"
        assert "count" in stage, "Each funnel stage must have 'count'"
        assert "drop_off_pct" in stage, "Each funnel stage must have 'drop_off_pct'"
        assert stage["count"] >= 0, f"Funnel count cannot be negative: {stage}"

    # Monotonically decreasing or equal
    for i in range(1, len(funnel)):
        assert funnel[i]["count"] <= funnel[i - 1]["count"], (
            f"Funnel not monotonic: stage '{funnel[i]['stage']}' "
            f"({funnel[i]['count']}) > '{funnel[i-1]['stage']}' "
            f"({funnel[i-1]['count']})"
        )


def assert_valid_heatmap(heatmap: List[Dict[str, Any]]) -> None:
    """Validate heatmap intensities are in [0, 100]."""
    for zone in heatmap:
        assert "zone_id" in zone, "Heatmap zone must have zone_id"
        assert "intensity" in zone, "Heatmap zone must have intensity"
        intensity = zone["intensity"]
        assert 0 <= intensity <= 100, (
            f"Heatmap intensity out of range for {zone['zone_id']}: {intensity}"
        )


def assert_staff_excluded_from_visitors(
    events: List[Dict[str, Any]], unique_visitors: int
) -> None:
    """Verify that staff events are NOT counted in unique_visitors."""
    all_visitor_ids = set()
    staff_visitor_ids = set()

    for event in events:
        vid = event.get("visitor_id")
        if vid:
            all_visitor_ids.add(vid)
            if event.get("is_staff", False):
                staff_visitor_ids.add(vid)

    customer_visitors = all_visitor_ids - staff_visitor_ids
    assert unique_visitors == len(customer_visitors), (
        f"unique_visitors ({unique_visitors}) should equal customer count "
        f"({len(customer_visitors)}), not total ({len(all_visitor_ids)})"
    )


def assert_no_reentry_double_count(
    events: List[Dict[str, Any]], funnel: List[Dict[str, Any]]
) -> None:
    """Verify that REENTRY events don't inflate the entry funnel stage."""
    visitor_ids_with_entry = set()
    for event in events:
        if event.get("event_type") in ("ENTRY", "REENTRY"):
            vid = event.get("visitor_id")
            if vid:
                visitor_ids_with_entry.add(vid)

    entry_stage = next(
        (s for s in funnel if s["stage"].lower() in ("entry", "entrance")), None
    )
    if entry_stage:
        assert entry_stage["count"] == len(visitor_ids_with_entry), (
            f"Entry count ({entry_stage['count']}) should equal unique visitors "
            f"who entered ({len(visitor_ids_with_entry)}), not total entry+reentry events"
        )


def assert_anomaly_format(anomalies: List[Dict[str, Any]]) -> None:
    """Validate anomaly objects conform to spec."""
    for anomaly in anomalies:
        assert "type" in anomaly, "Anomaly must have 'type'"
        assert "severity" in anomaly, "Anomaly must have 'severity'"
        assert anomaly["severity"] in VALID_SEVERITIES, (
            f"Invalid severity: {anomaly['severity']}"
        )
        assert "suggested_action" in anomaly, "Anomaly must have 'suggested_action'"
        assert isinstance(anomaly["suggested_action"], str), (
            "suggested_action must be a string"
        )
        assert len(anomaly["suggested_action"]) > 0, (
            "suggested_action must not be empty"
        )
