"""
Store Intelligence System — Analytics Engine Tests.

Tests footfall counting, zone heatmap computation, conversion funnel
calculation, and peak hours analysis via the StateStore.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set

import numpy as np
import pytest

from analytics.store import StateStore, ZoneOccupancy


# ═══════════════════════════════════════════════════════════
# Analytics Helper Functions (mirroring engine logic)
# ═══════════════════════════════════════════════════════════


def compute_zone_heatmap(
    zones: Dict[str, ZoneOccupancy],
) -> List[Dict[str, Any]]:
    """
    Compute normalised heatmap intensities from zone visit counts.
    Intensity = zone visits / max visits across all zones (0-1).
    """
    max_visits = max((z.total_visits for z in zones.values()), default=1) or 1
    heatmap = []
    for zone_id, zone in zones.items():
        heatmap.append({
            "zone_id": zone_id,
            "intensity": round(zone.total_visits / max_visits, 3),
            "visit_count": zone.total_visits,
            "avg_dwell_seconds": round(
                zone.total_dwell_seconds / zone.total_visits if zone.total_visits else 0, 1
            ),
        })
    return heatmap


def compute_conversion_funnel(
    funnel_counts: Dict[str, Set[int]],
) -> List[Dict[str, Any]]:
    """
    Build the conversion funnel: entrance → product → engagement → checkout.
    Percentage is relative to entrance count.
    """
    entrance_count = len(funnel_counts.get("entrance", set())) or 1
    stages = []
    for stage in ["entrance", "product", "engagement", "checkout"]:
        count = len(funnel_counts.get(stage, set()))
        stages.append({
            "stage": stage,
            "count": count,
            "percentage": round(count / entrance_count * 100, 1),
        })
    return stages


def compute_peak_hours(
    hourly_footfall: Dict[int, int],
) -> Dict[str, Any]:
    """Identify peak hour and return sorted hourly data."""
    if not hourly_footfall:
        return {"peak_hour": None, "peak_count": 0, "hourly": []}

    peak_hour = max(hourly_footfall, key=hourly_footfall.get)
    sorted_hours = sorted(hourly_footfall.items())
    return {
        "peak_hour": peak_hour,
        "peak_count": hourly_footfall[peak_hour],
        "hourly": [{"hour": h, "count": c} for h, c in sorted_hours],
    }


# ═══════════════════════════════════════════════════════════
# Tests: Footfall Counting
# ═══════════════════════════════════════════════════════════


class TestFootfallCounting:
    """Test visitor counting and footfall metrics."""

    def test_initial_count_is_zero(self, mock_state_store: StateStore):
        """Fresh store should have zero visitors."""
        assert mock_state_store.total_visitors_today == 0

    def test_new_tracks_increment_count(self, mock_state_store: StateStore):
        """Each new track should increment total_visitors_today."""
        for tid in range(10):
            mock_state_store.upsert_track(tid, "cam01", [0, 0, 1, 1])
        assert mock_state_store.total_visitors_today == 10

    def test_repeated_upserts_dont_double_count(self, mock_state_store: StateStore):
        """Re-upserting the same track should not increment count."""
        mock_state_store.upsert_track(42, "cam01", [0, 0, 1, 1])
        mock_state_store.upsert_track(42, "cam01", [0.1, 0.1, 0.9, 0.9])
        mock_state_store.upsert_track(42, "cam01", [0.2, 0.2, 0.8, 0.8])
        assert mock_state_store.total_visitors_today == 1

    def test_current_in_store(self, mock_state_store: StateStore):
        """get_current_in_store should return active track count."""
        for tid in range(5):
            mock_state_store.upsert_track(tid, "cam01", [0, 0, 1, 1])
        assert mock_state_store.get_current_in_store() == 5

        # Remove 2
        mock_state_store.mark_track_lost(0)
        mock_state_store.mark_track_lost(1)
        assert mock_state_store.get_current_in_store() == 3

    def test_hourly_footfall_distribution(self, mock_state_store: StateStore):
        """Footfall should be recorded under the correct hour bucket."""
        for tid in range(7):
            mock_state_store.upsert_track(tid, "cam01", [0, 0, 1, 1])
        hourly = mock_state_store.get_hourly_footfall()
        current_hour = datetime.utcnow().hour
        assert hourly[current_hour] == 7

    def test_footfall_rate_recording(self, mock_state_store: StateStore):
        """record_footfall_rate should append to history deque."""
        for rate in [5.0, 7.2, 3.1]:
            mock_state_store.record_footfall_rate(rate)
        assert len(mock_state_store.footfall_rate_history) == 3

    def test_frame_counter(self, mock_state_store: StateStore):
        """increment_frames_processed should track total frames."""
        for _ in range(100):
            mock_state_store.increment_frames_processed()
        assert mock_state_store.total_frames_processed == 100


# ═══════════════════════════════════════════════════════════
# Tests: Zone Heatmap
# ═══════════════════════════════════════════════════════════


class TestZoneHeatmap:
    """Test heatmap intensity computation."""

    def test_empty_zones(self, mock_state_store: StateStore):
        """Zones with no visits should have zero intensity."""
        zones = mock_state_store.get_all_zone_stats()
        heatmap = compute_zone_heatmap(zones)
        assert all(z["intensity"] == 0.0 for z in heatmap)

    def test_single_busy_zone(self, mock_state_store: StateStore):
        """The busiest zone should have intensity 1.0."""
        for tid in range(20):
            mock_state_store.zone_enter("entrance", tid, "entry_exit")
        zones = mock_state_store.get_all_zone_stats()
        heatmap = compute_zone_heatmap(zones)
        entrance = next(z for z in heatmap if z["zone_id"] == "entrance")
        assert entrance["intensity"] == 1.0

    def test_relative_intensities(self, mock_state_store: StateStore):
        """Less-visited zones should have proportionally lower intensity."""
        for tid in range(10):
            mock_state_store.zone_enter("entrance", tid, "entry_exit")
        for tid in range(5):
            mock_state_store.zone_enter("checkout", tid, "checkout")

        zones = mock_state_store.get_all_zone_stats()
        heatmap = compute_zone_heatmap(zones)

        entrance = next(z for z in heatmap if z["zone_id"] == "entrance")
        checkout = next(z for z in heatmap if z["zone_id"] == "checkout")

        assert entrance["intensity"] == 1.0
        assert checkout["intensity"] == 0.5

    def test_heatmap_avg_dwell(self, mock_state_store: StateStore):
        """Average dwell should be computed from zone stats."""
        for tid in range(3):
            mock_state_store.zone_enter("lipstick_aisle", tid, "product")
        mock_state_store.zone_exit("lipstick_aisle", 0, dwell_seconds=30.0)
        mock_state_store.zone_exit("lipstick_aisle", 1, dwell_seconds=60.0)
        mock_state_store.zone_exit("lipstick_aisle", 2, dwell_seconds=90.0)

        zones = mock_state_store.get_all_zone_stats()
        heatmap = compute_zone_heatmap(zones)
        lipstick = next(z for z in heatmap if z["zone_id"] == "lipstick_aisle")
        # 3 visits, total dwell 180s → avg 60s
        assert lipstick["avg_dwell_seconds"] == 60.0

    def test_zone_peak_count(self, mock_state_store: StateStore):
        """Peak count should track the maximum simultaneous occupancy."""
        for tid in range(8):
            mock_state_store.zone_enter("entrance", tid, "entry_exit")
        # Remove some
        mock_state_store.zone_exit("entrance", 0, dwell_seconds=10)
        mock_state_store.zone_exit("entrance", 1, dwell_seconds=10)

        zone = mock_state_store.get_zone_occupancy("entrance")
        assert zone.peak_count == 8  # Peak was 8
        assert zone.current_count == 6  # Currently 6


# ═══════════════════════════════════════════════════════════
# Tests: Conversion Funnel
# ═══════════════════════════════════════════════════════════


class TestConversionFunnel:
    """Test conversion funnel computation."""

    def test_empty_funnel(self, mock_state_store: StateStore):
        """Empty store should have zero counts at all stages."""
        funnel = compute_conversion_funnel(mock_state_store.funnel_counts)
        for stage in funnel:
            assert stage["count"] == 0

    def test_funnel_population(self, mock_state_store: StateStore):
        """Simulated journey should populate the funnel correctly."""
        store = mock_state_store

        # 10 people enter
        for tid in range(10):
            store.zone_enter("entrance", tid, "entry_exit")

        # 6 browse products
        for tid in range(6):
            store.zone_enter("lipstick_aisle", tid, "product")

        # 3 engage in trial
        for tid in range(3):
            store.zone_enter("trial_area", tid, "engagement")

        # 2 checkout
        for tid in range(2):
            store.zone_enter("checkout", tid, "checkout")

        funnel = compute_conversion_funnel(store.funnel_counts)

        entrance = next(s for s in funnel if s["stage"] == "entrance")
        product = next(s for s in funnel if s["stage"] == "product")
        engagement = next(s for s in funnel if s["stage"] == "engagement")
        checkout = next(s for s in funnel if s["stage"] == "checkout")

        assert entrance["count"] == 10
        assert entrance["percentage"] == 100.0
        assert product["count"] == 6
        assert product["percentage"] == 60.0
        assert engagement["count"] == 3
        assert engagement["percentage"] == 30.0
        assert checkout["count"] == 2
        assert checkout["percentage"] == 20.0

    def test_funnel_unique_visitors(self, mock_state_store: StateStore):
        """Same person visiting a zone twice should count as 1."""
        store = mock_state_store

        # Person 0 enters twice (re-enters)
        store.zone_enter("entrance", 0, "entry_exit")
        store.zone_enter("entrance", 0, "entry_exit")

        funnel = store.get_funnel_data()
        assert funnel["entrance"] == 1

    def test_conversion_rate_in_summary(self, mock_state_store: StateStore):
        """Store summary conversion rate should match funnel."""
        store = mock_state_store

        for tid in range(10):
            store.upsert_track(tid, "cam01", [0, 0, 1, 1], zone_id="entrance")
            store.zone_enter("entrance", tid, "entry_exit")

        for tid in [0, 1]:
            store.zone_enter("checkout", tid, "checkout")

        summary = store.get_store_summary()
        assert summary["conversion_rate"] == 20.0  # 2/10 * 100


# ═══════════════════════════════════════════════════════════
# Tests: Peak Hours
# ═══════════════════════════════════════════════════════════


class TestPeakHours:
    """Test peak hours calculation."""

    def test_empty_hourly(self):
        """No footfall data should return None peak."""
        result = compute_peak_hours({})
        assert result["peak_hour"] is None
        assert result["peak_count"] == 0

    def test_single_hour(self):
        """With one hour, it should be the peak."""
        result = compute_peak_hours({14: 25})
        assert result["peak_hour"] == 14
        assert result["peak_count"] == 25

    def test_multiple_hours(self):
        """Peak should be the hour with highest count."""
        hourly = {9: 10, 10: 30, 11: 20, 12: 50, 13: 15, 14: 45}
        result = compute_peak_hours(hourly)
        assert result["peak_hour"] == 12
        assert result["peak_count"] == 50

    def test_hourly_sorted(self):
        """Hourly data should be sorted by hour."""
        hourly = {14: 5, 9: 10, 12: 8, 11: 3}
        result = compute_peak_hours(hourly)
        hours = [h["hour"] for h in result["hourly"]]
        assert hours == [9, 11, 12, 14]

    def test_peak_hours_integration(self, mock_state_store: StateStore):
        """Integration: compute peak hours from state store footfall."""
        store = mock_state_store
        # Manually set hourly footfall
        store.hourly_footfall[9] = 15
        store.hourly_footfall[10] = 40
        store.hourly_footfall[11] = 25
        store.hourly_footfall[14] = 60
        store.hourly_footfall[17] = 35

        hourly = store.get_hourly_footfall()
        result = compute_peak_hours(hourly)
        assert result["peak_hour"] == 14
        assert result["peak_count"] == 60


# ═══════════════════════════════════════════════════════════
# Tests: Store Summary
# ═══════════════════════════════════════════════════════════


class TestStoreSummary:
    """Test the aggregate store summary KPIs."""

    def test_summary_empty_store(self, mock_state_store: StateStore):
        """Empty store should return zeroed KPIs."""
        summary = mock_state_store.get_store_summary()
        assert summary["total_visitors_today"] == 0
        assert summary["current_in_store"] == 0
        assert summary["active_anomalies"] == 0
        assert summary["avg_dwell_seconds"] == 0.0

    def test_summary_with_data(self, populated_state_store: StateStore):
        """Populated store should return non-zero KPIs."""
        summary = populated_state_store.get_store_summary()
        assert summary["total_visitors_today"] > 0
        assert summary["current_in_store"] > 0
        assert summary["active_anomalies"] == 1  # From the fixture anomaly

    def test_avg_dwell_calculation(self, mock_state_store: StateStore):
        """Average dwell should aggregate across all zones."""
        store = mock_state_store
        store.zone_enter("entrance", 0, "entry_exit")
        store.zone_exit("entrance", 0, dwell_seconds=20.0)
        store.zone_enter("checkout", 1, "checkout")
        store.zone_exit("checkout", 1, dwell_seconds=40.0)

        summary = store.get_store_summary()
        assert summary["avg_dwell_seconds"] == 30.0  # (20 + 40) / 2


# ═══════════════════════════════════════════════════════════
# Tests: Event History
# ═══════════════════════════════════════════════════════════


class TestEventHistory:
    """Test event storage and filtering."""

    def test_add_event(self, mock_state_store: StateStore):
        """Events should be added to history."""
        event = {"event_id": "e1", "event_type": "zone_enter", "zone_id": "entrance"}
        mock_state_store.add_event(event)
        events = mock_state_store.get_recent_events(10)
        assert len(events) == 1
        assert events[0]["event_id"] == "e1"

    def test_event_ordering(self, mock_state_store: StateStore):
        """Most recent events should come first."""
        for i in range(5):
            mock_state_store.add_event({"event_id": f"e{i}", "event_type": "zone_enter"})
        events = mock_state_store.get_recent_events(5)
        assert events[0]["event_id"] == "e4"

    def test_event_history_bounded(self):
        """Event history should respect max size."""
        store = StateStore(max_event_history=10)
        for i in range(20):
            store.add_event({"event_id": f"e{i}", "event_type": "test"})
        assert len(store.event_history) == 10

    def test_filter_by_event_type(self, mock_state_store: StateStore):
        """Filtering by event_type should work."""
        mock_state_store.add_event({"event_id": "e1", "event_type": "zone_enter", "zone_id": "entrance"})
        mock_state_store.add_event({"event_id": "e2", "event_type": "zone_exit", "zone_id": "entrance"})
        mock_state_store.add_event({"event_id": "e3", "event_type": "zone_enter", "zone_id": "checkout"})

        results, total = mock_state_store.get_filtered_events(event_types=["zone_enter"])
        assert total == 2
        assert all(e["event_type"] == "zone_enter" for e in results)

    def test_filter_by_zone(self, mock_state_store: StateStore):
        """Filtering by zone_id should work."""
        mock_state_store.add_event({"event_id": "e1", "event_type": "zone_enter", "zone_id": "entrance"})
        mock_state_store.add_event({"event_id": "e2", "event_type": "zone_enter", "zone_id": "checkout"})

        results, total = mock_state_store.get_filtered_events(zone_id="checkout")
        assert total == 1
        assert results[0]["zone_id"] == "checkout"

    def test_pagination(self, mock_state_store: StateStore):
        """Pagination should return correct slices."""
        for i in range(25):
            mock_state_store.add_event({"event_id": f"e{i}", "event_type": "test"})

        page1, total = mock_state_store.get_filtered_events(page=1, page_size=10)
        assert total == 25
        assert len(page1) == 10

        page3, _ = mock_state_store.get_filtered_events(page=3, page_size=10)
        assert len(page3) == 5  # 25 - 20 = 5 remaining
