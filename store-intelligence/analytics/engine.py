"""
Store Intelligence System — Analytics Engine.

Background asyncio task that periodically reads from the StateStore and
computes real-time retail metrics:

- **Footfall count**: unique visitors per hour / day
- **Zone heatmap**: visit frequency + avg dwell, normalized 0–1
- **Conversion funnel**: entrance → product → engagement → checkout
- **Peak hours**: sliding 1-hour window visitor count per hour
- **Repeat visitor detection**: track_ids that leave and re-enter

Emits summary events via a configurable ``event_emitter`` callback.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from analytics.store import StateStore, get_state_store
from config.settings import get_settings

logger = structlog.get_logger()


class AnalyticsEngine:
    """
    Central analytics processor for the Store Intelligence System.

    Usage::

        engine = AnalyticsEngine(event_emitter=my_publish_fn)
        await engine.start()
        # ... on shutdown ...
        await engine.stop()
    """

    def __init__(
        self,
        state_store: Optional[StateStore] = None,
        event_emitter: Optional[Callable] = None,
    ) -> None:
        self._store = state_store or get_state_store()
        self._event_emitter = event_emitter

        # Computed metrics (refreshed every cycle)
        self._footfall: Dict[str, Any] = {}
        self._heatmap: List[Dict[str, Any]] = []
        self._funnel: List[Dict[str, Any]] = []
        self._peak_hours: List[Dict[str, Any]] = []

        # Repeat-visitor tracking
        self._exited_track_ids: Set[int] = set()
        self._repeat_visitors: Set[int] = set()

        # Timing
        self._last_footfall_emit: float = 0.0
        self._last_summary_emit: float = 0.0

        # Task management
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background analytics loop."""
        if self._running:
            logger.warning("analytics_engine.already_running")
            return

        self._running = True
        self._task = asyncio.create_task(
            self._run_loop(), name="analytics-engine"
        )
        logger.info("analytics_engine.started")

    async def stop(self) -> None:
        """Stop the background analytics loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("analytics_engine.stopped")

    # ── Getters ──────────────────────────────────────────────

    def get_footfall(self) -> Dict[str, Any]:
        """Return the latest footfall metrics."""
        return dict(self._footfall)

    def get_heatmap(self) -> List[Dict[str, Any]]:
        """Return the latest zone heatmap data."""
        return list(self._heatmap)

    def get_funnel(self) -> List[Dict[str, Any]]:
        """Return the latest conversion funnel stages."""
        return list(self._funnel)

    def get_peak_hours(self) -> List[Dict[str, Any]]:
        """Return the latest peak-hours distribution."""
        return list(self._peak_hours)

    def get_repeat_visitors(self) -> Set[int]:
        """Return track IDs identified as repeat visitors."""
        return set(self._repeat_visitors)

    # ── Core Loop ────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main loop — compute metrics and emit summaries."""
        settings = get_settings()
        update_interval = settings.ANALYTICS_UPDATE_INTERVAL_SECONDS
        footfall_interval = settings.FOOTFALL_SUMMARY_INTERVAL_SECONDS
        summary_interval = settings.STORE_SUMMARY_INTERVAL_SECONDS

        try:
            while self._running:
                try:
                    self._compute_footfall()
                    self._compute_heatmap()
                    self._compute_funnel()
                    self._compute_peak_hours()
                    self._detect_repeat_visitors()

                    now = time.time()

                    # Emit footfall_summary event
                    if now - self._last_footfall_emit >= footfall_interval:
                        await self._emit_footfall_summary()
                        self._last_footfall_emit = now

                    # Emit store_summary event
                    if now - self._last_summary_emit >= summary_interval:
                        await self._emit_store_summary()
                        self._last_summary_emit = now

                except Exception:
                    logger.exception("analytics_engine.cycle_error")

                await asyncio.sleep(update_interval)

        except asyncio.CancelledError:
            logger.debug("analytics_engine.loop_cancelled")

    # ── Metric Computations ──────────────────────────────────

    def _compute_footfall(self) -> None:
        """Compute total unique visitors per hour and per day."""
        hourly = self._store.get_hourly_footfall()
        total_today = self._store.total_visitors_today
        current_hour = datetime.utcnow().hour

        self._footfall = {
            "total_today": total_today,
            "current_hour": hourly.get(current_hour, 0),
            "hourly_breakdown": hourly,
            "current_in_store": self._store.get_current_in_store(),
            "computed_at": datetime.utcnow().isoformat(),
        }

        # Record rate for Z-score anomaly detection
        rate = hourly.get(current_hour, 0)
        self._store.record_footfall_rate(float(rate))

    def _compute_heatmap(self) -> None:
        """
        Compute a normalized intensity heatmap across all zones.

        Intensity = 0.7 * normalized_visit_freq + 0.3 * normalized_avg_dwell
        """
        zones = self._store.get_all_zone_stats()
        if not zones:
            self._heatmap = []
            return

        visit_counts: Dict[str, int] = {}
        avg_dwells: Dict[str, float] = {}

        for zone_id, zone in zones.items():
            visit_counts[zone_id] = zone.total_visits
            if zone.dwell_times:
                avg_dwells[zone_id] = sum(zone.dwell_times) / len(zone.dwell_times)
            else:
                avg_dwells[zone_id] = 0.0

        max_visits = max(visit_counts.values()) if visit_counts else 1
        max_dwell = max(avg_dwells.values()) if avg_dwells else 1.0

        # Guard against division by zero
        max_visits = max(max_visits, 1)
        max_dwell = max(max_dwell, 1.0)

        heatmap: List[Dict[str, Any]] = []
        for zone_id, zone in zones.items():
            norm_visits = visit_counts[zone_id] / max_visits
            norm_dwell = avg_dwells[zone_id] / max_dwell
            intensity = round(0.7 * norm_visits + 0.3 * norm_dwell, 4)
            intensity = max(0.0, min(1.0, intensity))

            heatmap.append({
                "zone_id": zone_id,
                "intensity": intensity,
                "visit_count": visit_counts[zone_id],
                "avg_dwell_seconds": round(avg_dwells[zone_id], 1),
                "current_occupancy": zone.current_count,
                "peak_count": zone.peak_count,
            })

        # Sort by intensity descending
        heatmap.sort(key=lambda z: z["intensity"], reverse=True)
        self._heatmap = heatmap

    def _compute_funnel(self) -> None:
        """
        Compute the conversion funnel:
        entrance → product → engagement → checkout.
        """
        funnel_raw = self._store.get_funnel_data()
        entrance = funnel_raw.get("entrance", 0)

        stages = [
            ("entrance", funnel_raw.get("entrance", 0)),
            ("product", funnel_raw.get("product", 0)),
            ("engagement", funnel_raw.get("engagement", 0)),
            ("checkout", funnel_raw.get("checkout", 0)),
        ]

        funnel: List[Dict[str, Any]] = []
        for stage_name, count in stages:
            pct = (count / entrance * 100) if entrance > 0 else 0.0
            funnel.append({
                "stage": stage_name,
                "count": count,
                "percentage": round(pct, 1),
            })

        self._funnel = funnel

    def _compute_peak_hours(self) -> None:
        """
        Compute visitor count per hour using a sliding 1-hour window
        over the hourly_footfall data.
        """
        hourly = self._store.get_hourly_footfall()
        peak_hours: List[Dict[str, Any]] = []

        for hour in range(24):
            count = hourly.get(hour, 0)
            peak_hours.append({"hour": hour, "count": count})

        peak_hours.sort(key=lambda h: h["count"], reverse=True)
        self._peak_hours = peak_hours

    def _detect_repeat_visitors(self) -> None:
        """
        Detect repeat visitors: track_ids that appeared in lost_tracks
        (meaning they left) and now appear again in active tracks.
        """
        with self._store._lock:
            active_ids = set(self._store.tracks.keys())
            lost_ids = set(self._store.lost_tracks.keys())

        # Track IDs that we've seen exit
        self._exited_track_ids |= lost_ids

        # Repeat = previously exited AND currently active again
        repeats = active_ids & self._exited_track_ids
        if repeats:
            new_repeats = repeats - self._repeat_visitors
            if new_repeats:
                logger.info(
                    "analytics_engine.repeat_visitors_detected",
                    count=len(new_repeats),
                    track_ids=list(new_repeats)[:10],
                )
            self._repeat_visitors |= repeats

    # ── Event Emission ───────────────────────────────────────

    async def _emit_footfall_summary(self) -> None:
        """Emit a footfall_summary event via the event_emitter callback."""
        if self._event_emitter is None:
            return

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "footfall_summary",
            "camera_id": "system",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": {
                "total_today": self._footfall.get("total_today", 0),
                "current_hour_count": self._footfall.get("current_hour", 0),
                "current_in_store": self._footfall.get("current_in_store", 0),
                "repeat_visitors": len(self._repeat_visitors),
            },
        }

        try:
            result = self._event_emitter(event)
            if asyncio.iscoroutine(result):
                await result
            logger.debug("analytics_engine.footfall_summary_emitted")
        except Exception:
            logger.exception("analytics_engine.footfall_emit_error")

    async def _emit_store_summary(self) -> None:
        """Emit a store_summary event via the event_emitter callback."""
        if self._event_emitter is None:
            return

        summary = self._store.get_store_summary()
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "store_summary",
            "camera_id": "system",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": {
                "total_visitors_today": summary.get("total_visitors_today", 0),
                "current_in_store": summary.get("current_in_store", 0),
                "active_anomalies": summary.get("active_anomalies", 0),
                "avg_dwell_seconds": summary.get("avg_dwell_seconds", 0.0),
                "peak_hour": summary.get("peak_hour"),
                "peak_hour_count": summary.get("peak_hour_count", 0),
                "conversion_rate": summary.get("conversion_rate", 0.0),
                "funnel": self._funnel,
                "top_zones": self._heatmap[:5],
                "repeat_visitors": len(self._repeat_visitors),
            },
        }

        try:
            result = self._event_emitter(event)
            if asyncio.iscoroutine(result):
                await result
            logger.debug("analytics_engine.store_summary_emitted")
        except Exception:
            logger.exception("analytics_engine.summary_emit_error")
