"""
Store Intelligence System — Anomaly Detector.

Background asyncio task that checks for anomalies every 10 seconds.

Rule-based anomalies:
1. **Crowd surge** — zone count > mean + 2σ over last 30 min → HIGH
2. **Abandoned object** — stationary bbox with no person for > 60s → MEDIUM
3. **Loitering** — track in non-product zone > 5 min → MEDIUM
4. **Checkout queue buildup** — checkout zone count > threshold → LOW
5. **After-hours entry** — person detected outside business hours → HIGH

Statistical anomaly:
- Rolling Z-score on footfall rate; flag if |z| > ZSCORE_THRESHOLD

Deduplication: same anomaly_type + zone is suppressed for 5 minutes.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import structlog

from analytics.store import StateStore, get_state_store
from config.settings import get_settings

logger = structlog.get_logger()

# Deduplication window in seconds
_DEDUP_WINDOW_SECONDS = 300  # 5 minutes

# Anomaly check interval
_CHECK_INTERVAL_SECONDS = 10


class AnomalyDetector:
    """
    Real-time anomaly detector for the Store Intelligence System.

    Usage::

        detector = AnomalyDetector(
            anomaly_callback=my_publish_fn,
            zone_config=zone_config_dict,
        )
        await detector.start()
        # ... on shutdown ...
        await detector.stop()
    """

    def __init__(
        self,
        state_store: Optional[StateStore] = None,
        anomaly_callback: Optional[Callable] = None,
        zone_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._store = state_store or get_state_store()
        self._anomaly_callback = anomaly_callback
        # zone_config: zone_id -> {"type": "checkout"|"product"|..., "name": "..."}
        self._zone_config = zone_config or {}

        # Deduplication: (anomaly_type, zone_id) -> last_emit_timestamp
        self._dedup_cache: Dict[Tuple[str, str], float] = {}

        # Abandoned-object tracking: bbox_key -> first_seen_timestamp
        self._stationary_objects: Dict[str, float] = {}

        # Task management
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background anomaly detection loop."""
        if self._running:
            logger.warning("anomaly_detector.already_running")
            return

        self._running = True
        self._task = asyncio.create_task(
            self._run_loop(), name="anomaly-detector"
        )
        logger.info("anomaly_detector.started")

    async def stop(self) -> None:
        """Stop the background anomaly detection loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("anomaly_detector.stopped")

    # ── Core Loop ────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main detection loop — runs every 10 seconds."""
        try:
            while self._running:
                try:
                    await self._check_crowd_surge()
                    await self._check_abandoned_objects()
                    await self._check_loitering()
                    await self._check_checkout_queue()
                    await self._check_after_hours_entry()
                    await self._check_statistical_anomaly()
                    self._cleanup_dedup_cache()
                except Exception:
                    logger.exception("anomaly_detector.cycle_error")

                await asyncio.sleep(_CHECK_INTERVAL_SECONDS)

        except asyncio.CancelledError:
            logger.debug("anomaly_detector.loop_cancelled")

    # ── Rule-Based Checks ────────────────────────────────────

    async def _check_crowd_surge(self) -> None:
        """
        Rule 1: Crowd surge.
        Flag if current zone count > mean + 2 * std over the last 30 min.
        """
        with self._store._lock:
            zone_histories = dict(self._store.zone_count_history)

        for zone_id, history in zone_histories.items():
            if len(history) < 6:  # Need at least ~30 seconds of data
                continue

            counts = [count for _, count in history]
            n = len(counts)
            mean = sum(counts) / n
            variance = sum((c - mean) ** 2 for c in counts) / n
            std = math.sqrt(variance) if variance > 0 else 0.0

            current_count = counts[-1] if counts else 0
            threshold = mean + 2.0 * std

            # Also check against the absolute CROWD_THRESHOLD
            settings = get_settings()
            if current_count > threshold and current_count > 3 and std > 0:
                await self._emit_anomaly(
                    anomaly_type="crowd_surge",
                    severity="HIGH",
                    affected_zone=zone_id,
                    evidence={
                        "current_count": current_count,
                        "mean_30min": round(mean, 2),
                        "std_30min": round(std, 2),
                        "threshold": round(threshold, 2),
                        "data_points": n,
                    },
                    suggested_action=(
                        f"Zone '{zone_id}' has {current_count} people, "
                        f"exceeding the statistical threshold of {threshold:.1f}. "
                        "Consider crowd control measures or staff reallocation."
                    ),
                )

    async def _check_abandoned_objects(self) -> None:
        """
        Rule 2: Abandoned object.
        A bounding box that remains stationary (no associated active person
        track) for longer than ABANDONED_OBJECT_THRESHOLD_SECONDS.
        """
        settings = get_settings()
        threshold_s = settings.ABANDONED_OBJECT_THRESHOLD_SECONDS
        now = time.time()

        # Gather all active person track bboxes
        active_bboxes: Set[str] = set()
        with self._store._lock:
            for track in self._store.tracks.values():
                if track.bbox_history:
                    latest = track.bbox_history[-1]
                    bbox = latest.get("bbox")
                    if bbox:
                        active_bboxes.add(_bbox_key(bbox))

        # Check recent events for object detections without matching person
        recent_events = self._store.get_recent_events(200)
        object_bboxes: Dict[str, Dict[str, Any]] = {}

        for evt in recent_events:
            et = evt.get("event_type")
            bbox = evt.get("bbox")
            track_id = evt.get("track_id")

            if et == "person_detected" or not bbox:
                continue

            # Non-person detection with a bbox — potential object
            key = _bbox_key(bbox)
            if key not in active_bboxes:
                if key not in object_bboxes:
                    object_bboxes[key] = {
                        "bbox": bbox,
                        "zone_id": evt.get("zone_id", "unknown"),
                        "camera_id": evt.get("camera_id", "unknown"),
                    }

        # Track stationary objects
        current_keys = set(object_bboxes.keys())

        # Remove keys no longer seen
        stale_keys = set(self._stationary_objects.keys()) - current_keys
        for k in stale_keys:
            self._stationary_objects.pop(k, None)

        # Add new keys
        for k in current_keys:
            if k not in self._stationary_objects:
                self._stationary_objects[k] = now

        # Check for threshold breaches
        for key, first_seen in list(self._stationary_objects.items()):
            duration = now - first_seen
            if duration > threshold_s:
                info = object_bboxes.get(key, {})
                zone_id = info.get("zone_id", "unknown")
                await self._emit_anomaly(
                    anomaly_type="abandoned_object",
                    severity="MEDIUM",
                    affected_zone=zone_id,
                    evidence={
                        "bbox": info.get("bbox"),
                        "duration_seconds": round(duration, 1),
                        "threshold_seconds": threshold_s,
                    },
                    suggested_action=(
                        f"Stationary object detected in zone '{zone_id}' "
                        f"for {duration:.0f}s (threshold: {threshold_s}s). "
                        "Security should investigate."
                    ),
                )
                # Remove so we don't re-trigger immediately after dedup window
                self._stationary_objects.pop(key, None)

    async def _check_loitering(self) -> None:
        """
        Rule 3: Loitering.
        A single track_id in a non-product zone for longer than
        LOITERING_THRESHOLD_SECONDS (default 5 min).
        """
        settings = get_settings()
        threshold_s = settings.LOITERING_THRESHOLD_SECONDS
        now = datetime.utcnow()

        product_zones = {
            zid for zid, zcfg in self._zone_config.items()
            if zcfg.get("type") in ("product", "engagement")
        }

        with self._store._lock:
            tracks_snapshot = {
                tid: (t.current_zone, t.zone_enter_time)
                for tid, t in self._store.tracks.items()
                if t.current_zone and t.zone_enter_time
            }

        for track_id, (zone_id, enter_time) in tracks_snapshot.items():
            if zone_id in product_zones:
                continue  # Browsing product zones is expected

            dwell = (now - enter_time).total_seconds()
            if dwell > threshold_s:
                await self._emit_anomaly(
                    anomaly_type="loitering",
                    severity="MEDIUM",
                    affected_zone=zone_id,
                    evidence={
                        "track_id": track_id,
                        "zone_id": zone_id,
                        "dwell_seconds": round(dwell, 1),
                        "threshold_seconds": threshold_s,
                    },
                    suggested_action=(
                        f"Track {track_id} has been in non-product zone "
                        f"'{zone_id}' for {dwell:.0f}s (limit: {threshold_s}s). "
                        "Staff should check on the individual."
                    ),
                )

    async def _check_checkout_queue(self) -> None:
        """
        Rule 4: Checkout queue buildup.
        Flag if checkout zone occupancy exceeds CHECKOUT_QUEUE_THRESHOLD.
        """
        settings = get_settings()
        threshold = settings.CHECKOUT_QUEUE_THRESHOLD

        checkout_zones = {
            zid for zid, zcfg in self._zone_config.items()
            if zcfg.get("type") == "checkout"
        }

        # If no zone config, try to find checkout zones from store state
        if not checkout_zones:
            with self._store._lock:
                for zone_id in self._store.zones:
                    if "checkout" in zone_id.lower():
                        checkout_zones.add(zone_id)

        for zone_id in checkout_zones:
            zone = self._store.get_zone_occupancy(zone_id)
            if zone is None:
                continue

            if zone.current_count > threshold:
                await self._emit_anomaly(
                    anomaly_type="checkout_queue",
                    severity="LOW",
                    affected_zone=zone_id,
                    evidence={
                        "current_count": zone.current_count,
                        "threshold": threshold,
                        "active_tracks": list(zone.active_track_ids)[:20],
                    },
                    suggested_action=(
                        f"Checkout zone '{zone_id}' has {zone.current_count} "
                        f"people (threshold: {threshold}). "
                        "Open additional checkout lanes."
                    ),
                )

    async def _check_after_hours_entry(self) -> None:
        """
        Rule 5: After-hours entry.
        Any active person_detected event outside business hours → HIGH.
        """
        settings = get_settings()
        current_hour = datetime.utcnow().hour
        business_start = settings.BUSINESS_HOURS_START
        business_end = settings.BUSINESS_HOURS_END

        # Check if we're outside business hours
        if business_start <= current_hour < business_end:
            return  # Within business hours, nothing to flag

        # Any active tracks right now are after-hours
        with self._store._lock:
            active_tracks = list(self._store.tracks.values())

        if not active_tracks:
            return

        # Group by zone for consolidated alerts
        zone_tracks: Dict[str, List[int]] = {}
        for track in active_tracks:
            zone = track.current_zone or "unknown"
            zone_tracks.setdefault(zone, []).append(track.track_id)

        for zone_id, track_ids in zone_tracks.items():
            await self._emit_anomaly(
                anomaly_type="after_hours_entry",
                severity="HIGH",
                affected_zone=zone_id,
                evidence={
                    "current_hour": current_hour,
                    "business_hours": f"{business_start:02d}:00-{business_end:02d}:00",
                    "track_count": len(track_ids),
                    "track_ids": track_ids[:10],
                },
                suggested_action=(
                    f"{len(track_ids)} person(s) detected in zone '{zone_id}' "
                    f"outside business hours ({business_start:02d}:00-{business_end:02d}:00). "
                    "Security should investigate immediately."
                ),
            )

    # ── Statistical Anomaly ──────────────────────────────────

    async def _check_statistical_anomaly(self) -> None:
        """
        Rolling Z-score on footfall rate.
        Flag if |z| > ZSCORE_THRESHOLD (default 2.5).
        """
        settings = get_settings()

        with self._store._lock:
            history = list(self._store.footfall_rate_history)

        if len(history) < 10:
            return  # Not enough data for meaningful statistics

        rates = [rate for _, rate in history]
        n = len(rates)
        mean = sum(rates) / n
        variance = sum((r - mean) ** 2 for r in rates) / n
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0:
            return  # No variation — cannot compute Z-score

        latest_rate = rates[-1]
        z_score = (latest_rate - mean) / std

        if abs(z_score) > settings.ZSCORE_THRESHOLD:
            direction = "spike" if z_score > 0 else "drop"
            await self._emit_anomaly(
                anomaly_type="statistical_anomaly",
                severity="MEDIUM" if abs(z_score) < 4.0 else "HIGH",
                affected_zone="store-wide",
                evidence={
                    "z_score": round(z_score, 3),
                    "threshold": settings.ZSCORE_THRESHOLD,
                    "current_rate": latest_rate,
                    "mean_rate": round(mean, 2),
                    "std_rate": round(std, 2),
                    "direction": direction,
                    "data_points": n,
                },
                suggested_action=(
                    f"Footfall rate {direction} detected (z={z_score:.2f}, "
                    f"threshold=±{settings.ZSCORE_THRESHOLD}). "
                    f"Current rate: {latest_rate}, mean: {mean:.1f}. "
                    "Review camera feeds and staffing levels."
                ),
            )

    # ── Emission & Deduplication ─────────────────────────────

    async def _emit_anomaly(
        self,
        anomaly_type: str,
        severity: str,
        affected_zone: str,
        evidence: Dict[str, Any],
        suggested_action: str,
    ) -> None:
        """
        Emit an anomaly event, respecting the 5-minute deduplication window.
        """
        now = time.time()
        dedup_key = (anomaly_type, affected_zone)

        # Check deduplication
        last_emit = self._dedup_cache.get(dedup_key)
        if last_emit is not None and (now - last_emit) < _DEDUP_WINDOW_SECONDS:
            logger.debug(
                "anomaly_detector.dedup_suppressed",
                anomaly_type=anomaly_type,
                zone=affected_zone,
                seconds_since_last=round(now - last_emit, 1),
            )
            return

        self._dedup_cache[dedup_key] = now

        anomaly_id = str(uuid.uuid4())
        anomaly_record = {
            "anomaly_id": anomaly_id,
            "anomaly_type": anomaly_type,
            "severity": severity,
            "affected_zone": affected_zone,
            "camera_id": "system",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "evidence": evidence,
            "suggested_action": suggested_action,
            "resolved": False,
        }

        # Record in state store
        self._store.add_anomaly(anomaly_record)

        # Also emit as an event
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "anomaly_detected",
            "camera_id": "system",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": anomaly_record,
        }

        logger.warning(
            "anomaly_detector.anomaly_detected",
            anomaly_type=anomaly_type,
            severity=severity,
            zone=affected_zone,
        )

        if self._anomaly_callback is not None:
            try:
                result = self._anomaly_callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "anomaly_detector.callback_error",
                    anomaly_type=anomaly_type,
                )

    def _cleanup_dedup_cache(self) -> None:
        """Remove expired entries from the deduplication cache."""
        now = time.time()
        expired = [
            key for key, ts in self._dedup_cache.items()
            if (now - ts) > _DEDUP_WINDOW_SECONDS * 2
        ]
        for key in expired:
            del self._dedup_cache[key]


# ── Helpers ──────────────────────────────────────────────────


def _bbox_key(bbox: List[float], precision: int = 2) -> str:
    """
    Create a hashable key from a bounding box for stationary-object tracking.
    Rounds coordinates to reduce jitter from minor detection fluctuations.
    """
    rounded = tuple(round(v, precision) for v in bbox)
    return str(rounded)
