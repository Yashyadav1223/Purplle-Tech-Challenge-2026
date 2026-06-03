"""
Store Intelligence System — Zone Mapper.

Loads zone polygon definitions from ``config/zones.json``, determines
which zone each detected person is in, tracks zone transitions, and
emits zone-related events (enter / exit / dwell updates / dwell alerts).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import structlog

from analytics.store import get_state_store, StateStore
from api.schemas import BaseEvent, EventType, ZoneConfig
from config.settings import get_settings

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════
# Internal data
# ═══════════════════════════════════════════════════════════


@dataclass
class _ZoneInfo:
    """Pre-processed zone metadata used at runtime."""

    zone_id: str
    name: str
    zone_type: str
    polygon: np.ndarray  # shape (N, 1, 2) int32, for cv2.pointPolygonTest
    color: str


@dataclass
class _DwellState:
    """Per-(track_id, zone_id) dwell tracking."""

    zone_id: str
    enter_time: float  # monotonic
    last_update_time: float  # monotonic — last time we emitted dwell_time_update
    last_dwell_event_time: float = 0.0  # monotonic — last ZONE_DWELL spec event
    alerted: bool = False  # True once a dwell_alert has been emitted


@dataclass
class _BillingQueueEntry:
    """Records a person joining a billing/checkout zone queue."""

    track_id: int
    zone_id: str
    join_time: float  # monotonic


# ═══════════════════════════════════════════════════════════
# Zone Mapper
# ═══════════════════════════════════════════════════════════


class ZoneMapper:
    """
    Determines which zone each detection falls in, tracks zone
    enter/exit transitions, and emits zone-related events.

    Usage::

        zm = ZoneMapper()
        events = zm.process_detections("cam01", detections)
        for ev in events:
            await emitter.publish(ev)
    """

    def __init__(self, store: Optional[StateStore] = None) -> None:
        self._settings = get_settings()
        self._store = store or get_state_store()
        self._log = logger.bind(component="zone_mapper")

        # zone_id → _ZoneInfo
        self._zones: Dict[str, _ZoneInfo] = {}

        # (track_id) → current zone_id  (None if outside all zones)
        self._track_zones: Dict[int, Optional[str]] = {}

        # (track_id, zone_id) → _DwellState
        self._dwell_states: Dict[Tuple[int, str], _DwellState] = {}

        # Billing queue tracking: track_id → _BillingQueueEntry
        self._billing_queue: Dict[int, _BillingQueueEntry] = {}

        # POS transaction timestamps: track_id → monotonic time of last POS hit
        self._pos_transactions: Dict[int, float] = {}

        # Dwell event interval in seconds (from settings, converted from ms)
        self._dwell_event_interval: float = float(
            getattr(self._settings, "DWELL_EVENT_INTERVAL_MS", 30000)
        ) / 1000.0

        self._load_zones()

    # ── Zone loading ──────────────────────────────────────

    def _load_zones(self) -> None:
        """Load zone polygons from the JSON configuration file."""
        config_path = Path(getattr(self._settings, "STORE_LAYOUT_PATH", "config/store_layout.json"))
        if not config_path.is_absolute():
            # Resolve relative to the project root (cwd assumed)
            config_path = Path.cwd() / config_path

        if not config_path.exists():
            self._log.warning(
                "zone_mapper.config_not_found",
                path=str(config_path),
            )
            return

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            self._log.exception("zone_mapper.config_load_error", path=str(config_path))
            return

        for z in data.get("zones", []):
            try:
                raw_type = z.get("zone_type", z.get("type", "product")).upper()
                if raw_type == "ENTRY_EXIT":
                    z_type = "entry_exit"
                elif raw_type == "BILLING" or raw_type == "CHECKOUT":
                    z_type = "checkout"
                elif raw_type == "DISPLAY":
                    z_type = "engagement"
                else:
                    z_type = "product"
                    
                # Map from store_layout.json schema or fallback to legacy zones.json schema
                zc = ZoneConfig(
                    id=z.get("zone_id", z.get("id")),
                    name=z.get("zone_name", z.get("name", "Unknown")),
                    type=z_type,
                    polygon=z.get("polygon", []),
                    color=z.get("color", "#3B82F6")
                )
            except Exception:
                self._log.warning("zone_mapper.invalid_zone_entry", zone=z)
                continue

            # Convert polygon list to the format cv2.pointPolygonTest expects.
            polygon_np = np.array(zc.polygon, dtype=np.int32).reshape((-1, 1, 2))

            self._zones[zc.id] = _ZoneInfo(
                zone_id=zc.id,
                name=zc.name,
                zone_type=zc.type.value,
                polygon=polygon_np,
                color=zc.color,
            )
            # Initialise zone in the state store.
            self._store.init_zone(zc.id)

        self._log.info("zone_mapper.zones_loaded", count=len(self._zones))

    # ── Public API ────────────────────────────────────────

    @property
    def billing_queue_depth(self) -> int:
        return len(self._billing_queue)

    def process_detections(
        self,
        camera_id: str,
        detections: list,
    ) -> List[BaseEvent]:
        """
        Determine zone membership for each detection and emit
        zone-related events (enter / exit / dwell updates / alerts).

        Args:
            camera_id: Originating camera identifier.
            detections: List of ``Detection`` objects from the detector.

        Returns:
            List of ``BaseEvent`` instances.
        """
        from pipeline.detector import Detection  # local import to avoid circular

        events: List[BaseEvent] = []
        now_mono = time.monotonic()
        now_utc = datetime.utcnow()

        active_tracks: set[int] = set()

        for det in detections:
            if not isinstance(det, Detection):
                continue

            active_tracks.add(det.track_id)
            cx, cy = det.center  # pixel coordinates
            new_zone_id = self._get_zone(cx, cy)

            prev_zone_id = self._track_zones.get(det.track_id)

            # ── Zone transition ───────────────────────────
            if new_zone_id != prev_zone_id:
                # Exit previous zone
                if prev_zone_id is not None:
                    exit_events = self._handle_zone_exit(
                        det.track_id, prev_zone_id, camera_id, det, now_mono, now_utc,
                    )
                    events.extend(exit_events)

                # Enter new zone
                if new_zone_id is not None:
                    enter_events = self._handle_zone_enter(
                        det.track_id, new_zone_id, camera_id, det, now_mono, now_utc,
                    )
                    events.extend(enter_events)

                self._track_zones[det.track_id] = new_zone_id

            # ── Dwell updates for active dwellers ─────────
            if new_zone_id is not None:
                dwell_events = self._check_dwell(
                    det.track_id, new_zone_id, camera_id, det, now_mono, now_utc,
                )
                events.extend(dwell_events)

        # Cleanup dwell states for tracks that have disappeared.
        self._cleanup_stale_dwell_states(active_tracks)

        return events

    # ── Zone lookup ───────────────────────────────────────

    def _get_zone(self, cx: float, cy: float) -> Optional[str]:
        """
        Return the zone_id whose polygon contains the point ``(cx, cy)``,
        or ``None`` if the point is outside all zones.

        If a point lies in multiple overlapping zones, the first match wins.
        """
        point = (float(cx), float(cy))
        for zone in self._zones.values():
            dist = cv2.pointPolygonTest(zone.polygon, point, measureDist=False)
            if dist >= 0:  # inside or on edge
                return zone.zone_id
        return None

    def get_zone(self, cx: float, cy: float) -> Optional[str]:
        """Public accessor for zone lookup (used by external code)."""
        return self._get_zone(cx, cy)

    # ── Zone enter / exit ─────────────────────────────────

    def _handle_zone_enter(
        self,
        track_id: int,
        zone_id: str,
        camera_id: str,
        det: Any,
        now_mono: float,
        now_utc: datetime,
    ) -> List[BaseEvent]:
        """Handle a person entering a zone."""
        zone_info = self._zones.get(zone_id)
        zone_type = zone_info.zone_type if zone_info else "unknown"
        zone_name = zone_info.name if zone_info else zone_id

        # Record in store
        self._store.zone_enter(zone_id, track_id, zone_type)

        # Start dwell tracking
        self._dwell_states[(track_id, zone_id)] = _DwellState(
            zone_id=zone_id,
            enter_time=now_mono,
            last_update_time=now_mono,
        )

        event = BaseEvent(
            event_type=EventType.ZONE_ENTER,
            camera_id=camera_id,
            timestamp=now_utc,
            track_id=track_id,
            bbox=list(det.bbox_normalized),
            zone_id=zone_id,
            metadata={
                "zone_name": zone_name,
                "zone_type": zone_type,
            },
        )

        self._log.info(
            "zone.enter",
            track_id=track_id,
            zone_id=zone_id,
            zone_name=zone_name,
        )
        return [event]

    def _handle_zone_exit(
        self,
        track_id: int,
        zone_id: str,
        camera_id: str,
        det: Any,
        now_mono: float,
        now_utc: datetime,
    ) -> List[BaseEvent]:
        """Handle a person exiting a zone."""
        zone_info = self._zones.get(zone_id)
        zone_type = zone_info.zone_type if zone_info else "unknown"
        zone_name = zone_info.name if zone_info else zone_id

        # Compute dwell time
        dwell_seconds = 0.0
        key = (track_id, zone_id)
        if key in self._dwell_states:
            dwell_seconds = now_mono - self._dwell_states[key].enter_time
            del self._dwell_states[key]

        # Record in store
        self._store.zone_exit(zone_id, track_id, dwell_seconds)

        event = BaseEvent(
            event_type=EventType.ZONE_EXIT,
            camera_id=camera_id,
            timestamp=now_utc,
            track_id=track_id,
            bbox=list(det.bbox_normalized),
            zone_id=zone_id,
            metadata={
                "zone_name": zone_name,
                "zone_type": zone_type,
                "dwell_seconds": round(dwell_seconds, 2),
            },
        )

        self._log.info(
            "zone.exit",
            track_id=track_id,
            zone_id=zone_id,
            dwell_seconds=round(dwell_seconds, 2),
        )
        return [event]

    # ── Dwell time ────────────────────────────────────────

    def _check_dwell(
        self,
        track_id: int,
        zone_id: str,
        camera_id: str,
        det: Any,
        now_mono: float,
        now_utc: datetime,
    ) -> List[BaseEvent]:
        """
        Emit periodic dwell_time_update and one-shot dwell_alert events
        for a person currently dwelling in a zone.

        Also emits a ZONE_DWELL-compatible event every
        ``DWELL_EVENT_INTERVAL_MS / 1000`` seconds of continuous presence.
        """
        events: List[BaseEvent] = []
        key = (track_id, zone_id)
        state = self._dwell_states.get(key)
        if state is None:
            return events

        dwell_seconds = now_mono - state.enter_time
        since_last_update = now_mono - state.last_update_time

        zone_info = self._zones.get(zone_id)
        zone_name = zone_info.name if zone_info else zone_id

        # Periodic dwell update
        if since_last_update >= self._settings.DWELL_UPDATE_INTERVAL_SECONDS:
            state.last_update_time = now_mono
            events.append(
                BaseEvent(
                    event_type=EventType.DWELL_TIME_UPDATE,
                    camera_id=camera_id,
                    timestamp=now_utc,
                    track_id=track_id,
                    bbox=list(det.bbox_normalized),
                    zone_id=zone_id,
                    metadata={
                        "zone_name": zone_name,
                        "dwell_seconds": round(dwell_seconds, 2),
                    },
                )
            )

        # Periodic ZONE_DWELL spec event (every dwell_event_interval seconds)
        since_last_dwell_event = now_mono - state.last_dwell_event_time
        if since_last_dwell_event >= self._dwell_event_interval:
            state.last_dwell_event_time = now_mono
            events.append(
                BaseEvent(
                    event_type=EventType.DWELL_TIME_UPDATE,
                    camera_id=camera_id,
                    timestamp=now_utc,
                    track_id=track_id,
                    bbox=list(det.bbox_normalized),
                    zone_id=zone_id,
                    metadata={
                        "zone_name": zone_name,
                        "dwell_seconds": round(dwell_seconds, 2),
                        "dwell_ms": int(dwell_seconds * 1000),
                        "spec_event_type": "ZONE_DWELL",
                    },
                )
            )
            self._log.debug(
                "zone.dwell_spec_event",
                track_id=track_id,
                zone_id=zone_id,
                dwell_ms=int(dwell_seconds * 1000),
            )

        # One-shot dwell alert
        if (
            not state.alerted
            and dwell_seconds >= self._settings.DWELL_ALERT_THRESHOLD_SECONDS
        ):
            state.alerted = True
            events.append(
                BaseEvent(
                    event_type=EventType.DWELL_ALERT,
                    camera_id=camera_id,
                    timestamp=now_utc,
                    track_id=track_id,
                    bbox=list(det.bbox_normalized),
                    zone_id=zone_id,
                    metadata={
                        "zone_name": zone_name,
                        "dwell_seconds": round(dwell_seconds, 2),
                        "threshold_seconds": self._settings.DWELL_ALERT_THRESHOLD_SECONDS,
                    },
                )
            )
            self._log.warning(
                "zone.dwell_alert",
                track_id=track_id,
                zone_id=zone_id,
                dwell_seconds=round(dwell_seconds, 2),
            )

        return events

    # ── Cleanup ───────────────────────────────────────────

    def _cleanup_stale_dwell_states(self, active_tracks: set[int]) -> None:
        """Remove dwell states for tracks that are no longer active."""
        stale_keys = [
            key for key in self._dwell_states if key[0] not in active_tracks
        ]
        for key in stale_keys:
            del self._dwell_states[key]

    def remove_track(self, track_id: int) -> None:
        """
        Explicitly remove all zone state for a given track.

        Called when a track is marked lost so that lingering
        dwell timers are cleaned up.
        """
        self._track_zones.pop(track_id, None)
        stale_keys = [k for k in self._dwell_states if k[0] == track_id]
        for key in stale_keys:
            del self._dwell_states[key]

    @property
    def zone_count(self) -> int:
        """Number of loaded zones."""
        return len(self._zones)

    @property
    def zone_ids(self) -> List[str]:
        """List of loaded zone identifiers."""
        return list(self._zones.keys())

    # ── Billing / checkout queue helpers ──────────────────

    def _is_billing_zone(self, zone_id: str) -> bool:
        """Return ``True`` if *zone_id* is a checkout or billing zone."""
        zone_info = self._zones.get(zone_id)
        if zone_info is None:
            return False
        return zone_info.zone_type in ("checkout", "billing")

    @property
    def billing_queue_depth(self) -> int:
        """
        Current count of people in zones of type ``checkout`` or ``billing``.

        Iterates all zone occupancies and sums the ``current_count`` for
        qualifying zone types.
        """
        depth = 0
        for zone_id, zone_info in self._zones.items():
            if zone_info.zone_type in ("checkout", "billing"):
                occ = self._store.get_zone_occupancy(zone_id)
                if occ is not None:
                    depth += occ.current_count
        return depth

    def record_billing_queue_join(
        self, track_id: int, zone_id: str, now_mono: float
    ) -> None:
        """
        Note that *track_id* has joined a billing zone queue.

        Called automatically when a person enters a billing zone and
        the queue depth is already > 0.
        """
        self._billing_queue[track_id] = _BillingQueueEntry(
            track_id=track_id,
            zone_id=zone_id,
            join_time=now_mono,
        )
        self._log.info(
            "zone.billing_queue_join",
            track_id=track_id,
            zone_id=zone_id,
            queue_depth=self.billing_queue_depth,
        )

    def record_pos_transaction(self, track_id: int, now_mono: float) -> None:
        """
        Record a POS transaction timestamp for *track_id*.

        This is called by external code (e.g. POS integration) when a
        purchase is completed.
        """
        self._pos_transactions[track_id] = now_mono

    def check_billing_abandon(
        self, track_id: int, zone_id: str, now_mono: float
    ) -> bool:
        """
        Check if a person who exited a billing zone abandoned the queue.

        A queue is considered *abandoned* when:
        - The person was in the billing queue (joined via
          :meth:`record_billing_queue_join`).
        - No POS transaction was recorded within 5 minutes (300 s)
          of their queue join.

        Returns:
            ``True`` if the exit represents a queue abandon.
        """
        entry = self._billing_queue.pop(track_id, None)
        if entry is None:
            return False

        pos_time = self._pos_transactions.get(track_id)
        if pos_time is not None and (pos_time - entry.join_time) <= 300.0:
            # POS transaction occurred within 5 min → not abandoned
            return False

        self._log.warning(
            "zone.billing_queue_abandon",
            track_id=track_id,
            zone_id=zone_id,
            time_in_queue_s=round(now_mono - entry.join_time, 2),
        )
        return True

    def get_track_zone(self, track_id: int) -> Optional[str]:
        """
        Return the current zone_id for *track_id*, or ``None``.

        Useful for external callers that need to know where a
        detection currently resides.
        """
        return self._track_zones.get(track_id)
