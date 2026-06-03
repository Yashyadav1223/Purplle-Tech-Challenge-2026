"""
Store Intelligence System — Track State Manager.

Bridges per-frame YOLO detections with the central ``StateStore``.
Detects new track appearances and lost tracks, emitting the appropriate
``person_detected`` and ``person_lost`` events for downstream processing.

Extended capabilities:
- **visitor_id** generation (``VIS_`` + 6 random hex chars) for each track.
- **session_seq** counter per visitor that increments with each event.
- **Re-entry detection**: if a new track appears within
  ``REENTRY_WINDOW_SECONDS`` of a prior exit from the same camera, the
  prior ``visitor_id`` is reused and the event is tagged as *REENTRY*.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import structlog

from analytics.store import get_state_store, StateStore
from api.schemas import BaseEvent, EventType
from config.settings import get_settings
from pipeline.detector import Detection

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════
# Internal bookkeeping
# ═══════════════════════════════════════════════════════════

LOST_TRACK_TIMEOUT_SECONDS: float = 5.0
"""Seconds since last observation before a track is considered lost."""


@dataclass
class _TrackMeta:
    """Lightweight per-track metadata kept by the ``TrackManager``."""

    track_id: int
    camera_id: str
    last_seen_time: float = field(default_factory=time.monotonic)
    first_frame: bool = True
    visitor_id: str = ""
    is_reentry: bool = False


@dataclass
class _ExitRecord:
    """Records a recently exited visitor for re-entry matching."""

    visitor_id: str
    camera_id: str
    exit_time: float  # monotonic


# ═══════════════════════════════════════════════════════════
# Track Manager
# ═══════════════════════════════════════════════════════════


class TrackManager:
    """
    Stateful manager that sits between the detector and the zone mapper.

    For each frame:
    1. Receives a list of ``Detection`` objects.
    2. Upserts every detection into the ``StateStore``.
    3. Identifies **new** tracks → emits ``person_detected``.
    4. Identifies **lost** tracks (not seen for
       ``LOST_TRACK_TIMEOUT_SECONDS``) → emits ``person_lost``.

    Additionally maintains:
    - ``visitor_id`` mapping (track_id → ``VIS_XXXXXX``)
    - ``session_seq`` counters (track_id → int)
    - Re-entry detection via ``_exited_visitor_ids``

    Usage::

        mgr = TrackManager()
        events = mgr.process_frame("cam01", detections)
        for event in events:
            await emitter.publish(event)
    """

    def __init__(self, store: Optional[StateStore] = None) -> None:
        self._settings = get_settings()
        self._store = store or get_state_store()
        self._log = logger.bind(component="track_manager")

        # Local bookkeeping: track_id → _TrackMeta
        self._active: Dict[int, _TrackMeta] = {}

        # Visitor ID mapping: track_id → visitor_id string
        self._visitor_ids: Dict[int, str] = {}

        # Session sequence counters: track_id → int
        self._session_seqs: Dict[int, int] = {}

        # Recently exited visitors for re-entry detection
        self._exited_visitors: List[_ExitRecord] = []

        # Re-entry window from settings
        self._reentry_window: float = float(
            getattr(self._settings, "REENTRY_WINDOW_SECONDS", 300)
        )

    # ── Visitor ID helpers ────────────────────────────────

    @staticmethod
    def _generate_visitor_id() -> str:
        """Generate a visitor ID like ``VIS_a3f1b2``."""
        return "VIS_" + secrets.token_hex(3)

    def get_visitor_id(self, track_id: int) -> str:
        """
        Return the visitor_id associated with *track_id*.

        Returns an empty string if the track has never been seen.
        """
        return self._visitor_ids.get(track_id, "")

    def get_session_seq(self, track_id: int) -> int:
        """
        Return the current session sequence number for *track_id*.

        The counter starts at ``0`` (no events emitted yet).
        """
        return self._session_seqs.get(track_id, 0)

    def increment_session_seq(self, track_id: int) -> None:
        """Increment the session sequence counter for *track_id*."""
        self._session_seqs[track_id] = self._session_seqs.get(track_id, 0) + 1

    # ── Re-entry detection ────────────────────────────────

    def _find_reentry_visitor(self, camera_id: str, now_mono: float) -> Optional[str]:
        """
        Check if a recently exited visitor from the same camera can be
        matched for re-entry.  Returns the prior ``visitor_id`` or ``None``.
        """
        cutoff = now_mono - self._reentry_window
        # Prune expired exits
        self._exited_visitors = [
            r for r in self._exited_visitors if r.exit_time >= cutoff
        ]
        # Search newest-first for a match from the same camera
        for i in range(len(self._exited_visitors) - 1, -1, -1):
            record = self._exited_visitors[i]
            if record.camera_id == camera_id:
                # Consume the record so it isn't reused
                self._exited_visitors.pop(i)
                return record.visitor_id
        return None

    # ── Public API ────────────────────────────────────────

    def process_frame(
        self,
        camera_id: str,
        detections: List[Detection],
    ) -> List[BaseEvent]:
        """
        Process a single frame's detections and return a list of events.

        Args:
            camera_id: Originating camera identifier.
            detections: Detections produced by ``Detector.detect()``.

        Returns:
            List of ``BaseEvent`` instances (person_detected / person_lost).
        """
        events: List[BaseEvent] = []
        now_mono = time.monotonic()
        now_utc = datetime.utcnow()

        # Track IDs seen in *this* frame
        seen_ids: Set[int] = set()

        for det in detections:
            seen_ids.add(det.track_id)

            # Upsert into the central store (zone_id determined later by zone_mapper)
            is_new, _prev_zone, _cur_zone = self._store.upsert_track(
                track_id=det.track_id,
                camera_id=camera_id,
                bbox=list(det.bbox_normalized),
            )

            # Update local bookkeeping
            if det.track_id in self._active:
                meta = self._active[det.track_id]
                meta.last_seen_time = now_mono
                meta.first_frame = False
            else:
                # ── Assign visitor_id (with re-entry check) ──
                reentry_vid = self._find_reentry_visitor(camera_id, now_mono)
                is_reentry = reentry_vid is not None
                visitor_id = reentry_vid or self._generate_visitor_id()

                self._visitor_ids[det.track_id] = visitor_id
                self._session_seqs[det.track_id] = 0

                self._active[det.track_id] = _TrackMeta(
                    track_id=det.track_id,
                    camera_id=camera_id,
                    last_seen_time=now_mono,
                    first_frame=True,
                    visitor_id=visitor_id,
                    is_reentry=is_reentry,
                )
                is_new = True  # treat as new even if store already had it but we didn't

            # Emit person_detected for genuinely new tracks
            if is_new:
                meta = self._active[det.track_id]
                event = BaseEvent(
                    event_type=EventType.PERSON_DETECTED,
                    camera_id=camera_id,
                    timestamp=now_utc,
                    track_id=det.track_id,
                    bbox=list(det.bbox_normalized),
                    metadata={
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "visitor_id": meta.visitor_id,
                        "is_reentry": meta.is_reentry,
                    },
                )
                events.append(event)
                self._log.info(
                    "track.person_detected",
                    track_id=det.track_id,
                    camera_id=camera_id,
                    confidence=det.confidence,
                    visitor_id=meta.visitor_id,
                    is_reentry=meta.is_reentry,
                )

        # ── Detect lost tracks ────────────────────────────
        lost_events = self._check_lost_tracks(now_mono, now_utc)
        events.extend(lost_events)

        return events

    # ── Lost-track detection ──────────────────────────────

    def _check_lost_tracks(
        self,
        now_mono: float,
        now_utc: datetime,
    ) -> List[BaseEvent]:
        """
        Identify tracks that have not been observed for longer than
        ``LOST_TRACK_TIMEOUT_SECONDS`` and emit ``person_lost`` events.
        """
        events: List[BaseEvent] = []
        to_remove: List[int] = []

        for track_id, meta in self._active.items():
            elapsed = now_mono - meta.last_seen_time
            if elapsed > LOST_TRACK_TIMEOUT_SECONDS:
                # Retrieve full state from store for richer metadata
                track_state = self._store.get_track(track_id)
                duration: float = 0.0
                zones_visited: List[str] = []

                if track_state is not None:
                    duration = (
                        track_state.last_seen - track_state.first_seen
                    ).total_seconds()
                    zones_visited = [
                        zh["zone_id"] for zh in track_state.zone_history
                    ]

                # Mark lost in the central store
                self._store.mark_track_lost(track_id)

                # Record exit for re-entry detection
                visitor_id = self._visitor_ids.get(track_id, "")
                if visitor_id:
                    self._exited_visitors.append(
                        _ExitRecord(
                            visitor_id=visitor_id,
                            camera_id=meta.camera_id,
                            exit_time=now_mono,
                        )
                    )

                event = BaseEvent(
                    event_type=EventType.PERSON_LOST,
                    camera_id=meta.camera_id,
                    timestamp=now_utc,
                    track_id=track_id,
                    metadata={
                        "duration_seconds": round(duration, 2),
                        "zones_visited": zones_visited,
                        "visitor_id": visitor_id,
                    },
                )
                events.append(event)
                to_remove.append(track_id)

                self._log.info(
                    "track.person_lost",
                    track_id=track_id,
                    duration_seconds=round(duration, 2),
                    zones_visited=zones_visited,
                    visitor_id=visitor_id,
                )

        for tid in to_remove:
            del self._active[tid]
            # Note: we intentionally keep _visitor_ids and _session_seqs
            # so that downstream code can still look them up after loss.

        return events

    # ── Utilities ─────────────────────────────────────────

    @property
    def active_track_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._active)

    def is_reentry(self, track_id: int) -> bool:
        """Return whether *track_id* was detected as a re-entry."""
        meta = self._active.get(track_id)
        return meta.is_reentry if meta is not None else False

    def reset(self) -> None:
        """Clear all local track state (e.g. on camera switch)."""
        self._active.clear()
        self._visitor_ids.clear()
        self._session_seqs.clear()
        self._exited_visitors.clear()
        self._log.info("track_manager.reset")
