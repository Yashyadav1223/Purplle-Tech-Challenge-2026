"""
Store Intelligence System — Thread-Safe In-Memory State Store.

Central state store used by analytics engine, anomaly detector, and API layer.
All access is protected by threading.RLock for concurrent safety.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Set


@dataclass
class TrackState:
    """State of a single tracked person."""

    track_id: int
    camera_id: str
    first_seen: datetime
    last_seen: datetime
    bbox_history: List[Dict[str, Any]] = field(default_factory=list)
    zone_history: List[Dict[str, Any]] = field(default_factory=list)
    current_zone: Optional[str] = None
    zone_enter_time: Optional[datetime] = None
    active: bool = True


@dataclass
class ZoneOccupancy:
    """Current state of a zone."""

    zone_id: str
    current_count: int = 0
    active_track_ids: Set[int] = field(default_factory=set)
    total_visits: int = 0
    total_dwell_seconds: float = 0.0
    peak_count: int = 0
    dwell_times: List[float] = field(default_factory=list)


@dataclass
class FootfallEntry:
    """Single footfall data point for time-series."""

    timestamp: datetime
    count: int
    zone_id: Optional[str] = None


class StateStore:
    """
    Thread-safe in-memory state store for the entire system.
    
    Stores active tracks, zone occupancy, footfall counters,
    event history, anomaly records, and rolling windows.
    """

    def __init__(self, max_event_history: int = 5000, max_footfall_history: int = 1440):
        self._lock = threading.RLock()

        # ── Track State ──────────────────────────────────
        self.tracks: Dict[int, TrackState] = {}
        self.lost_tracks: Dict[int, TrackState] = {}

        # ── Zone Occupancy ───────────────────────────────
        self.zones: Dict[str, ZoneOccupancy] = {}

        # ── Event History (bounded deque) ────────────────
        self.event_history: Deque[Dict[str, Any]] = deque(maxlen=max_event_history)

        # ── Anomaly Records ──────────────────────────────
        self.anomalies: Dict[str, Dict[str, Any]] = {}

        # ── Footfall Counters ────────────────────────────
        self.total_visitors_today: int = 0
        self.hourly_footfall: Dict[int, int] = defaultdict(int)  # hour -> count
        self.footfall_history: Deque[FootfallEntry] = deque(maxlen=max_footfall_history)

        # ── Rolling Windows ──────────────────────────────
        self.zone_count_history: Dict[str, Deque[tuple]] = defaultdict(
            lambda: deque(maxlen=360)  # 30 min at 5s intervals
        )
        self.footfall_rate_history: Deque[tuple] = deque(maxlen=1440)  # 24h at 1min intervals

        # ── Conversion Funnel ────────────────────────────
        self.funnel_counts: Dict[str, Set[int]] = {
            "entrance": set(),
            "product": set(),
            "engagement": set(),
            "checkout": set(),
        }

        # ── Camera State ─────────────────────────────────
        self.cameras: Dict[str, Dict[str, Any]] = {}

        # ── Timing ───────────────────────────────────────
        self.start_time: datetime = datetime.utcnow()
        self.total_events_emitted: int = 0
        self.total_frames_processed: int = 0

    # ═══════════════════════════════════════════════════════
    # Track Operations
    # ═══════════════════════════════════════════════════════

    def upsert_track(self, track_id: int, camera_id: str, bbox: List[float],
                     zone_id: Optional[str] = None) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Update or insert a track. Returns (is_new, prev_zone, current_zone).
        """
        with self._lock:
            now = datetime.utcnow()
            is_new = track_id not in self.tracks

            if is_new:
                self.tracks[track_id] = TrackState(
                    track_id=track_id,
                    camera_id=camera_id,
                    first_seen=now,
                    last_seen=now,
                    current_zone=zone_id,
                    zone_enter_time=now if zone_id else None,
                )
                self.total_visitors_today += 1
                hour = now.hour
                self.hourly_footfall[hour] += 1

            track = self.tracks[track_id]
            prev_zone = track.current_zone
            track.last_seen = now
            track.active = True
            track.bbox_history.append({
                "bbox": bbox, "timestamp": now.isoformat(), "zone_id": zone_id
            })
            # Keep bbox history bounded
            if len(track.bbox_history) > 500:
                track.bbox_history = track.bbox_history[-500:]

            # Zone transition
            if zone_id != prev_zone:
                track.current_zone = zone_id
                track.zone_enter_time = now if zone_id else None
                if zone_id:
                    track.zone_history.append({
                        "zone_id": zone_id, "enter_time": now.isoformat()
                    })

            return is_new, prev_zone, zone_id

    def mark_track_lost(self, track_id: int) -> Optional[TrackState]:
        """Mark a track as lost and move to lost_tracks."""
        with self._lock:
            if track_id in self.tracks:
                track = self.tracks.pop(track_id)
                track.active = False
                self.lost_tracks[track_id] = track
                return track
            return None

    def get_track(self, track_id: int) -> Optional[TrackState]:
        """Get a track by ID (active or lost)."""
        with self._lock:
            return self.tracks.get(track_id) or self.lost_tracks.get(track_id)

    def get_active_track_ids(self) -> Set[int]:
        """Get all active track IDs."""
        with self._lock:
            return set(self.tracks.keys())

    # ═══════════════════════════════════════════════════════
    # Zone Operations
    # ═══════════════════════════════════════════════════════

    def init_zone(self, zone_id: str):
        """Initialize a zone if not already present."""
        with self._lock:
            if zone_id not in self.zones:
                self.zones[zone_id] = ZoneOccupancy(zone_id=zone_id)

    def zone_enter(self, zone_id: str, track_id: int, zone_type: str):
        """Record a person entering a zone."""
        with self._lock:
            self.init_zone(zone_id)
            zone = self.zones[zone_id]
            zone.active_track_ids.add(track_id)
            zone.current_count = len(zone.active_track_ids)
            zone.total_visits += 1
            if zone.current_count > zone.peak_count:
                zone.peak_count = zone.current_count

            # Update funnel
            funnel_key = self._zone_type_to_funnel(zone_type)
            if funnel_key:
                self.funnel_counts[funnel_key].add(track_id)

            # Record count for anomaly detection
            self.zone_count_history[zone_id].append(
                (time.time(), zone.current_count)
            )

    def zone_exit(self, zone_id: str, track_id: int, dwell_seconds: float):
        """Record a person exiting a zone."""
        with self._lock:
            self.init_zone(zone_id)
            zone = self.zones[zone_id]
            zone.active_track_ids.discard(track_id)
            zone.current_count = len(zone.active_track_ids)
            zone.total_dwell_seconds += dwell_seconds
            zone.dwell_times.append(dwell_seconds)
            # Keep dwell times bounded
            if len(zone.dwell_times) > 1000:
                zone.dwell_times = zone.dwell_times[-1000:]

    def get_zone_occupancy(self, zone_id: str) -> Optional[ZoneOccupancy]:
        """Get current zone occupancy."""
        with self._lock:
            return self.zones.get(zone_id)

    def get_all_zone_stats(self) -> Dict[str, ZoneOccupancy]:
        """Get all zone occupancy data."""
        with self._lock:
            return dict(self.zones)

    @staticmethod
    def _zone_type_to_funnel(zone_type: str) -> Optional[str]:
        mapping = {
            "entry_exit": "entrance",
            "product": "product",
            "engagement": "engagement",
            "checkout": "checkout",
        }
        return mapping.get(zone_type)

    # ═══════════════════════════════════════════════════════
    # Event Operations
    # ═══════════════════════════════════════════════════════

    def add_event(self, event: Dict[str, Any]):
        """Add an event to history."""
        with self._lock:
            self.event_history.appendleft(event)
            self.total_events_emitted += 1

    def get_recent_events(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get most recent N events."""
        with self._lock:
            return list(self.event_history)[:count]

    def get_filtered_events(
        self,
        event_types: Optional[List[str]] = None,
        zone_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Filter and paginate events. Returns (events, total_count)."""
        with self._lock:
            filtered = list(self.event_history)

            if event_types:
                filtered = [e for e in filtered if e.get("event_type") in event_types]
            if zone_id:
                filtered = [e for e in filtered if e.get("zone_id") == zone_id]
            if camera_id:
                filtered = [e for e in filtered if e.get("camera_id") == camera_id]
            if start_time:
                start_iso = start_time.isoformat()
                filtered = [e for e in filtered if e.get("timestamp", "") >= start_iso]
            if end_time:
                end_iso = end_time.isoformat()
                filtered = [e for e in filtered if e.get("timestamp", "") <= end_iso]

            total = len(filtered)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            return filtered[start_idx:end_idx], total

    # ═══════════════════════════════════════════════════════
    # Anomaly Operations
    # ═══════════════════════════════════════════════════════

    def add_anomaly(self, anomaly: Dict[str, Any]):
        """Add an anomaly record."""
        with self._lock:
            anomaly_id = anomaly.get("anomaly_id", "")
            self.anomalies[anomaly_id] = anomaly

    def get_active_anomalies(self) -> List[Dict[str, Any]]:
        """Get all unresolved anomalies."""
        with self._lock:
            return [a for a in self.anomalies.values() if not a.get("resolved", False)]

    def resolve_anomaly(self, anomaly_id: str, notes: str = "") -> bool:
        """Resolve an anomaly. Returns True if found."""
        with self._lock:
            if anomaly_id in self.anomalies:
                self.anomalies[anomaly_id]["resolved"] = True
                self.anomalies[anomaly_id]["resolved_at"] = datetime.utcnow().isoformat()
                self.anomalies[anomaly_id]["resolved_notes"] = notes
                return True
            return False

    def get_anomalies_filtered(
        self,
        severity: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        resolved: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Filter anomalies."""
        with self._lock:
            result = list(self.anomalies.values())
            if severity:
                result = [a for a in result if a.get("severity") == severity]
            if anomaly_type:
                result = [a for a in result if a.get("anomaly_type") == anomaly_type]
            if resolved is not None:
                result = [a for a in result if a.get("resolved", False) == resolved]
            return result

    # ═══════════════════════════════════════════════════════
    # Analytics Helpers
    # ═══════════════════════════════════════════════════════

    def get_current_in_store(self) -> int:
        """Number of people currently in the store."""
        with self._lock:
            return len(self.tracks)

    def get_store_summary(self) -> Dict[str, Any]:
        """Get store-wide KPIs."""
        with self._lock:
            active_anomalies = len([a for a in self.anomalies.values() if not a.get("resolved")])

            # Average dwell
            all_dwells = []
            for zone in self.zones.values():
                all_dwells.extend(zone.dwell_times)
            avg_dwell = sum(all_dwells) / len(all_dwells) if all_dwells else 0.0

            # Peak hour
            peak_hour = max(self.hourly_footfall, key=self.hourly_footfall.get) if self.hourly_footfall else None
            peak_count = self.hourly_footfall.get(peak_hour, 0) if peak_hour is not None else 0

            # Conversion rate
            entered = len(self.funnel_counts["entrance"])
            checked_out = len(self.funnel_counts["checkout"])
            conversion = (checked_out / entered * 100) if entered > 0 else 0.0

            return {
                "total_visitors_today": self.total_visitors_today,
                "current_in_store": len(self.tracks),
                "active_anomalies": active_anomalies,
                "avg_dwell_seconds": round(avg_dwell, 1),
                "peak_hour": peak_hour,
                "peak_hour_count": peak_count,
                "conversion_rate": round(conversion, 1),
            }

    def get_funnel_data(self) -> Dict[str, int]:
        """Get conversion funnel counts."""
        with self._lock:
            return {k: len(v) for k, v in self.funnel_counts.items()}

    def get_hourly_footfall(self) -> Dict[int, int]:
        """Get hourly footfall distribution."""
        with self._lock:
            return dict(self.hourly_footfall)

    def increment_frames_processed(self):
        """Increment the total frames processed counter."""
        with self._lock:
            self.total_frames_processed += 1

    def record_footfall_rate(self, rate: float):
        """Record a footfall rate data point for Z-score computation."""
        with self._lock:
            self.footfall_rate_history.append((time.time(), rate))

    # ═══════════════════════════════════════════════════════
    # Camera Operations
    # ═══════════════════════════════════════════════════════

    def register_camera(self, camera_id: str, source: str):
        """Register a camera."""
        with self._lock:
            self.cameras[camera_id] = {
                "camera_id": camera_id,
                "source": source,
                "status": "active",
                "frames_processed": 0,
                "started_at": datetime.utcnow().isoformat(),
            }

    def update_camera_status(self, camera_id: str, status: str):
        """Update camera status."""
        with self._lock:
            if camera_id in self.cameras:
                self.cameras[camera_id]["status"] = status

    def get_cameras(self) -> List[Dict[str, Any]]:
        """Get all camera info."""
        with self._lock:
            return list(self.cameras.values())

    def increment_camera_frames(self, camera_id: str):
        """Increment frames processed for a camera."""
        with self._lock:
            if camera_id in self.cameras:
                self.cameras[camera_id]["frames_processed"] += 1


# Singleton
_store: Optional[StateStore] = None


def get_state_store() -> StateStore:
    """Get the global state store singleton."""
    global _store
    if _store is None:
        _store = StateStore()
    return _store
