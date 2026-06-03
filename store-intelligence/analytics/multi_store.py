"""
Store Intelligence System — Multi-Store State Manager.

Thread-safe in-memory state that maintains per-store event data, visitor
sessions, zone heatmaps, conversion funnels, queue depth tracking, and
anomaly detection.  Designed for the spec-required ``/stores`` endpoints.

All public methods are protected by ``threading.RLock`` for concurrent
safety, matching the pattern established by :class:`analytics.store.StateStore`.
"""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════
# Data-classes for per-store bookkeeping
# ═══════════════════════════════════════════════════════════


@dataclass
class _StoreData:
    """Internal bookkeeping for a single store."""

    # Raw events keyed by event_id for quick lookup
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Visitor tracking (excl. staff)
    unique_visitors: Set[str] = field(default_factory=set)

    # Session tracking – visitor_id → ordered list of events
    sessions: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Zone tracking – zone_id → list of (visitor_id, dwell_seconds)
    zone_visits: Dict[str, List[Tuple[str, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Queue tracking
    queue_current: Set[str] = field(default_factory=set)
    queue_abandoned: int = 0
    queue_completed: int = 0

    # Timestamps
    last_event_ts: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════
# MultiStoreState
# ═══════════════════════════════════════════════════════════


class MultiStoreState:
    """
    Thread-safe in-memory state for multiple stores.

    Responsibilities
    ----------------
    * Store events grouped by ``store_id``.
    * Track unique ``visitor_id`` per store (excluding ``is_staff=true``).
    * Group events by ``visitor_id`` to build visitor sessions / journeys.
    * Compute conversion funnel from sessions
      (Entry → Zone Visit → Billing Queue → Purchase).
    * Compute zone heatmap from zone events.
    * Track queue depth from ``BILLING_QUEUE_JOIN`` / ``BILLING_QUEUE_ABANDON``
      events and ``PURCHASE`` events.
    * Compute ``abandonment_rate = abandoned / (abandoned + completed)``.
    * Detect anomalies: ``queue_spike``, ``dead_zone``, ``conversion_drop``.
    * Maintain ``last_event_timestamp`` per store for health checks.
    * Support ``event_id`` deduplication (idempotent ingest).
    """

    # Maximum ingest batch size
    MAX_BATCH_SIZE: int = 500

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stores: Dict[str, _StoreData] = {}
        self._seen_event_ids: Set[str] = set()

    # ───────────────────────────────────────────────────
    # Internal helpers
    # ───────────────────────────────────────────────────

    def _ensure_store(self, store_id: str) -> _StoreData:
        """Return (or lazily create) the ``_StoreData`` for *store_id*."""
        if store_id not in self._stores:
            self._stores[store_id] = _StoreData()
        return self._stores[store_id]

    @staticmethod
    def _parse_ts(raw: Any) -> Optional[datetime]:
        """Best-effort parse of an ISO-8601 timestamp string or datetime."""
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            try:
                # Strip trailing Z and parse
                cleaned = raw.replace("Z", "+00:00")
                return datetime.fromisoformat(cleaned)
            except (ValueError, TypeError):
                return None
        return None

    # ═══════════════════════════════════════════════════════
    # Ingest
    # ═══════════════════════════════════════════════════════

    def ingest_event(self, event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Ingest a single validated event dict.

        Returns ``(accepted, error_message)``.  If the ``event_id`` was already
        seen the event is silently accepted (idempotent).
        """
        with self._lock:
            event_id: str = event.get("event_id", "")
            store_id: str = event.get("store_id", "")

            if not store_id:
                return False, "Missing store_id"

            # Dedup – silently accept duplicates
            if event_id in self._seen_event_ids:
                return True, None

            self._seen_event_ids.add(event_id)
            sd = self._ensure_store(store_id)
            sd.events.append(event)

            # Timestamp bookkeeping
            ts = self._parse_ts(event.get("timestamp"))
            if ts:
                if sd.last_event_ts is None or ts > sd.last_event_ts:
                    sd.last_event_ts = ts

            # Visitor tracking
            visitor_id: Optional[str] = event.get("visitor_id")
            metadata: Dict[str, Any] = event.get("metadata", {})
            is_staff: bool = event.get("is_staff", False)

            if visitor_id and not is_staff:
                sd.unique_visitors.add(visitor_id)

            # Session tracking (all visitors, even staff, get session data
            # so funnels can be computed, but metrics exclude staff)
            if visitor_id:
                sd.sessions[visitor_id].append(event)

            event_type: str = event.get("event_type", "")

            # Zone tracking
            zone_id: Optional[str] = event.get("zone_id") or metadata.get("zone_id")
            dwell_seconds: float = float(metadata.get("dwell_seconds", 0))
            if zone_id and visitor_id and event_type in (
                "ZONE_VISIT", "ZONE_ENTER", "ZONE_EXIT", "DWELL_TIME_UPDATE",
            ):
                sd.zone_visits[zone_id].append((visitor_id, dwell_seconds))

            # Queue tracking
            if event_type == "BILLING_QUEUE_JOIN" and visitor_id:
                sd.queue_current.add(visitor_id)
            elif event_type == "BILLING_QUEUE_ABANDON" and visitor_id:
                sd.queue_current.discard(visitor_id)
                sd.queue_abandoned += 1
            elif event_type == "PURCHASE" and visitor_id:
                sd.queue_current.discard(visitor_id)
                sd.queue_completed += 1

            return True, None

    # ═══════════════════════════════════════════════════════
    # Store existence
    # ═══════════════════════════════════════════════════════

    def has_store(self, store_id: str) -> bool:
        """Return whether any events have been ingested for *store_id*."""
        with self._lock:
            return store_id in self._stores

    def get_store_ids(self) -> List[str]:
        """Return all known store IDs."""
        with self._lock:
            return list(self._stores.keys())

    # ═══════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════

    def get_metrics(self, store_id: str) -> Dict[str, Any]:
        """
        Compute today's metrics for *store_id*.

        Returns dict with:
        ``unique_visitors``, ``conversion_rate``, ``avg_dwell_per_zone``,
        ``queue_depth``, ``abandonment_rate``.
        """
        with self._lock:
            sd = self._ensure_store(store_id)

            unique_visitors = len(sd.unique_visitors)

            # Conversion rate: visitors who purchased / total unique visitors
            purchasers: Set[str] = set()
            for vid, evts in sd.sessions.items():
                for e in evts:
                    if e.get("event_type") == "PURCHASE":
                        if not e.get("is_staff", False):
                            purchasers.add(vid)
                            break

            conversion_rate = (
                round(len(purchasers) / unique_visitors * 100, 2)
                if unique_visitors > 0
                else 0.0
            )

            # Avg dwell per zone
            avg_dwell_per_zone: Dict[str, float] = {}
            for zone_id, visits in sd.zone_visits.items():
                dwells = [d for _, d in visits if d > 0]
                avg_dwell_per_zone[zone_id] = (
                    round(sum(dwells) / len(dwells), 2) if dwells else 0.0
                )

            # Queue depth
            queue_depth = len(sd.queue_current)

            # Abandonment rate
            total_queue = sd.queue_abandoned + sd.queue_completed
            abandonment_rate = (
                round(sd.queue_abandoned / total_queue * 100, 2)
                if total_queue > 0
                else 0.0
            )

            return {
                "store_id": store_id,
                "unique_visitors": unique_visitors,
                "conversion_rate": conversion_rate,
                "avg_dwell_per_zone": avg_dwell_per_zone,
                "queue_depth": queue_depth,
                "abandonment_rate": abandonment_rate,
            }

    # ═══════════════════════════════════════════════════════
    # Funnel
    # ═══════════════════════════════════════════════════════

    def get_funnel(self, store_id: str) -> List[Dict[str, Any]]:
        """
        Session-based conversion funnel for *store_id*.

        Stages: Entry → Zone Visit → Billing Queue → Purchase.
        Counts by unique sessions (``visitor_id``), not raw events.
        Re-entries do NOT double-count.
        """
        with self._lock:
            sd = self._ensure_store(store_id)

            entry_visitors: Set[str] = set()
            zone_visitors: Set[str] = set()
            billing_visitors: Set[str] = set()
            purchase_visitors: Set[str] = set()

            for vid, evts in sd.sessions.items():
                event_types = {e.get("event_type") for e in evts}

                if event_types & {"ENTRY", "STORE_ENTER", "ZONE_ENTER", "PERSON_DETECTED"}:
                    entry_visitors.add(vid)
                if event_types & {"ZONE_VISIT", "ZONE_ENTER", "DWELL_TIME_UPDATE"}:
                    zone_visitors.add(vid)
                if event_types & {"BILLING_QUEUE_JOIN"}:
                    billing_visitors.add(vid)
                if event_types & {"PURCHASE"}:
                    purchase_visitors.add(vid)

            # If we have unique_visitors but no explicit entry events, treat
            # all non-staff visitors as "entered"
            if not entry_visitors and sd.unique_visitors:
                entry_visitors = set(sd.unique_visitors)

            entry_count = len(entry_visitors) or 1  # avoid div-by-zero for pct

            stages = [
                {
                    "stage": "Entry",
                    "count": len(entry_visitors),
                    "percentage": 100.0,
                    "drop_off_pct": 0.0,
                },
                {
                    "stage": "Zone Visit",
                    "count": len(zone_visitors),
                    "percentage": round(len(zone_visitors) / entry_count * 100, 2),
                    "drop_off_pct": round(
                        (len(entry_visitors) - len(zone_visitors))
                        / max(len(entry_visitors), 1) * 100, 2
                    ),
                },
                {
                    "stage": "Billing Queue",
                    "count": len(billing_visitors),
                    "percentage": round(len(billing_visitors) / entry_count * 100, 2),
                    "drop_off_pct": round(
                        (len(zone_visitors) - len(billing_visitors))
                        / max(len(zone_visitors), 1) * 100, 2
                    ),
                },
                {
                    "stage": "Purchase",
                    "count": len(purchase_visitors),
                    "percentage": round(len(purchase_visitors) / entry_count * 100, 2),
                    "drop_off_pct": round(
                        (len(billing_visitors) - len(purchase_visitors))
                        / max(len(billing_visitors), 1) * 100, 2
                    ),
                },
            ]

            return stages

    # ═══════════════════════════════════════════════════════
    # Heatmap
    # ═══════════════════════════════════════════════════════

    def get_heatmap(self, store_id: str) -> Dict[str, Any]:
        """
        Zone visit frequency + avg dwell, normalised 0-100.

        Returns ``{"zones": [...], "data_confidence": bool}``.
        ``data_confidence`` is ``False`` if fewer than 20 unique sessions
        contributed to zone data.
        """
        with self._lock:
            sd = self._ensure_store(store_id)

            total_sessions = len(sd.sessions)
            data_confidence = total_sessions >= 20

            zone_data: List[Dict[str, Any]] = []
            max_visits = 0

            # First pass — compute raw counts
            zone_summaries: Dict[str, Dict[str, Any]] = {}
            for zone_id, visits in sd.zone_visits.items():
                unique_v = len({vid for vid, _ in visits})
                dwells = [d for _, d in visits if d > 0]
                avg_dwell = round(sum(dwells) / len(dwells), 2) if dwells else 0.0
                zone_summaries[zone_id] = {
                    "zone_id": zone_id,
                    "visit_count": len(visits),
                    "unique_visitors": unique_v,
                    "avg_dwell_seconds": avg_dwell,
                }
                if len(visits) > max_visits:
                    max_visits = len(visits)

            # Second pass — normalise intensity 0-100
            for zs in zone_summaries.values():
                intensity = (
                    round(zs["visit_count"] / max_visits * 100, 2)
                    if max_visits > 0
                    else 0.0
                )
                zone_data.append({
                    "zone_id": zs["zone_id"],
                    "visit_count": zs["visit_count"],
                    "unique_visitors": zs["unique_visitors"],
                    "avg_dwell_seconds": zs["avg_dwell_seconds"],
                    "intensity": intensity,
                })

            return {
                "store_id": store_id,
                "zones": zone_data,
                "data_confidence": data_confidence,
            }

    # ═══════════════════════════════════════════════════════
    # Anomalies
    # ═══════════════════════════════════════════════════════

    def get_anomalies(self, store_id: str) -> List[Dict[str, Any]]:
        """
        Detect and return active anomalies for *store_id*.

        Types:
        * ``queue_spike`` – current queue depth > 5 → WARN / CRITICAL
        * ``dead_zone`` – a zone with no visits in the last 30 minutes → INFO
        * ``conversion_drop`` – conversion rate < 5 % with ≥ 10 visitors → WARN
        """
        with self._lock:
            sd = self._ensure_store(store_id)
            anomalies: List[Dict[str, Any]] = []
            now = datetime.now(timezone.utc)

            # ── queue_spike ───────────────────────────────
            queue_depth = len(sd.queue_current)
            if queue_depth > 5:
                severity = "CRITICAL" if queue_depth > 10 else "WARN"
                anomalies.append({
                    "anomaly_id": str(uuid.uuid4()),
                    "type": "queue_spike",
                    "severity": severity,
                    "store_id": store_id,
                    "detail": {
                        "current_queue_depth": queue_depth,
                    },
                    "suggested_action": (
                        "Open additional billing counters to reduce queue wait time."
                    ),
                    "detected_at": now.isoformat().replace("+00:00", "Z"),
                })

            # ── dead_zone ─────────────────────────────────
            threshold = now - timedelta(minutes=30)
            for zone_id, visits in sd.zone_visits.items():
                # Find most recent event timestamp for this zone
                latest_ts: Optional[datetime] = None
                for vid, _ in visits:
                    # Look up that visitor's events in this zone
                    pass  # timestamps are on events, not zone_visits tuples

                # Check zone activity from raw events
                zone_active = False
                for evt in reversed(sd.events):
                    evt_zone = evt.get("zone_id") or evt.get("metadata", {}).get("zone_id")
                    if evt_zone == zone_id:
                        evt_ts = self._parse_ts(evt.get("timestamp"))
                        if evt_ts and evt_ts >= threshold:
                            zone_active = True
                            break

                if not zone_active and visits:
                    anomalies.append({
                        "anomaly_id": str(uuid.uuid4()),
                        "type": "dead_zone",
                        "severity": "INFO",
                        "store_id": store_id,
                        "detail": {
                            "zone_id": zone_id,
                            "minutes_inactive": 30,
                        },
                        "suggested_action": (
                            f"Zone '{zone_id}' has had no visitor activity for "
                            f"30+ minutes. Consider repositioning merchandise "
                            f"or signage to attract foot traffic."
                        ),
                        "detected_at": now.isoformat().replace("+00:00", "Z"),
                    })

            # ── conversion_drop ───────────────────────────
            unique_visitors = len(sd.unique_visitors)
            if unique_visitors >= 10:
                purchasers: Set[str] = set()
                for vid, evts in sd.sessions.items():
                    for e in evts:
                        if e.get("event_type") == "PURCHASE":
                            if not e.get("is_staff", False):
                                purchasers.add(vid)
                                break

                conv_rate = len(purchasers) / unique_visitors * 100
                if conv_rate < 5.0:
                    anomalies.append({
                        "anomaly_id": str(uuid.uuid4()),
                        "type": "conversion_drop",
                        "severity": "WARN",
                        "store_id": store_id,
                        "detail": {
                            "conversion_rate_pct": round(conv_rate, 2),
                            "unique_visitors": unique_visitors,
                            "purchasers": len(purchasers),
                        },
                        "suggested_action": (
                            "Conversion rate has dropped below 5%. Review "
                            "checkout experience, pricing, and in-store "
                            "promotions to improve purchase completion."
                        ),
                        "detected_at": now.isoformat().replace("+00:00", "Z"),
                    })

            return anomalies

    # ═══════════════════════════════════════════════════════
    # Health
    # ═══════════════════════════════════════════════════════

    def get_health_stores(self) -> List[Dict[str, Any]]:
        """
        Per-store health information.

        Each entry includes ``store_id``, ``last_event_timestamp``, ``status``,
        and ``event_count``.  Status is ``STALE_FEED`` when the last event
        is more than 10 minutes old.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            stale_threshold = now - timedelta(minutes=10)
            result: List[Dict[str, Any]] = []

            for store_id, sd in self._stores.items():
                last_ts = sd.last_event_ts
                is_stale = False

                if last_ts is not None:
                    # Normalise to naive UTC for comparison
                    cmp_ts = last_ts.replace(tzinfo=None) if last_ts.tzinfo else last_ts
                    is_stale = cmp_ts < stale_threshold

                result.append({
                    "store_id": store_id,
                    "last_event_timestamp": (
                        last_ts.isoformat().replace("+00:00", "Z") if last_ts else None
                    ),
                    "event_count": len(sd.events),
                    "status": "STALE_FEED" if is_stale else "OK",
                })

            return result

    def get_last_event_ts(self, store_id: str) -> Optional[datetime]:
        """Return the last event timestamp for *store_id*, or ``None``."""
        with self._lock:
            sd = self._stores.get(store_id)
            return sd.last_event_ts if sd else None


# ═══════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════

_multi_store: Optional[MultiStoreState] = None


def get_multi_store_state() -> MultiStoreState:
    """Get the global :class:`MultiStoreState` singleton."""
    global _multi_store
    if _multi_store is None:
        _multi_store = MultiStoreState()
    return _multi_store
