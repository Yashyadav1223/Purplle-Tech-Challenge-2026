"""
Store Intelligence System — Event Emitter.

Central event builder and publisher.  Receives ``BaseEvent`` objects from
the pipeline stages, serialises them to JSON, publishes to the configured
``StreamBackend``, stores them in the ``StateStore``, and optionally
broadcasts via a WebSocket callback.

Also manages periodic background tasks that emit footfall and store
summary events at configurable intervals.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

import structlog

from analytics.store import get_state_store, StateStore
from api.schemas import BaseEvent, EventType
from config.settings import get_settings
from streaming.base import StreamBackend

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════
# Topic mapping
# ═══════════════════════════════════════════════════════════

_TOPIC_MAP: Dict[EventType, str] = {
    EventType.PERSON_DETECTED: "store.person_detected",
    EventType.PERSON_LOST: "store.person_detected",
    EventType.ZONE_ENTER: "store.zone_events",
    EventType.ZONE_EXIT: "store.zone_events",
    EventType.DWELL_TIME_UPDATE: "store.zone_events",
    EventType.DWELL_ALERT: "store.zone_events",
    EventType.CROWD_ALERT: "store.anomalies",
    EventType.ANOMALY_DETECTED: "store.anomalies",
    EventType.FOOTFALL_SUMMARY: "store.summaries",
    EventType.STORE_SUMMARY: "store.summaries",
}


def _topic_for(event_type: EventType) -> str:
    """Resolve the Kafka topic for a given event type."""
    return _TOPIC_MAP.get(event_type, "store.events")


# ═══════════════════════════════════════════════════════════
# Event Emitter
# ═══════════════════════════════════════════════════════════


class EventEmitter:
    """
    Publishes pipeline events to the streaming backend and the state store.

    Usage::

        emitter = EventEmitter(stream_backend)
        await emitter.start()          # begin periodic summary tasks
        await emitter.publish(event)   # publish a single event
        await emitter.stop()           # cancel periodic tasks
    """

    def __init__(
        self,
        backend: StreamBackend,
        *,
        store: Optional[StateStore] = None,
        ws_broadcast: Optional[
            Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]
        ] = None,
    ) -> None:
        """
        Args:
            backend: Streaming backend (Kafka / Redis / in-memory).
            store: Optional state store override (defaults to singleton).
            ws_broadcast: Optional async callback ``(event_dict) -> None``
                          used to push events to connected WebSocket clients.
        """
        self._settings = get_settings()
        self._backend = backend
        self._store = store or get_state_store()
        self._ws_broadcast = ws_broadcast
        self._log = logger.bind(component="event_emitter")

        # Periodic task handles
        self._footfall_task: Optional[asyncio.Task[None]] = None
        self._summary_task: Optional[asyncio.Task[None]] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────

    async def start(self) -> None:
        """Launch periodic summary emitter tasks."""
        if self._running:
            return
        self._running = True

        self._footfall_task = asyncio.create_task(
            self._periodic_footfall_summary(),
            name="emitter-footfall",
        )
        self._summary_task = asyncio.create_task(
            self._periodic_store_summary(),
            name="emitter-store-summary",
        )
        self._log.info("event_emitter.started")

    async def stop(self) -> None:
        """Cancel periodic tasks gracefully."""
        self._running = False
        for task in (self._footfall_task, self._summary_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._footfall_task = None
        self._summary_task = None
        self._log.info("event_emitter.stopped")

    # ── Publishing ────────────────────────────────────────

    async def publish(self, event: BaseEvent) -> bool:
        """
        Serialise and publish a single event.

        Steps:
        1. Serialise to dict (then JSON for the backend).
        2. Publish to streaming backend on the correct topic.
        3. Store in the in-memory event history.
        4. Optionally broadcast via WebSocket.

        Returns:
            ``True`` if the event was successfully published.
        """
        topic = _topic_for(event.event_type)
        event_dict = event.model_dump(mode="json")

        # 1. Publish to streaming backend
        published = False
        try:
            published = await self._backend.publish(
                topic=topic,
                event=event_dict,
                key=event.camera_id,
            )
        except Exception:
            self._log.exception(
                "event_emitter.publish_failed",
                topic=topic,
                event_type=event.event_type.value,
            )

        # 2. Store in state store
        try:
            self._store.add_event(event_dict)
        except Exception:
            self._log.exception("event_emitter.store_failed")

        # 3. WebSocket broadcast
        if self._ws_broadcast is not None:
            try:
                await self._ws_broadcast(event_dict)
            except Exception:
                self._log.exception("event_emitter.ws_broadcast_failed")

        if published:
            self._log.debug(
                "event_emitter.published",
                topic=topic,
                event_type=event.event_type.value,
                event_id=event.event_id,
            )

        return published

    async def publish_many(self, events: List[BaseEvent]) -> int:
        """
        Publish a batch of events.

        Returns:
            Number of events successfully published.
        """
        count = 0
        for event in events:
            if await self.publish(event):
                count += 1
        return count



    # ── Event builders ────────────────────────────────────

    @staticmethod
    def build_footfall_summary(
        camera_id: str,
        period: str,
        count: int,
        start_time: datetime,
        end_time: datetime,
    ) -> BaseEvent:
        """Build a footfall summary event."""
        return BaseEvent(
            event_type=EventType.FOOTFALL_SUMMARY,
            camera_id=camera_id,
            timestamp=datetime.utcnow(),
            metadata={
                "period": period,
                "count": count,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
        )

    @staticmethod
    def build_store_summary(summary: Dict[str, Any]) -> BaseEvent:
        """Build a store-wide summary event."""
        return BaseEvent(
            event_type=EventType.STORE_SUMMARY,
            camera_id="system",
            timestamp=datetime.utcnow(),
            metadata=summary,
        )

    @staticmethod
    def build_anomaly_event(
        camera_id: str,
        anomaly_type: str,
        severity: str,
        affected_zone: str,
        evidence: Dict[str, Any],
        suggested_action: str,
        *,
        track_id: Optional[int] = None,
        bbox: Optional[List[float]] = None,
    ) -> BaseEvent:
        """Build an anomaly detection event."""
        return BaseEvent(
            event_type=EventType.ANOMALY_DETECTED,
            camera_id=camera_id,
            timestamp=datetime.utcnow(),
            track_id=track_id,
            bbox=bbox,
            zone_id=affected_zone,
            metadata={
                "anomaly_type": anomaly_type,
                "severity": severity,
                "affected_zone": affected_zone,
                "evidence": evidence,
                "suggested_action": suggested_action,
            },
        )

    @staticmethod
    def build_crowd_alert(
        camera_id: str,
        zone_id: str,
        current_count: int,
        threshold: int,
    ) -> BaseEvent:
        """Build a crowd alert event."""
        return BaseEvent(
            event_type=EventType.CROWD_ALERT,
            camera_id=camera_id,
            timestamp=datetime.utcnow(),
            zone_id=zone_id,
            metadata={
                "current_count": current_count,
                "threshold": threshold,
                "severity": "HIGH" if current_count >= threshold * 1.5 else "MEDIUM",
            },
        )

    # ── Spec-compliant event builders ─────────────────────

    def build_spec_event(
        self,
        event_type: str,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        timestamp: datetime,
        confidence: float = 0.0,
        is_staff: bool = False,
        zone_id: Optional[str] = None,
        dwell_ms: int = 0,
        queue_depth: Optional[int] = None,
        sku_zone: Optional[str] = None,
        session_seq: int = 0,
    ) -> dict:
        """
        Build a spec-compliant event dict matching the challenge schema.

        Args:
            event_type: One of ENTRY, EXIT, REENTRY, ZONE_ENTER,
                        ZONE_EXIT, ZONE_DWELL, etc.
            store_id: Store identifier.
            camera_id: Camera identifier.
            visitor_id: Unique visitor ID (``VIS_XXXXXX``).
            timestamp: UTC timestamp for the event.
            confidence: Detection confidence score (0-1).
            is_staff: Whether the visitor is classified as staff.
            zone_id: Zone the event relates to (if any).
            dwell_ms: Dwell time in milliseconds.
            queue_depth: Current billing queue depth (if relevant).
            sku_zone: SKU zone label (if relevant).
            session_seq: Session sequence counter for the visitor.

        Returns:
            A dict ready for JSON serialisation and publishing.
        """
        return {
            "event_id": str(uuid.uuid4()),
            "store_id": store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": event_type,
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(confidence, 2),
            "metadata": {
                "queue_depth": queue_depth,
                "sku_zone": sku_zone,
                "session_seq": session_seq,
            },
        }

    async def publish_spec_event(self, spec_event: dict) -> None:
        """
        Store a spec-compliant event in the state store and broadcast
        it via WebSocket.

        This is separate from :meth:`publish` which handles internal
        ``BaseEvent`` objects.  Spec events follow the challenge schema
        and are stored / broadcast as plain dicts.

        Args:
            spec_event: A dict produced by :meth:`build_spec_event`.
        """
        # Store in the new MultiStoreState
        try:
            from analytics.multi_store import get_multi_store_state
            ms = get_multi_store_state()
            ms.ingest_event(spec_event)
        except Exception:
            self._log.exception(
                "event_emitter.spec_event_store_failed",
                event_type=spec_event.get("event_type"),
            )

        # Broadcast via WebSocket
        if self._ws_broadcast is not None:
            try:
                await self._ws_broadcast(spec_event)
            except Exception:
                self._log.exception(
                    "event_emitter.spec_event_ws_failed",
                    event_type=spec_event.get("event_type"),
                )

        self._log.debug(
            "event_emitter.spec_event_published",
            event_type=spec_event.get("event_type"),
            event_id=spec_event.get("event_id"),
            visitor_id=spec_event.get("visitor_id"),
        )

    # ── Periodic tasks ────────────────────────────────────

    async def _periodic_footfall_summary(self) -> None:
        """Emit a footfall summary every ``FOOTFALL_SUMMARY_INTERVAL_SECONDS``."""
        interval = self._settings.FOOTFALL_SUMMARY_INTERVAL_SECONDS
        self._log.info(
            "event_emitter.footfall_task_started",
            interval_seconds=interval,
        )

        while self._running:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            try:
                now = datetime.utcnow()
                start = datetime(now.year, now.month, now.day, now.hour, now.minute)
                hourly = self._store.get_hourly_footfall()
                current_hour_count = hourly.get(now.hour, 0)

                event = self.build_footfall_summary(
                    camera_id="system",
                    period="1min",
                    count=current_hour_count,
                    start_time=start,
                    end_time=now,
                )
                await self.publish(event)
                self._log.debug(
                    "event_emitter.footfall_summary_emitted",
                    count=current_hour_count,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                self._log.exception("event_emitter.footfall_summary_error")

    async def _periodic_store_summary(self) -> None:
        """Emit a full store summary every ``STORE_SUMMARY_INTERVAL_SECONDS``."""
        interval = self._settings.STORE_SUMMARY_INTERVAL_SECONDS
        self._log.info(
            "event_emitter.store_summary_task_started",
            interval_seconds=interval,
        )

        while self._running:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            try:
                summary = self._store.get_store_summary()
                event = self.build_store_summary(summary)
                await self.publish(event)
                self._log.debug(
                    "event_emitter.store_summary_emitted",
                    total_visitors=summary.get("total_visitors_today"),
                    current_in_store=summary.get("current_in_store"),
                )
            except asyncio.CancelledError:
                break
            except Exception:
                self._log.exception("event_emitter.store_summary_error")
