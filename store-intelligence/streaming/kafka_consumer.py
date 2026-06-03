"""
Store Intelligence System — Kafka Event Consumer.

Standalone consumer that:
1. Consumes from all ``store.*`` Kafka topics.
2. Deserializes JSON events.
3. Persists events to the database via SQLAlchemy async.
4. Feeds events into an analytics callback for real-time processing.

Designed to run as a long-lived background asyncio task.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog
from sqlalchemy import Column, DateTime, Integer, String, Text, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config.settings import get_settings

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════
# SQLAlchemy Model
# ═══════════════════════════════════════════════════════════


class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    """Persisted event row."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), unique=True, nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    track_id = Column(Integer, nullable=True)
    zone_id = Column(String(64), nullable=True, index=True)
    topic = Column(String(128), nullable=False)
    payload = Column(Text, nullable=False)  # Full JSON


# ═══════════════════════════════════════════════════════════
# Consumer
# ═══════════════════════════════════════════════════════════


class KafkaEventConsumer:
    """
    Kafka consumer that persists events and dispatches to analytics.

    Usage::

        consumer = KafkaEventConsumer(
            topics=["store.events", "store.zone_events"],
            analytics_callback=my_engine.ingest_event,
        )
        await consumer.start()
        # ... on shutdown ...
        await consumer.stop()
    """

    def __init__(
        self,
        topics: Optional[List[str]] = None,
        analytics_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        settings = get_settings()

        self._topics = topics or [
            f"{settings.KAFKA_TOPIC_PREFIX}.events",
            f"{settings.KAFKA_TOPIC_PREFIX}.zone_events",
            f"{settings.KAFKA_TOPIC_PREFIX}.anomalies",
        ]
        self._analytics_callback = analytics_callback

        # Database
        self._engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Kafka thread coordination
        self._stop_event = threading.Event()
        self._consumer_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running: bool = False

        # Batch buffer for DB writes
        self._batch: List[Dict[str, Any]] = []
        self._batch_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._batch_size = 50
        self._flush_interval_seconds = 5.0

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize the database schema and start consuming."""
        self._loop = asyncio.get_running_loop()

        # Create tables if they don't exist
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("kafka_consumer.db_initialized")

        self._running = True

        # Periodic DB flush task
        self._flush_task = asyncio.create_task(
            self._periodic_flush(), name="kafka-consumer-flush"
        )

        # Start Kafka polling in a background thread
        self._stop_event.clear()
        self._consumer_thread = threading.Thread(
            target=self._consume_loop,
            daemon=True,
            name="kafka-event-consumer",
        )
        self._consumer_thread.start()
        logger.info("kafka_consumer.started", topics=self._topics)

    async def stop(self) -> None:
        """Gracefully shut down consumer and flush remaining events."""
        self._running = False
        self._stop_event.set()

        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=10)
            self._consumer_thread = None

        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush
        await self._flush_batch()

        await self._engine.dispose()
        logger.info("kafka_consumer.stopped")

    # ── Kafka Thread ─────────────────────────────────────────

    def _consume_loop(self) -> None:
        """
        Runs in a daemon thread. Polls Kafka, dispatches events
        into the asyncio event loop for persistence and analytics.
        """
        try:
            from confluent_kafka import Consumer, KafkaError  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "kafka_consumer.import_error",
                hint="Install confluent-kafka>=2.3.0",
            )
            return

        settings = get_settings()
        conf = {
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "store-intelligence-event-persister",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
            "session.timeout.ms": 30000,
            "max.poll.interval.ms": 300000,
        }

        consumer: Optional[Any] = None
        try:
            consumer = Consumer(conf)
            consumer.subscribe(self._topics)
            logger.info("kafka_consumer.consumer_created", topics=self._topics)

            while not self._stop_event.is_set():
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    error = msg.error()
                    if error.code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("kafka_consumer.poll_error", error=str(error))
                    continue

                try:
                    raw = msg.value()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    event: Dict[str, Any] = json.loads(raw)
                    event["_topic"] = msg.topic()
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning(
                        "kafka_consumer.deserialize_error",
                        error=str(exc),
                    )
                    continue

                # Schedule processing on the event loop
                if self._loop is not None and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._process_event(event), self._loop
                    )

        except Exception:
            logger.exception("kafka_consumer.consume_loop_fatal")
        finally:
            if consumer is not None:
                try:
                    consumer.close()
                except Exception:
                    logger.exception("kafka_consumer.consumer_close_error")

    # ── Event Processing ─────────────────────────────────────

    async def _process_event(self, event: Dict[str, Any]) -> None:
        """Buffer event for DB persistence and dispatch to analytics."""
        # Analytics callback
        if self._analytics_callback is not None:
            try:
                result = self._analytics_callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "kafka_consumer.analytics_callback_error",
                    event_type=event.get("event_type"),
                )

        # Buffer for batch DB write
        async with self._batch_lock:
            self._batch.append(event)
            if len(self._batch) >= self._batch_size:
                await self._flush_batch()

    async def _flush_batch(self) -> None:
        """Write buffered events to the database in a single transaction."""
        async with self._batch_lock:
            if not self._batch:
                return
            batch = self._batch.copy()
            self._batch.clear()

        records: List[EventRecord] = []
        for evt in batch:
            try:
                records.append(
                    EventRecord(
                        event_id=evt.get("event_id", ""),
                        event_type=evt.get("event_type", ""),
                        camera_id=evt.get("camera_id", ""),
                        timestamp=_parse_timestamp(evt.get("timestamp")),
                        track_id=evt.get("track_id"),
                        zone_id=evt.get("zone_id"),
                        topic=evt.pop("_topic", "unknown"),
                        payload=json.dumps(evt, default=str),
                    )
                )
            except Exception:
                logger.exception("kafka_consumer.record_build_error")

        if not records:
            return

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    session.add_all(records)
            logger.debug("kafka_consumer.batch_flushed", count=len(records))
        except Exception:
            logger.exception(
                "kafka_consumer.db_write_error",
                count=len(records),
            )

    async def _periodic_flush(self) -> None:
        """Flush the batch buffer periodically to bound write latency."""
        try:
            while self._running:
                await asyncio.sleep(self._flush_interval_seconds)
                await self._flush_batch()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("kafka_consumer.periodic_flush_error")


# ── Helpers ──────────────────────────────────────────────────


def _parse_timestamp(value: Any) -> datetime:
    """Best-effort ISO timestamp parsing with fallback to now()."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            # Handle trailing "Z"
            cleaned = value.rstrip("Z")
            return datetime.fromisoformat(cleaned)
        except (ValueError, TypeError):
            pass
    return datetime.utcnow()
