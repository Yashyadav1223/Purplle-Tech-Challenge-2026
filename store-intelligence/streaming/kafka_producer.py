"""
Store Intelligence System — Kafka Stream Backend.

Production-grade Kafka backend using confluent_kafka.
Features:
- Delivery callbacks with structured logging
- Exponential-backoff retry on transient publish failures
- In-memory deque cache per topic for fast ``get_recent()``
- Background consumer thread for ``subscribe()``
- Graceful flush on disconnect / SIGTERM
"""

from __future__ import annotations

import asyncio
import json
import signal
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional

import structlog

from config.settings import get_settings
from streaming.base import StreamBackend

logger = structlog.get_logger()

# Maximum events cached per topic for ``get_recent()``
_CACHE_MAXLEN = 5000


class KafkaStreamBackend(StreamBackend):
    """
    Kafka-backed streaming using ``confluent_kafka.Producer`` and ``Consumer``.

    Thread-safety notes:
    - The Producer is thread-safe; publish() can be called from any coroutine.
    - Each subscribe() call spins up a dedicated daemon thread with its own
      Consumer instance; the handler is dispatched back into the asyncio
      event loop.
    """

    def __init__(self) -> None:
        self._producer: Optional[Any] = None  # confluent_kafka.Producer
        self._connected: bool = False
        self._shutting_down: bool = False

        # In-memory cache: topic -> deque of event dicts (newest at right)
        self._cache: Dict[str, deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=_CACHE_MAXLEN)
        )

        # Consumer threads & their stop-events
        self._consumer_threads: List[threading.Thread] = []
        self._consumer_stop_events: List[threading.Event] = []

        # Event loop reference (set on connect)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── StreamBackend Interface ──────────────────────────────

    async def connect(self) -> None:
        """
        Create the confluent_kafka Producer and register SIGTERM handler
        for graceful flush.
        """
        try:
            from confluent_kafka import Producer  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "kafka_backend.import_error",
                hint="Install confluent-kafka: pip install confluent-kafka>=2.3.0",
            )
            raise

        settings = get_settings()
        conf = {
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "client.id": "store-intelligence-producer",
            "acks": "all",
            "retries": 0,  # we handle retries ourselves
            "linger.ms": 5,
            "compression.type": "snappy",
            "queue.buffering.max.messages": 100000,
            "queue.buffering.max.kbytes": 1048576,  # 1 GB
        }

        try:
            self._producer = Producer(conf)
            self._connected = True
            self._loop = asyncio.get_running_loop()

            # Register SIGTERM handler for graceful flush (best-effort on Windows)
            self._register_signal_handlers()

            logger.info(
                "kafka_backend.connected",
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            )
        except Exception:
            self._connected = False
            logger.exception("kafka_backend.connect_failed")
            raise

    async def disconnect(self) -> None:
        """Flush pending messages and tear down producer + consumer threads."""
        self._shutting_down = True
        self._connected = False

        # Stop consumer threads
        for stop_event in self._consumer_stop_events:
            stop_event.set()
        for thread in self._consumer_threads:
            thread.join(timeout=10)
        self._consumer_threads.clear()
        self._consumer_stop_events.clear()

        # Flush the producer
        if self._producer is not None:
            try:
                remaining = self._producer.flush(timeout=10)
                if remaining > 0:
                    logger.warning(
                        "kafka_backend.flush_incomplete",
                        remaining=remaining,
                    )
                else:
                    logger.info("kafka_backend.flush_complete")
            except Exception:
                logger.exception("kafka_backend.flush_error")
            self._producer = None

        logger.info("kafka_backend.disconnected")

    async def publish(
        self, topic: str, event: Dict[str, Any], key: Optional[str] = None
    ) -> bool:
        """
        Publish an event to Kafka with retry logic.

        Retries up to ``KAFKA_MAX_RETRIES`` times with exponential backoff
        starting at ``KAFKA_RETRY_BACKOFF_MS``.
        """
        if not self._connected or self._producer is None:
            logger.warning("kafka_backend.publish_failed", reason="not_connected")
            return False

        settings = get_settings()
        max_retries = settings.KAFKA_MAX_RETRIES
        backoff_ms = settings.KAFKA_RETRY_BACKOFF_MS
        value = json.dumps(event, default=str).encode("utf-8")
        encoded_key = key.encode("utf-8") if key else None

        for attempt in range(1, max_retries + 1):
            try:
                self._producer.produce(
                    topic=topic,
                    value=value,
                    key=encoded_key,
                    callback=self._delivery_callback,
                )
                # Trigger delivery callbacks without blocking
                self._producer.poll(0)

                # Cache locally for fast get_recent()
                self._cache[topic].append(event)

                return True

            except BufferError:
                # Local queue full — poll to drain, then retry
                logger.warning(
                    "kafka_backend.buffer_full",
                    topic=topic,
                    attempt=attempt,
                )
                self._producer.poll(1)
            except Exception as exc:
                logger.warning(
                    "kafka_backend.produce_error",
                    topic=topic,
                    attempt=attempt,
                    error=str(exc),
                )

            if attempt < max_retries:
                sleep_secs = (backoff_ms / 1000) * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_secs)

        logger.error(
            "kafka_backend.publish_exhausted",
            topic=topic,
            max_retries=max_retries,
        )
        return False

    async def subscribe(self, topics: List[str], handler: Callable) -> None:
        """
        Subscribe to Kafka topics. A background daemon thread runs the
        confluent_kafka Consumer; incoming messages are dispatched to the
        asyncio event loop via ``loop.call_soon_threadsafe``.
        """
        if not self._connected:
            logger.warning("kafka_backend.subscribe_failed", reason="not_connected")
            return

        loop = asyncio.get_running_loop()
        stop_event = threading.Event()
        self._consumer_stop_events.append(stop_event)

        thread = threading.Thread(
            target=self._consumer_thread_target,
            args=(topics, handler, loop, stop_event),
            daemon=True,
            name=f"kafka-consumer-{'|'.join(topics)}",
        )
        thread.start()
        self._consumer_threads.append(thread)
        logger.info("kafka_backend.subscribed", topics=topics)

    async def get_recent(self, topic: str, count: int = 50) -> List[Dict[str, Any]]:
        """
        Return the most recent *count* events from the in-memory cache.

        This avoids a costly Kafka consumer seek; the cache is populated
        by both ``publish()`` and the subscriber thread.
        """
        cache = self._cache.get(topic)
        if not cache:
            return []
        items = list(cache)
        items.reverse()
        return items[:count]

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Internal ─────────────────────────────────────────────

    @staticmethod
    def _delivery_callback(err: Any, msg: Any) -> None:
        """confluent_kafka delivery report callback."""
        if err is not None:
            logger.error(
                "kafka_backend.delivery_failed",
                topic=msg.topic() if msg else "unknown",
                error=str(err),
            )
        else:
            logger.debug(
                "kafka_backend.delivered",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
            )

    def _consumer_thread_target(
        self,
        topics: List[str],
        handler: Callable,
        loop: asyncio.AbstractEventLoop,
        stop_event: threading.Event,
    ) -> None:
        """
        Target function for the background consumer thread.
        Polls Kafka, deserializes JSON, and dispatches to the async handler
        via the event loop.
        """
        try:
            from confluent_kafka import Consumer, KafkaError  # type: ignore[import-untyped]
        except ImportError:
            logger.error("kafka_backend.consumer_import_error")
            return

        settings = get_settings()
        conf = {
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "store-intelligence-consumer",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
            "session.timeout.ms": 30000,
        }

        consumer: Optional[Any] = None
        try:
            consumer = Consumer(conf)
            consumer.subscribe(topics)
            logger.info("kafka_backend.consumer_started", topics=topics)

            while not stop_event.is_set():
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    error = msg.error()
                    if error.code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(
                        "kafka_backend.consumer_error",
                        error=str(error),
                    )
                    continue

                try:
                    value = msg.value()
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    event: Dict[str, Any] = json.loads(value)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning(
                        "kafka_backend.deserialize_error",
                        error=str(exc),
                        topic=msg.topic(),
                    )
                    continue

                topic = msg.topic()

                # Update local cache
                self._cache[topic].append(event)

                # Dispatch to async handler on the event loop
                if asyncio.iscoroutinefunction(handler):
                    asyncio.run_coroutine_threadsafe(handler(topic, event), loop)
                else:
                    loop.call_soon_threadsafe(handler, topic, event)

        except Exception:
            logger.exception("kafka_backend.consumer_thread_fatal")
        finally:
            if consumer is not None:
                try:
                    consumer.close()
                except Exception:
                    logger.exception("kafka_backend.consumer_close_error")
            logger.info("kafka_backend.consumer_stopped")

    def _register_signal_handlers(self) -> None:
        """
        Register SIGTERM / SIGINT handlers for graceful producer flush.
        On Windows, only SIGINT is supported.
        """

        def _flush_on_signal(signum: int, _frame: Any) -> None:
            logger.info("kafka_backend.signal_received", signal=signum)
            if self._producer is not None:
                try:
                    self._producer.flush(timeout=5)
                except Exception:
                    pass

        try:
            signal.signal(signal.SIGTERM, _flush_on_signal)
        except (OSError, ValueError):
            # SIGTERM not supported on Windows in non-main threads
            pass

        try:
            signal.signal(signal.SIGINT, _flush_on_signal)
        except (OSError, ValueError):
            pass
