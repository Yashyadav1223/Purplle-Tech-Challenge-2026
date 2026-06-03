"""
Store Intelligence System — Redis Streams Backend.

Uses Redis Streams (XADD / XREAD / XREVRANGE) as a lightweight,
persistent streaming backend. Falls back gracefully when Redis
is unavailable.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

import structlog

from config.settings import get_settings
from streaming.base import StreamBackend

logger = structlog.get_logger()


class RedisStreamBackend(StreamBackend):
    """
    Redis Streams-based event backend.

    Uses ``redis.asyncio`` for non-blocking I/O. Each topic maps
    directly to a Redis Stream key.
    """

    def __init__(self) -> None:
        self._client: Optional[Any] = None  # redis.asyncio.Redis
        self._connected: bool = False
        self._tasks: List[asyncio.Task] = []
        self._shutting_down: bool = False

    # ── StreamBackend Interface ──────────────────────────────

    async def connect(self) -> None:
        """Create an async Redis connection pool."""
        try:
            import redis.asyncio as aioredis
        except ImportError:
            logger.error(
                "redis_backend.import_error",
                hint="Install redis: pip install redis>=5.0.0",
            )
            raise

        settings = get_settings()
        try:
            self._client = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True,
            )
            # Verify connectivity
            await self._client.ping()
            self._connected = True
            logger.info(
                "redis_backend.connected",
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
            )
        except Exception:
            self._connected = False
            logger.exception("redis_backend.connect_failed")
            raise

    async def disconnect(self) -> None:
        """Cancel subscriber tasks and close the Redis connection."""
        self._shutting_down = True
        self._connected = False

        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                logger.exception("redis_backend.close_error")
            self._client = None

        logger.info("redis_backend.disconnected")

    async def publish(self, topic: str, event: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Publish an event to a Redis Stream via XADD.

        The event dict is serialized to JSON and stored under the ``data`` field.
        An optional ``key`` field is added for consumer-side partitioning hints.
        """
        if not self._connected or self._client is None:
            logger.warning("redis_backend.publish_failed", reason="not_connected")
            return False

        settings = get_settings()
        try:
            fields = {"data": json.dumps(event, default=str)}
            if key:
                fields["key"] = key

            await self._client.xadd(
                topic,
                fields,
                maxlen=settings.REDIS_STREAM_MAXLEN,
                approximate=True,
            )
            return True

        except Exception:
            logger.exception("redis_backend.publish_error", topic=topic)
            self._connected = False
            return False

    async def subscribe(self, topics: List[str], handler: Callable) -> None:
        """
        Subscribe to Redis Streams. Launches a background task that
        calls XREAD in a blocking loop and dispatches events to *handler*.
        """
        if not self._connected or self._client is None:
            logger.warning("redis_backend.subscribe_failed", reason="not_connected")
            return

        task = asyncio.create_task(
            self._read_loop(topics, handler),
            name=f"redis-sub-{'|'.join(topics)}",
        )
        self._tasks.append(task)
        logger.info("redis_backend.subscribed", topics=topics)

    async def get_recent(self, topic: str, count: int = 50) -> List[Dict[str, Any]]:
        """Retrieve the most recent *count* events from a Redis Stream (newest first)."""
        if not self._connected or self._client is None:
            return []

        try:
            # XREVRANGE returns entries newest → oldest
            entries = await self._client.xrevrange(topic, count=count)
            results: List[Dict[str, Any]] = []
            for _entry_id, fields in entries:
                data_str = fields.get("data")
                if data_str:
                    try:
                        results.append(json.loads(data_str))
                    except json.JSONDecodeError:
                        logger.warning("redis_backend.decode_error", topic=topic)
            return results

        except Exception:
            logger.exception("redis_backend.get_recent_error", topic=topic)
            return []

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Internal ─────────────────────────────────────────────

    async def _read_loop(self, topics: List[str], handler: Callable) -> None:
        """
        Blocking XREAD loop that dispatches incoming stream entries
        to the provided handler callback.
        """
        # Start reading from the latest entry already in each stream
        last_ids: Dict[str, str] = {t: "$" for t in topics}

        logger.debug("redis_backend.read_loop_started", topics=topics)
        try:
            while self._connected and not self._shutting_down:
                try:
                    if self._client is None:
                        break

                    streams = {t: last_ids[t] for t in topics}
                    # block for up to 1 second before rechecking shutdown flag
                    response = await self._client.xread(streams, count=100, block=1000)

                    if response is None:
                        continue

                    for stream_name, entries in response:
                        # stream_name may be bytes or str depending on decode_responses
                        topic_str = (
                            stream_name.decode()
                            if isinstance(stream_name, bytes)
                            else stream_name
                        )
                        for entry_id, fields in entries:
                            entry_id_str = (
                                entry_id.decode()
                                if isinstance(entry_id, bytes)
                                else entry_id
                            )
                            last_ids[topic_str] = entry_id_str

                            data_str = fields.get("data")
                            if not data_str:
                                continue

                            try:
                                event = json.loads(data_str)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "redis_backend.decode_error",
                                    topic=topic_str,
                                    entry_id=entry_id_str,
                                )
                                continue

                            try:
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(topic_str, event)
                                else:
                                    handler(topic_str, event)
                            except Exception:
                                logger.exception(
                                    "redis_backend.handler_error",
                                    topic=topic_str,
                                    event_type=event.get("event_type"),
                                )

                except asyncio.CancelledError:
                    raise
                except Exception:
                    if not self._shutting_down:
                        logger.exception("redis_backend.read_loop_error")
                        await asyncio.sleep(2.0)  # reconnect backoff

        except asyncio.CancelledError:
            logger.debug("redis_backend.read_loop_cancelled")
        except Exception:
            logger.exception("redis_backend.read_loop_fatal")
