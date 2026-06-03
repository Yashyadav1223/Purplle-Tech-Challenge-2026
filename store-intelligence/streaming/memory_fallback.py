"""
Store Intelligence System — In-Memory Stream Backend.

Pure asyncio-based stream backend for local development and testing.
No external dependencies required. Uses deques for storage and
asyncio.Queues for fan-out pub/sub.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional

import structlog

from streaming.base import StreamBackend

logger = structlog.get_logger()


class MemoryStreamBackend(StreamBackend):
    """
    In-memory streaming backend backed by asyncio primitives.

    Architecture:
    - Each topic has a bounded deque for message history (newest-first retrieval).
    - Each topic has a list of asyncio.Queues — one per active subscriber.
    - publish() appends to the deque and fans out to all subscriber queues.
    - subscribe() creates a dedicated queue and consumes from it in a loop.
    """

    def __init__(self, max_history: int = 5000) -> None:
        self._connected: bool = False
        self._max_history = max_history

        # topic -> deque of event dicts (newest appended at right)
        self._store: Dict[str, deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._max_history)
        )
        # topic -> list of subscriber queues
        self._subscribers: Dict[str, List[asyncio.Queue[Dict[str, Any]]]] = defaultdict(list)
        # background tasks for subscriber loops
        self._tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()

    # ── StreamBackend Interface ──────────────────────────────

    async def connect(self) -> None:
        """Initialize the in-memory backend (no-op aside from state flag)."""
        self._connected = True
        logger.info("memory_backend.connected")

    async def disconnect(self) -> None:
        """Cancel all subscriber tasks and clear state."""
        self._connected = False

        # Cancel running subscriber tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Drain subscriber queues
        async with self._lock:
            self._subscribers.clear()

        logger.info("memory_backend.disconnected")

    async def publish(self, topic: str, event: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Publish an event to an in-memory topic.

        The event is stored in the topic deque and fanned out to all
        active subscriber queues.
        """
        if not self._connected:
            logger.warning("memory_backend.publish_failed", reason="not_connected")
            return False

        try:
            async with self._lock:
                self._store[topic].append(event)

                dead_queues: List[int] = []
                for idx, queue in enumerate(self._subscribers.get(topic, [])):
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        # Drop oldest to make room — backpressure strategy
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            queue.put_nowait(event)
                        except asyncio.QueueFull:
                            dead_queues.append(idx)

                # Remove any permanently broken queues
                if dead_queues and topic in self._subscribers:
                    for idx in reversed(dead_queues):
                        self._subscribers[topic].pop(idx)

            return True

        except Exception:
            logger.exception("memory_backend.publish_error", topic=topic)
            return False

    async def subscribe(self, topics: List[str], handler: Callable) -> None:
        """
        Subscribe to topics and invoke ``handler(topic, event_dict)``
        for each message. Runs a background task per topic.
        """
        if not self._connected:
            logger.warning("memory_backend.subscribe_failed", reason="not_connected")
            return

        for topic in topics:
            queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)

            async with self._lock:
                self._subscribers[topic].append(queue)

            task = asyncio.create_task(
                self._consume_loop(topic, queue, handler),
                name=f"memory-sub-{topic}",
            )
            self._tasks.append(task)
            logger.info("memory_backend.subscribed", topic=topic)

    async def get_recent(self, topic: str, count: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent *count* events from the topic (newest first)."""
        async with self._lock:
            store = self._store.get(topic)
            if not store:
                return []
            # deque stores oldest→newest; we want newest first
            items = list(store)
            items.reverse()
            return items[:count]

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Internal ─────────────────────────────────────────────

    async def _consume_loop(
        self,
        topic: str,
        queue: asyncio.Queue[Dict[str, Any]],
        handler: Callable,
    ) -> None:
        """Consume events from a subscriber queue and dispatch to handler."""
        logger.debug("memory_backend.consume_loop_started", topic=topic)
        try:
            while self._connected:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(topic, event)
                    else:
                        handler(topic, event)
                except Exception:
                    logger.exception(
                        "memory_backend.handler_error",
                        topic=topic,
                        event_type=event.get("event_type"),
                    )

        except asyncio.CancelledError:
            logger.debug("memory_backend.consume_loop_cancelled", topic=topic)
        except Exception:
            logger.exception("memory_backend.consume_loop_fatal", topic=topic)
