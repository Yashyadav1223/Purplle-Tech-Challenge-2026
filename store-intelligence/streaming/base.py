"""
Store Intelligence System — Abstract Stream Backend.

Defines the interface that Kafka, Redis, and in-memory backends implement.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional


class StreamBackend(abc.ABC):
    """Abstract base class for event streaming backends."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Initialize the connection to the streaming backend."""
        ...

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        ...

    @abc.abstractmethod
    async def publish(self, topic: str, event: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Publish an event to a topic/stream.

        Args:
            topic: Topic name (e.g., 'store.zone_events')
            event: Serializable event dict
            key: Optional partition key (e.g., camera_id)

        Returns:
            True if published successfully, False otherwise.
        """
        ...

    @abc.abstractmethod
    async def subscribe(self, topics: List[str], handler: Callable) -> None:
        """
        Subscribe to one or more topics and invoke handler for each message.

        Args:
            topics: List of topic names
            handler: Async callable(topic, event_dict)
        """
        ...

    @abc.abstractmethod
    async def get_recent(self, topic: str, count: int = 50) -> List[Dict[str, Any]]:
        """
        Get the most recent N events from a topic.

        Args:
            topic: Topic name
            count: Number of events to retrieve

        Returns:
            List of event dicts, newest first.
        """
        ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Whether the backend is currently connected."""
        ...
