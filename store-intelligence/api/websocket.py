"""
Store Intelligence System — WebSocket Connection Manager.

Manages active WebSocket connections with optional per-client event filtering.
On connect, sends the last 50 events as a backfill. Subsequent events are
broadcast only to clients whose filters match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from analytics.store import get_state_store

logger = structlog.get_logger()


@dataclass
class ClientConnection:
    """Tracks a single WebSocket client and its filter preferences."""

    websocket: WebSocket
    event_types: Optional[Set[str]] = field(default=None)
    zone: Optional[str] = None

    def matches(self, event: Dict[str, Any]) -> bool:
        """Return True if the event passes this client's filters."""
        if self.event_types:
            event_type = event.get("event_type", "")
            if event_type not in self.event_types:
                return False
        if self.zone:
            if event.get("zone_id") != self.zone:
                return False
        return True


class ConnectionManager:
    """
    Manages WebSocket connections for real-time event streaming.

    Supports:
    - Per-client filters (event types, zone)
    - Backfill of recent events on connect
    - Filtered broadcast to matching clients
    """

    def __init__(self) -> None:
        self._connections: List[ClientConnection] = []

    @property
    def active_count(self) -> int:
        return len(self._connections)

    async def connect(
        self,
        websocket: WebSocket,
        types: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> ClientConnection:
        """
        Accept a WebSocket connection and send recent event backfill.

        Args:
            websocket: The incoming WebSocket connection.
            types: Comma-separated event type filter (e.g. "zone_enter,zone_exit").
            zone: Zone ID filter — only events from this zone will be sent.
        """
        await websocket.accept()

        # Parse filter params
        event_types: Optional[Set[str]] = None
        if types:
            event_types = {t.strip() for t in types.split(",") if t.strip()}

        client = ClientConnection(
            websocket=websocket,
            event_types=event_types,
            zone=zone,
        )
        self._connections.append(client)

        logger.info(
            "websocket.connected",
            active_connections=self.active_count,
            filters={"types": list(event_types) if event_types else None, "zone": zone},
        )

        # Send backfill of last 50 events (filtered for this client)
        try:
            store = get_state_store()
            recent = store.get_recent_events(count=50)
            backfill = [e for e in recent if client.matches(e)]
            if backfill:
                await websocket.send_json({
                    "type": "backfill",
                    "events": backfill,
                    "count": len(backfill),
                })
        except Exception:
            logger.exception("websocket.backfill_failed")

        return client

    def disconnect(self, client: ClientConnection) -> None:
        """Remove a client from the active connection list."""
        try:
            self._connections.remove(client)
        except ValueError:
            pass
        logger.info("websocket.disconnected", active_connections=self.active_count)

    async def broadcast_event(self, event: Dict[str, Any]) -> None:
        """
        Broadcast an event to all connected clients whose filters match.

        Disconnected or errored clients are silently removed.
        """
        stale: List[ClientConnection] = []

        for client in self._connections:
            if not client.matches(event):
                continue
            try:
                await client.websocket.send_json({
                    "type": "event",
                    "event": event,
                })
            except (WebSocketDisconnect, RuntimeError, Exception):
                stale.append(client)

        # Clean up stale connections
        for client in stale:
            self.disconnect(client)

    async def broadcast_raw(self, message: Dict[str, Any]) -> None:
        """Broadcast a raw JSON message to all clients (unfiltered)."""
        stale: List[ClientConnection] = []

        for client in self._connections:
            try:
                await client.websocket.send_json(message)
            except (WebSocketDisconnect, RuntimeError, Exception):
                stale.append(client)

        for client in stale:
            self.disconnect(client)


# ── Singleton ──────────────────────────────────────────────
_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get the global ConnectionManager singleton."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager
