from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import structlog

from analytics.store import get_state_store
from config.settings import get_settings
from pipeline.detector import Detector, Detection
from pipeline.tracker import TrackManager
from pipeline.zone_mapper import ZoneMapper
from pipeline.event_emitter import EventEmitter
from pipeline.staff_classifier import StaffClassifier
from pipeline.ingestion import VideoSource
from streaming.memory_fallback import MemoryStreamBackend
from api.websocket import get_connection_manager
from api.schemas import EventType

logger = structlog.get_logger()


class PipelineRunner:
    """Spins up the full video pipeline for a single camera feed."""

    def __init__(
        self,
        camera_id: str,
        source: str,
        store_id: str = "STORE_BLR_002",
    ):
        self.camera_id = camera_id
        self.source = source
        self.store_id = store_id
        self.running = False
        self._task: asyncio.Task | None = None
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=2)

        self.video_source = VideoSource(camera_id=self.camera_id, source=self.source)
        self.detector = Detector()
        self.track_manager = TrackManager()
        self.zone_mapper = ZoneMapper()
        self.staff_classifier = StaffClassifier()

        # EventEmitter requires a StreamBackend — use in-memory for local dev
        self._stream_backend = MemoryStreamBackend()
        ws_manager = get_connection_manager()
        self.event_emitter = EventEmitter(
            backend=self._stream_backend,
            ws_broadcast=ws_manager.broadcast_event,
        )

        # Load zone polygons for drawing overlays
        self._zone_polygons = self._load_zone_config()

        # Track which tracks were active last frame for delta detection
        self._prev_active_track_ids: Set[int] = set()
        # Track previous zone assignments: track_id → zone_id
        self._prev_track_zones: Dict[int, Optional[str]] = {}

    async def start(self) -> None:
        """Start the pipeline."""
        if self.running:
            return

        self.running = True

        # Connect the stream backend
        await self._stream_backend.connect()

        await self.video_source.start()
        await self.event_emitter.start()

        self._task = asyncio.create_task(self._loop())
        logger.info("pipeline.started", camera_id=self.camera_id, source=self.source)

    async def stop(self) -> None:
        """Stop the pipeline."""
        if not self.running:
            return

        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await self.event_emitter.stop()
        await self.video_source.stop()
        await self._stream_backend.disconnect()
        logger.info("pipeline.stopped", camera_id=self.camera_id)

    async def _loop(self) -> None:
        """Pipeline pump loop."""
        try:
            while self.running:
                packet = await self.video_source.get_frame()
                if packet is None:
                    # EOF or stopped
                    self.running = False
                    break

                now_utc = datetime.utcnow()
                now_mono = time.monotonic()

                # 1. Detection & Tracking (CPU-bound — run off the event loop)
                loop = asyncio.get_running_loop()
                detections = await loop.run_in_executor(
                    None, self.detector.detect, packet.frame_data
                )

                # 2. Track management — emits person_detected / person_lost
                track_events = await loop.run_in_executor(
                    None, self.track_manager.process_frame,
                    self.camera_id, detections
                )

                # 3. Zone transitions, dwell updates & alerts
                zone_events = self.zone_mapper.process_detections(
                    self.camera_id, detections
                )

                # ── Spec event generation ─────────────────────
                # Determine current active track IDs
                current_track_ids: Set[int] = {det.track_id for det in detections}

                # Update staff classifier for each detection
                for det in detections:
                    zone_id = self.zone_mapper.get_track_zone(det.track_id)
                    self.staff_classifier.update(
                        track_id=det.track_id,
                        zone_id=zone_id,
                        timestamp=now_mono,
                    )

                # Classify all active tracks
                for det in detections:
                    zone_id = self.zone_mapper.get_track_zone(det.track_id)
                    zones_visited: List[str] = []
                    session_duration = 0.0
                    track_state = get_state_store().get_track(det.track_id)
                    if track_state is not None:
                        session_duration = (
                            track_state.last_seen - track_state.first_seen
                        ).total_seconds()
                        zones_visited = [
                            zh["zone_id"] for zh in track_state.zone_history
                        ]
                    self.staff_classifier.classify(
                        track_id=det.track_id,
                        zones_visited=zones_visited,
                        session_duration_s=session_duration,
                    )

                staff_flags = self.staff_classifier.get_staff_flags()

                # Emit spec events for internal events
                for ev in track_events:
                    track_id = ev.track_id
                    if track_id is None:
                        continue

                    visitor_id = self.track_manager.get_visitor_id(track_id)
                    is_staff = staff_flags.get(track_id, False)
                    confidence = ev.metadata.get("confidence", 0.0)

                    if ev.event_type == EventType.PERSON_DETECTED:
                        # Determine ENTRY vs REENTRY
                        is_reentry = ev.metadata.get("is_reentry", False)
                        spec_type = "REENTRY" if is_reentry else "ENTRY"

                        self.track_manager.increment_session_seq(track_id)
                        spec_event = self.event_emitter.build_spec_event(
                            event_type=spec_type,
                            store_id=self.store_id,
                            camera_id=self.camera_id,
                            visitor_id=visitor_id,
                            timestamp=now_utc,
                            confidence=confidence,
                            is_staff=is_staff,
                            session_seq=self.track_manager.get_session_seq(track_id),
                        )
                        await self.event_emitter.publish_spec_event(spec_event)

                    elif ev.event_type == EventType.PERSON_LOST:
                        self.track_manager.increment_session_seq(track_id)
                        duration_s = ev.metadata.get("duration_seconds", 0.0)
                        spec_event = self.event_emitter.build_spec_event(
                            event_type="EXIT",
                            store_id=self.store_id,
                            camera_id=self.camera_id,
                            visitor_id=visitor_id,
                            timestamp=now_utc,
                            confidence=confidence,
                            is_staff=is_staff,
                            dwell_ms=int(duration_s * 1000),
                            session_seq=self.track_manager.get_session_seq(track_id),
                        )
                        await self.event_emitter.publish_spec_event(spec_event)
                        # Clean up staff classifier state for lost tracks
                        self.staff_classifier.remove_track(track_id)

                for ev in zone_events:
                    track_id = ev.track_id
                    if track_id is None:
                        continue

                    visitor_id = self.track_manager.get_visitor_id(track_id)
                    is_staff = staff_flags.get(track_id, False)
                    zone_id = ev.zone_id

                    if ev.event_type == EventType.ZONE_ENTER:
                        self.track_manager.increment_session_seq(track_id)
                        queue_depth = self.zone_mapper.billing_queue_depth
                        spec_event = self.event_emitter.build_spec_event(
                            event_type="ZONE_ENTER",
                            store_id=self.store_id,
                            camera_id=self.camera_id,
                            visitor_id=visitor_id,
                            timestamp=now_utc,
                            is_staff=is_staff,
                            zone_id=zone_id,
                            queue_depth=queue_depth,
                            session_seq=self.track_manager.get_session_seq(track_id),
                        )
                        await self.event_emitter.publish_spec_event(spec_event)

                    elif ev.event_type == EventType.ZONE_EXIT:
                        self.track_manager.increment_session_seq(track_id)
                        dwell_s = ev.metadata.get("dwell_seconds", 0.0)
                        spec_event = self.event_emitter.build_spec_event(
                            event_type="ZONE_EXIT",
                            store_id=self.store_id,
                            camera_id=self.camera_id,
                            visitor_id=visitor_id,
                            timestamp=now_utc,
                            is_staff=is_staff,
                            zone_id=zone_id,
                            dwell_ms=int(dwell_s * 1000),
                            session_seq=self.track_manager.get_session_seq(track_id),
                        )
                        await self.event_emitter.publish_spec_event(spec_event)

                    elif ev.event_type == EventType.DWELL_TIME_UPDATE:
                        # Check for ZONE_DWELL spec events
                        if ev.metadata.get("spec_event_type") == "ZONE_DWELL":
                            self.track_manager.increment_session_seq(track_id)
                            dwell_ms = ev.metadata.get("dwell_ms", 0)
                            spec_event = self.event_emitter.build_spec_event(
                                event_type="ZONE_DWELL",
                                store_id=self.store_id,
                                camera_id=self.camera_id,
                                visitor_id=visitor_id,
                                timestamp=now_utc,
                                is_staff=is_staff,
                                zone_id=zone_id,
                                dwell_ms=dwell_ms,
                                queue_depth=self.zone_mapper.billing_queue_depth,
                                session_seq=self.track_manager.get_session_seq(track_id),
                            )
                            await self.event_emitter.publish_spec_event(spec_event)

                # 4. Annotate frame and push to stream queue
                annotated = self._annotate_frame(packet.frame_data, detections)
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await self._frame_queue.put(annotated)

                # 5. Publish all events
                all_events = track_events + zone_events
                if all_events:
                    await self.event_emitter.publish_many(all_events)

                # 6. Update frame counter
                store = get_state_store()
                store.increment_frames_processed()
                store.increment_camera_frames(self.camera_id)

                # Update previous-frame state for next iteration
                self._prev_active_track_ids = current_track_ids
                for det in detections:
                    self._prev_track_zones[det.track_id] = (
                        self.zone_mapper.get_track_zone(det.track_id)
                    )

                # Yield slightly to avoid blocking other tasks entirely
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("pipeline.fatal_error", camera_id=self.camera_id)
        finally:
            self.running = False
            store = get_state_store()
            store.update_camera_status(self.camera_id, "stopped")
            # Notify frontend that the pipeline has finished
            try:
                ws_manager = get_connection_manager()
                await ws_manager.broadcast_event({
                    "event_type": "pipeline_stopped",
                    "camera_id": self.camera_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
            except Exception:
                pass
            logger.info("pipeline.finished", camera_id=self.camera_id)

    # ── Frame streaming helpers ──────────────────────────

    async def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent annotated frame, or *None* after 1 s."""
        try:
            return await asyncio.wait_for(self._frame_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    # ── Zone config loader ───────────────────────────────

    @staticmethod
    def _load_zone_config() -> List[Dict]:
        """Read ``config/zones.json`` and return a list of zone dicts."""
        zones_path = Path(__file__).resolve().parent.parent / "config" / "zones.json"
        if not zones_path.exists():
            logger.warning("pipeline.zones_config_missing", path=str(zones_path))
            return []
        try:
            data = json.loads(zones_path.read_text(encoding="utf-8"))
            zones: List[Dict] = []
            for z in data.get("zones", []):
                hex_color = z.get("color", "#00FF00").lstrip("#")
                b, g, r = (
                    int(hex_color[4:6], 16),
                    int(hex_color[2:4], 16),
                    int(hex_color[0:2], 16),
                )
                zones.append(
                    {
                        "id": z["id"],
                        "name": z["name"],
                        "polygon": np.array(z["polygon"], dtype=np.int32),
                        "color": (b, g, r),
                    }
                )
            return zones
        except Exception:
            logger.exception("pipeline.zones_config_load_error")
            return []

    # ── Frame annotation ─────────────────────────────────

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List,
    ) -> np.ndarray:
        """Draw zone overlays, bounding boxes, and a stats HUD on *frame*."""
        annotated = frame.copy()
        overlay = annotated.copy()

        # ── Zone overlays (semi-transparent fill + border + label) ───
        for zone in self._zone_polygons:
            pts = zone["polygon"]
            color: Tuple[int, ...] = zone["color"]
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
            # Zone name label
            label_pos = (int(pts[:, 0].mean()), int(pts[:, 1].min()) - 8)
            cv2.putText(
                annotated,
                zone["name"],
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # Blend the filled overlay at 25% opacity
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        # ── Bounding boxes ───────────────────────────────
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{det.track_id} {det.confidence:.0%}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # ── Stats HUD (top-left) ─────────────────────────
        store = get_state_store()
        stats_lines = [
            f"Visitors: {len(detections)}",
            f"Frames: {store.total_frames_processed}",
        ]
        y_offset = 20
        max_w = 200
        cv2.rectangle(annotated, (5, 5), (5 + max_w, 5 + len(stats_lines) * 25 + 10), (0, 0, 0), -1)
        for line in stats_lines:
            cv2.putText(
                annotated,
                line,
                (10, y_offset + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += 25

        return annotated

