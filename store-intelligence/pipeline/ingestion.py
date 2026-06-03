"""
Store Intelligence System — Video Ingestion Pipeline.

Captures frames from video files (.mp4/.avi) or RTSP streams,
applies preprocessing (resize, normalize, optional CLAHE), and pushes
FramePacket objects into an asyncio buffer queue for downstream consumers.

Each source is tagged with a ``camera_id`` for multi-camera support.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import Thread
from typing import Optional

import cv2
import numpy as np
import structlog

from config.settings import get_settings

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Immutable container for a single preprocessed video frame."""

    camera_id: str
    frame_number: int
    timestamp: datetime
    frame_data: np.ndarray  # BGR uint8, resized to (H, W, 3)
    original_width: int = 0
    original_height: int = 0


# ═══════════════════════════════════════════════════════════
# Video Source
# ═══════════════════════════════════════════════════════════


class VideoSource:
    """
    Async-friendly video capture source.

    Usage::

        source = VideoSource("cam01", "rtsp://192.168.1.10/stream")
        await source.start()
        while True:
            packet = await source.get_frame()
            if packet is None:
                break
            # process packet …
        await source.stop()

    The capture loop runs in a background daemon thread so that
    blocking ``cv2.VideoCapture.read()`` calls never stall the
    asyncio event loop.  Frames are deposited into an
    ``asyncio.Queue`` and retrieved with ``get_frame()``.
    """

    def __init__(
        self,
        camera_id: str,
        source: str | int,
        *,
        fps: Optional[int] = None,
        buffer_size: Optional[int] = None,
    ) -> None:
        self._settings = get_settings()
        self.camera_id = camera_id
        self.source = source
        self.fps = fps or self._settings.VIDEO_FPS
        self.buffer_size = buffer_size or self._settings.FRAME_BUFFER_SIZE

        # Internal state
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: Optional[asyncio.Queue[Optional[FramePacket]]] = None
        self._running = False
        self._thread: Optional[Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._frame_count = 0
        self._clahe: Optional[cv2.CLAHE] = None

        # Pre-build the CLAHE object once (it is reusable and thread-safe).
        if self._settings.ENABLE_CLAHE:
            self._clahe = cv2.createCLAHE(
                clipLimit=self._settings.CLAHE_CLIP_LIMIT,
                tileGridSize=(
                    self._settings.CLAHE_GRID_SIZE,
                    self._settings.CLAHE_GRID_SIZE,
                ),
            )

        self._log = logger.bind(camera_id=camera_id, source=str(source))

    # ── Lifecycle ─────────────────────────────────────────

    async def start(self) -> None:
        """Open the capture device and begin the background reader."""
        if self._running:
            self._log.warning("video_source.already_running")
            return

        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._log.error("video_source.open_failed")
            raise RuntimeError(
                f"Cannot open video source '{self.source}' for camera '{self.camera_id}'"
            )

        self._queue = asyncio.Queue(maxsize=self.buffer_size)
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._frame_count = 0

        self._thread = Thread(
            target=self._capture_loop,
            name=f"videosrc-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        self._log.info(
            "video_source.started",
            fps=self.fps,
            buffer_size=self.buffer_size,
            clahe=self._settings.ENABLE_CLAHE,
        )

    async def stop(self) -> None:
        """Signal the reader to stop and release resources."""
        if not self._running:
            return
        self._running = False

        # Unblock the producer if the queue is full.
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)  # sentinel
            except asyncio.QueueFull:
                pass

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._log.info("video_source.stopped", frames_captured=self._frame_count)

    async def get_frame(self, timeout: float = 2.0) -> Optional[FramePacket]:
        """
        Await the next preprocessed frame from the buffer.

        Returns ``None`` when the source has been stopped or exhausted.
        """
        if self._queue is None:
            return None
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ── Internal: blocking capture loop (runs in thread) ──

    def _capture_loop(self) -> None:
        """
        Read frames at the configured FPS, preprocess, and push
        ``FramePacket``s into the asyncio queue.

        This method blocks and is expected to run inside a daemon thread.
        """
        assert self._cap is not None
        assert self._queue is not None
        assert self._loop is not None

        interval = 1.0 / self.fps
        source_fps = self._cap.get(cv2.CAP_PROP_FPS) or self.fps

        self._log.debug(
            "video_source.capture_loop_started",
            target_fps=self.fps,
            source_fps=source_fps,
        )

        # If the source FPS is higher than our target, we compute
        # how many source frames to skip between reads.
        skip_ratio = max(1, int(round(source_fps / self.fps))) if source_fps > self.fps else 1
        raw_frame_idx = 0

        while self._running:
            t0 = time.monotonic()

            try:
                ok, raw = self._cap.read()
            except Exception:
                self._log.exception("video_source.read_exception")
                continue

            if not ok:
                # End of file or broken stream
                self._log.info("video_source.stream_ended")
                break

            raw_frame_idx += 1
            if raw_frame_idx % skip_ratio != 0:
                continue

            # ── Preprocess ──
            try:
                frame = self._preprocess(raw)
            except Exception:
                self._log.exception(
                    "video_source.preprocess_error",
                    frame_number=self._frame_count,
                )
                continue

            self._frame_count += 1
            packet = FramePacket(
                camera_id=self.camera_id,
                frame_number=self._frame_count,
                timestamp=datetime.utcnow(),
                frame_data=frame,
                original_width=raw.shape[1],
                original_height=raw.shape[0],
            )

            # Deposit into the asyncio queue (thread-safe call).
            future = asyncio.run_coroutine_threadsafe(
                self._enqueue(packet), self._loop
            )
            try:
                future.result(timeout=2.0)
            except Exception:
                self._log.warning(
                    "video_source.enqueue_timeout",
                    frame_number=self._frame_count,
                )

            # Throttle to target FPS
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Push sentinel to signal consumers.
        try:
            asyncio.run_coroutine_threadsafe(
                self._enqueue(None), self._loop
            ).result(timeout=2.0)
        except Exception:
            pass

        self._running = False
        self._log.debug("video_source.capture_loop_exited")

    async def _enqueue(self, packet: Optional[FramePacket]) -> None:
        """Put a packet into the async queue, dropping oldest if full."""
        assert self._queue is not None
        if self._queue.full():
            try:
                self._queue.get_nowait()  # drop oldest frame
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(packet)

    # ── Preprocessing ─────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize to target dimensions and optionally apply CLAHE.

        Returns a BGR ``uint8`` ndarray of shape
        ``(VIDEO_FRAME_HEIGHT, VIDEO_FRAME_WIDTH, 3)``.
        """
        target_w = self._settings.VIDEO_FRAME_WIDTH
        target_h = self._settings.VIDEO_FRAME_HEIGHT

        resized = cv2.resize(
            frame,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )

        if self._clahe is not None:
            # Apply CLAHE on the L-channel of LAB colour space.
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            l_chan = self._clahe.apply(l_chan)
            lab = cv2.merge((l_chan, a_chan, b_chan))
            resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return resized
