"""
Store Intelligence System — MJPEG Video Stream Router.

Provides a live annotated video stream for each camera via
``GET /cameras/{camera_id}/stream``.  The response uses the MJPEG
multipart format so it can be consumed directly by ``<img>`` tags.
"""

from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np
import structlog
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/cameras", tags=["Video Stream"])


def _make_placeholder(text: str, sub: str = "") -> bytes:
    """Create a placeholder JPEG frame with centered text."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)  # near-black background
    cv2.putText(frame, text, (180, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (170, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def _encode_mjpeg_frame(jpg_bytes: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"


async def _generate_mjpeg(camera_id: str):
    """Generate MJPEG frames from the pipeline runner."""
    from api.routers.cameras import _active_runners

    # Phase 1: Wait up to 15s for the pipeline to become ready.
    # This avoids flashing "No Feed" during model loading / file upload.
    start = time.monotonic()
    loading_frame = _make_placeholder("Loading...", f"Starting pipeline for {camera_id}")
    while time.monotonic() - start < 15:
        runner = _active_runners.get(camera_id)
        if runner is not None and runner.running:
            break
        yield _encode_mjpeg_frame(loading_frame)
        await asyncio.sleep(0.5)

    # Phase 2: Stream annotated frames
    no_feed_frame = _make_placeholder("No Feed", "Pipeline is not running")
    while True:
        runner = _active_runners.get(camera_id)

        # Pipeline not running → show "No Feed" and stop the stream
        if runner is None or not runner.running:
            yield _encode_mjpeg_frame(no_feed_frame)
            # Don't keep looping forever after pipeline ends
            await asyncio.sleep(0.5)
            # Check again — if still not running, end the stream
            runner = _active_runners.get(camera_id)
            if runner is None or not runner.running:
                return

        # Get next annotated frame
        frame = await runner.get_latest_frame()
        if frame is None:
            await asyncio.sleep(0.03)
            continue

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield _encode_mjpeg_frame(buf.tobytes())


@router.get("/{camera_id}/stream")
async def video_stream(camera_id: str):
    """MJPEG video stream with bounding box annotations."""
    return StreamingResponse(
        _generate_mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
