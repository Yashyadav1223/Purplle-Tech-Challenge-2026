"""
Store Intelligence System — Camera Management Router.

Provides endpoints to list cameras and start/stop processing.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel
import asyncio
import os
import shutil
import tempfile
from analytics.store import get_state_store
from api.schemas import ApiResponse, CameraInfo, ErrorDetail, ResponseMeta
from config.settings import get_settings
from pipeline.runner import PipelineRunner

logger = structlog.get_logger()
router = APIRouter(prefix="/cameras", tags=["Cameras"])

class StartCameraRequest(BaseModel):
    source: str = "0" # Default to webcam or a video path

# Store active runners
_active_runners = {}

def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@router.get("", response_model=ApiResponse)
async def list_cameras(request: Request) -> ApiResponse:
    """List all registered cameras with their current status."""
    try:
        store = get_state_store()
        cameras = store.get_cameras()
        return ApiResponse(
            status="success",
            data=cameras,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("cameras.list_failed")
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="CAMERA_LIST_FAILED", message=str(exc)),
        )


@router.post("/{camera_id}/start", response_model=ApiResponse)
async def start_camera(camera_id: str, request: Request, file: UploadFile | None = None, source: str | None = Form(None)) -> ApiResponse:
    """
    Start processing for a camera.

    Registers the camera in the state store with ``active`` status
    and spins up the video pipeline runner.
    """
    try:
        store = get_state_store()

        if camera_id in _active_runners:
            return ApiResponse(
                status="error",
                meta=ResponseMeta(request_id=_get_request_id(request)),
                error=ErrorDetail(code="CAMERA_ALREADY_RUNNING", message="Camera is already running"),
            )

        video_source = "0"
        
        if file:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            video_source = temp_path
        elif source:
            video_source = source

        # Register new camera
        store.register_camera(camera_id=camera_id, source=video_source)
        store.update_camera_status(camera_id, "active")
        
        # Start pipeline
        runner = PipelineRunner(camera_id=camera_id, source=video_source)
        _active_runners[camera_id] = runner
        asyncio.create_task(runner.start())
        logger.info("camera.started", camera_id=camera_id, source=video_source)

        cameras = store.get_cameras()
        camera = next((c for c in cameras if c.get("camera_id") == camera_id), None)

        return ApiResponse(
            status="success",
            data=camera,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("cameras.start_failed", camera_id=camera_id)
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="CAMERA_START_FAILED", message=str(exc)),
        )


@router.post("/{camera_id}/stop", response_model=ApiResponse)
async def stop_camera(camera_id: str, request: Request) -> ApiResponse:
    """
    Stop processing for a camera.

    Sets the camera status to ``stopped`` in the state store.
    """
    try:
        store = get_state_store()
        existing = [c for c in store.get_cameras() if c.get("camera_id") == camera_id]
        if not existing:
            return ApiResponse(
                status="error",
                meta=ResponseMeta(request_id=_get_request_id(request)),
                error=ErrorDetail(
                    code="CAMERA_NOT_FOUND",
                    message=f"Camera '{camera_id}' is not registered.",
                ),
            )

        store.update_camera_status(camera_id, "stopped")
        
        # Stop runner
        runner = _active_runners.pop(camera_id, None)
        if runner:
            await runner.stop()
            
        logger.info("camera.stopped", camera_id=camera_id)

        cameras = store.get_cameras()
        camera = next((c for c in cameras if c.get("camera_id") == camera_id), None)

        return ApiResponse(
            status="success",
            data=camera,
            meta=ResponseMeta(request_id=_get_request_id(request)),
        )
    except Exception as exc:
        logger.exception("cameras.stop_failed", camera_id=camera_id)
        return ApiResponse(
            status="error",
            meta=ResponseMeta(request_id=_get_request_id(request)),
            error=ErrorDetail(code="CAMERA_STOP_FAILED", message=str(exc)),
        )
