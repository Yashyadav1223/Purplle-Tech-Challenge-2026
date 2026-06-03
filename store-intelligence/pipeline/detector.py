"""
Store Intelligence System — YOLOv8 Object Detector.

Wraps the Ultralytics YOLO model to perform detection + tracking on
individual frames.  Returns lightweight ``Detection`` dataclass objects
ready for downstream consumption by tracker and zone-mapper stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import structlog
from ultralytics import YOLO

from config.settings import get_settings

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Detection:
    """Single detection result for one tracked object in a frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    bbox_normalized: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in 0-1 range

    @property
    def center(self) -> Tuple[float, float]:
        """Centre point of the bounding box in pixel coordinates."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def center_normalized(self) -> Tuple[float, float]:
        """Centre point of the bounding box in normalised 0-1 coordinates."""
        x1, y1, x2, y2 = self.bbox_normalized
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ═══════════════════════════════════════════════════════════
# Detector
# ═══════════════════════════════════════════════════════════


class Detector:
    """
    YOLOv8 detector with built-in ByteTrack / BoTSORT tracking.

    Usage::

        det = Detector()
        detections = det.detect(frame, frame_width=640, frame_height=640)
        for d in detections:
            print(d.track_id, d.class_name, d.confidence)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._log = logger.bind(component="detector")

        device = "cuda" if self._settings.USE_CUDA else "cpu"
        self._log.info(
            "detector.loading_model",
            model=self._settings.YOLO_MODEL,
            device=device,
            confidence=self._settings.CONFIDENCE_THRESHOLD,
            tracker=self._settings.TRACKER_TYPE,
            target_classes=self._settings.TARGET_CLASSES,
        )

        try:
            self._model = YOLO(self._settings.YOLO_MODEL)
            if self._settings.USE_CUDA:
                self._model.to("cuda")
        except Exception:
            self._log.exception("detector.model_load_failed")
            raise

        # Cache the COCO class-name map from the model.
        self._class_names: dict[int, str] = self._model.names or {}
        self._log.info(
            "detector.model_ready",
            num_classes=len(self._class_names),
        )

    # ── Public API ────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> List[Detection]:
        """
        Run detection + tracking on a single BGR frame.

        Args:
            frame: BGR ``uint8`` ndarray ``(H, W, 3)``.
            frame_width: Width used for bbox normalisation.
                         Defaults to ``frame.shape[1]``.
            frame_height: Height used for bbox normalisation.
                          Defaults to ``frame.shape[0]``.

        Returns:
            List of ``Detection`` objects for every tracked entity.
        """
        h, w = frame.shape[:2]
        fw = frame_width or w
        fh = frame_height or h

        try:
            results = self._model.track(
                frame,
                persist=True,
                tracker=self._settings.TRACKER_TYPE,
                conf=self._settings.CONFIDENCE_THRESHOLD,
                classes=self._settings.TARGET_CLASSES,
                verbose=False,
            )
        except Exception:
            self._log.exception("detector.track_failed")
            return []

        return self._parse_results(results, fw, fh)

    def reset_tracker(self) -> None:
        """
        Reset the internal tracker state.

        Useful when switching between video sources on the same
        ``Detector`` instance.
        """
        try:
            self._model.predictor = None  # type: ignore[assignment]
            self._log.info("detector.tracker_reset")
        except Exception:
            self._log.warning("detector.tracker_reset_failed")

    # ── Internal ──────────────────────────────────────────

    def _parse_results(
        self,
        results: list,
        frame_width: int,
        frame_height: int,
    ) -> List[Detection]:
        """Parse Ultralytics ``Results`` into ``Detection`` objects."""
        detections: List[Detection] = []

        if not results:
            return detections

        result = results[0]  # single-image inference → one result
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # ``boxes.id`` is None when the tracker has no active tracks.
        track_ids = boxes.id
        if track_ids is None:
            self._log.debug("detector.no_active_tracks")
            return detections

        track_ids_list = track_ids.int().cpu().tolist()
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.int().cpu().tolist()

        for tid, xyxy, conf, cid in zip(track_ids_list, xyxys, confs, class_ids):
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            bbox_norm = (
                x1 / frame_width,
                y1 / frame_height,
                x2 / frame_width,
                y2 / frame_height,
            )

            detections.append(
                Detection(
                    track_id=int(tid),
                    class_id=int(cid),
                    class_name=self._class_names.get(int(cid), f"class_{cid}"),
                    confidence=round(float(conf), 4),
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_normalized=bbox_norm,
                )
            )

        return detections
