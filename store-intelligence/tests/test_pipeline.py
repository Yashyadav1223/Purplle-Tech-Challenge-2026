"""
Store Intelligence System — Pipeline Unit Tests.

Tests video source initialization, frame preprocessing, zone polygon
containment, and track state management without requiring a real YOLO model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from analytics.store import StateStore, TrackState
from config.settings import Settings


# ═══════════════════════════════════════════════════════════
# Zone Polygon Helpers (pure geometry — no model needed)
# ═══════════════════════════════════════════════════════════


def point_in_polygon(point: Tuple[float, float], polygon: List[List[int]]) -> bool:
    """
    Ray-casting algorithm to test if a point lies inside a polygon.
    This mirrors what the zone mapper does internally.
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def point_in_polygon_cv2(point: Tuple[float, float], polygon: List[List[int]]) -> bool:
    """Use cv2.pointPolygonTest for a second opinion."""
    contour = np.array(polygon, dtype=np.float32)
    result = cv2.pointPolygonTest(contour, point, measureDist=False)
    return result >= 0  # >=0 means inside or on edge


# ═══════════════════════════════════════════════════════════
# Tests: VideoSource Initialization
# ═══════════════════════════════════════════════════════════


class TestVideoSourceInit:
    """Test VideoSource construction (without actually opening a stream)."""

    def test_video_source_attributes(self, test_settings: Settings):
        """VideoSource should store camera_id, source, and settings-driven defaults."""
        with patch("pipeline.ingestion.get_settings", return_value=test_settings):
            from pipeline.ingestion import VideoSource

            vs = VideoSource("cam01", "test_data/sample_store.mp4")
            assert vs.camera_id == "cam01"
            assert vs.source == "test_data/sample_store.mp4"
            assert vs.fps == test_settings.VIDEO_FPS
            assert vs.buffer_size == test_settings.FRAME_BUFFER_SIZE
            assert vs.is_running is False
            assert vs.frame_count == 0

    def test_video_source_custom_fps(self, test_settings: Settings):
        """Custom fps should override the settings default."""
        with patch("pipeline.ingestion.get_settings", return_value=test_settings):
            from pipeline.ingestion import VideoSource

            vs = VideoSource("cam02", 0, fps=30, buffer_size=64)
            assert vs.fps == 30
            assert vs.buffer_size == 64

    def test_video_source_clahe_disabled(self, test_settings: Settings):
        """When CLAHE is disabled, _clahe attribute should be None."""
        test_settings.ENABLE_CLAHE = False
        with patch("pipeline.ingestion.get_settings", return_value=test_settings):
            from pipeline.ingestion import VideoSource

            vs = VideoSource("cam03", "fake.mp4")
            assert vs._clahe is None

    def test_video_source_clahe_enabled(self, test_settings: Settings):
        """When CLAHE is enabled, _clahe attribute should be a cv2.CLAHE object."""
        test_settings.ENABLE_CLAHE = True
        with patch("pipeline.ingestion.get_settings", return_value=test_settings):
            from pipeline.ingestion import VideoSource

            vs = VideoSource("cam04", "fake.mp4")
            assert vs._clahe is not None


# ═══════════════════════════════════════════════════════════
# Tests: Frame Preprocessing
# ═══════════════════════════════════════════════════════════


class TestFramePreprocessing:
    """Test the _preprocess method of VideoSource."""

    def test_resize_to_target(self, test_settings: Settings, sample_frame_1080p: np.ndarray):
        """A 1080p frame should be resized to VIDEO_FRAME_WIDTH×VIDEO_FRAME_HEIGHT."""
        test_settings.ENABLE_CLAHE = False
        with patch("pipeline.ingestion.get_settings", return_value=test_settings):
            from pipeline.ingestion import VideoSource

            vs = VideoSource("cam-resize", "fake.mp4")
            result = vs._preprocess(sample_frame_1080p)
            assert result.shape == (test_settings.VIDEO_FRAME_HEIGHT, test_settings.VIDEO_FRAME_WIDTH, 3)
            assert result.dtype == np.uint8

    def test_resize_preserves_dtype(self, test_settings: Settings, sample_frame_480p: np.ndarray):
        """Output dtype should always be uint8."""
        test_settings.ENABLE_CLAHE = False
        with patch("pipeline.ingestion.get_settings", return_value=test_settings):
            from pipeline.ingestion import VideoSource

            vs = VideoSource("cam-dtype", "fake.mp4")
            result = vs._preprocess(sample_frame_480p)
            assert result.dtype == np.uint8

    def test_clahe_changes_pixel_values(self, test_settings: Settings):
        """CLAHE should modify pixel values in the luminance channel."""
        test_settings.ENABLE_CLAHE = True
        test_settings.VIDEO_FRAME_WIDTH = 100
        test_settings.VIDEO_FRAME_HEIGHT = 100

        # Create a frame with low-contrast gradient
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(40, 60, 100, dtype=np.uint8)  # subtle gradient in blue
        frame[:, :, 1] = 50
        frame[:, :, 2] = 50

        with patch("pipeline.ingestion.get_settings", return_value=test_settings):
            from pipeline.ingestion import VideoSource

            vs = VideoSource("cam-clahe", "fake.mp4")
            result = vs._preprocess(frame)

            # CLAHE should enhance contrast → pixel range should increase
            assert result.shape == (100, 100, 3)
            # Not identical to input
            assert not np.array_equal(result, frame)

    def test_preprocess_no_clahe_passthrough(self, test_settings: Settings):
        """Without CLAHE, output should only differ by resize."""
        test_settings.ENABLE_CLAHE = False
        test_settings.VIDEO_FRAME_WIDTH = 320
        test_settings.VIDEO_FRAME_HEIGHT = 320

        frame = np.full((320, 320, 3), 128, dtype=np.uint8)

        with patch("pipeline.ingestion.get_settings", return_value=test_settings):
            from pipeline.ingestion import VideoSource

            vs = VideoSource("cam-pass", "fake.mp4")
            result = vs._preprocess(frame)
            # Same size → should be identical (no resize interpolation artefacts at exact size)
            np.testing.assert_array_equal(result, frame)


# ═══════════════════════════════════════════════════════════
# Tests: Zone Polygon Containment
# ═══════════════════════════════════════════════════════════


class TestZonePolygon:
    """Test point-in-polygon logic used by the zone mapper."""

    def test_point_inside_entrance(self, zones_config):
        """A point well inside the entrance polygon should be detected."""
        entrance = zones_config[0]["polygon"]
        assert point_in_polygon((100, 520), entrance) is True
        assert point_in_polygon_cv2((100, 520), entrance) is True

    def test_point_outside_entrance(self, zones_config):
        """A point far from the entrance should not be inside."""
        entrance = zones_config[0]["polygon"]
        assert point_in_polygon((400, 100), entrance) is False
        assert point_in_polygon_cv2((400, 100), entrance) is False

    def test_point_in_lipstick_aisle(self, zones_config):
        """Centre of lipstick aisle polygon."""
        lipstick = zones_config[1]["polygon"]
        assert point_in_polygon((320, 175), lipstick) is True
        assert point_in_polygon_cv2((320, 175), lipstick) is True

    def test_point_in_skincare_aisle(self, zones_config):
        """Centre of skincare aisle polygon."""
        skincare = zones_config[2]["polygon"]
        assert point_in_polygon((540, 175), skincare) is True

    def test_point_in_trial_area(self, zones_config):
        """Centre of trial area polygon."""
        trial = zones_config[3]["polygon"]
        assert point_in_polygon((320, 420), trial) is True

    def test_point_in_checkout(self, zones_config):
        """Centre of checkout polygon."""
        checkout = zones_config[4]["polygon"]
        assert point_in_polygon((540, 520), checkout) is True

    def test_point_between_zones(self, zones_config):
        """A point in the gap between zones should not be in any zone."""
        gap_point = (210, 310)  # between entrance and lipstick
        for zone in zones_config:
            assert point_in_polygon(gap_point, zone["polygon"]) is False

    def test_edge_point(self, zones_config):
        """A point on the polygon edge — cv2 considers it inside."""
        entrance = zones_config[0]["polygon"]
        # (0, 400) is a vertex
        assert point_in_polygon_cv2((0, 400), entrance) is True


# ═══════════════════════════════════════════════════════════
# Tests: Track State Management
# ═══════════════════════════════════════════════════════════


class TestTrackState:
    """Test StateStore track operations."""

    def test_new_track_insert(self, mock_state_store: StateStore):
        """First upsert for a track_id should report is_new=True."""
        is_new, prev_zone, cur_zone = mock_state_store.upsert_track(
            100, "cam01", [0.1, 0.2, 0.3, 0.4], zone_id="entrance"
        )
        assert is_new is True
        assert prev_zone == "entrance"
        assert cur_zone == "entrance"

    def test_existing_track_update(self, mock_state_store: StateStore):
        """Second upsert should report is_new=False."""
        mock_state_store.upsert_track(200, "cam01", [0.1, 0.2, 0.3, 0.4])
        is_new, _, _ = mock_state_store.upsert_track(200, "cam01", [0.2, 0.3, 0.4, 0.5])
        assert is_new is False

    def test_zone_transition(self, mock_state_store: StateStore):
        """Upsert with a different zone_id should return prev_zone and new zone."""
        mock_state_store.upsert_track(300, "cam01", [0.1, 0.2, 0.3, 0.4], zone_id="entrance")
        is_new, prev_zone, cur_zone = mock_state_store.upsert_track(
            300, "cam01", [0.4, 0.1, 0.6, 0.4], zone_id="lipstick_aisle"
        )
        assert is_new is False
        assert prev_zone == "entrance"
        assert cur_zone == "lipstick_aisle"

    def test_mark_track_lost(self, mock_state_store: StateStore):
        """Marking a track lost should move it from tracks to lost_tracks."""
        mock_state_store.upsert_track(400, "cam01", [0.1, 0.2, 0.3, 0.4])
        assert 400 in mock_state_store.tracks

        lost = mock_state_store.mark_track_lost(400)
        assert lost is not None
        assert lost.active is False
        assert 400 not in mock_state_store.tracks
        assert 400 in mock_state_store.lost_tracks

    def test_mark_nonexistent_track_lost(self, mock_state_store: StateStore):
        """Marking a non-existent track should return None."""
        assert mock_state_store.mark_track_lost(9999) is None

    def test_get_active_track_ids(self, mock_state_store: StateStore):
        """get_active_track_ids should return current active tracks."""
        for tid in [10, 20, 30]:
            mock_state_store.upsert_track(tid, "cam01", [0, 0, 1, 1])
        ids = mock_state_store.get_active_track_ids()
        assert ids == {10, 20, 30}

    def test_visitor_count_increments(self, mock_state_store: StateStore):
        """Each new track should increment total_visitors_today."""
        initial = mock_state_store.total_visitors_today
        for tid in [501, 502, 503]:
            mock_state_store.upsert_track(tid, "cam01", [0, 0, 1, 1])
        assert mock_state_store.total_visitors_today == initial + 3


# ═══════════════════════════════════════════════════════════
# Tests: Detection Dataclass
# ═══════════════════════════════════════════════════════════


class TestDetectionDataclass:
    """Test the Detection dataclass properties."""

    def test_center_calculation(self):
        """center property should return midpoint of bbox."""
        from pipeline.detector import Detection

        d = Detection(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox_xyxy=(100.0, 200.0, 300.0, 400.0),
            bbox_normalized=(0.15625, 0.3125, 0.46875, 0.625),
        )
        cx, cy = d.center
        assert cx == pytest.approx(200.0)
        assert cy == pytest.approx(300.0)

    def test_center_normalized(self):
        """center_normalized should return midpoint in 0-1 range."""
        from pipeline.detector import Detection

        d = Detection(
            track_id=2,
            class_id=0,
            class_name="person",
            confidence=0.88,
            bbox_xyxy=(0, 0, 640, 640),
            bbox_normalized=(0.0, 0.0, 1.0, 1.0),
        )
        cx, cy = d.center_normalized
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
