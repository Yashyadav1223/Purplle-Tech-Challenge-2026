"""
Store Intelligence System — Configuration via Pydantic BaseSettings.

All settings are loaded from environment variables / .env file.
No hardcoded values — everything is configurable.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class StreamBackendType(str, Enum):
    KAFKA = "kafka"
    REDIS = "redis"
    MEMORY = "memory"


class Settings(BaseSettings):
    """Central application configuration."""

    # ── General ──────────────────────────────────────────────
    APP_NAME: str = "Store Intelligence System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: LogLevel = LogLevel.INFO

    # ── Video Ingestion ──────────────────────────────────────
    VIDEO_FPS: int = Field(default=5, ge=1, le=30, description="Frames per second to process")
    VIDEO_FRAME_WIDTH: int = 640
    VIDEO_FRAME_HEIGHT: int = 640
    FRAME_BUFFER_SIZE: int = 128
    ENABLE_CLAHE: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_GRID_SIZE: int = 8

    # ── Detection (YOLOv8) ───────────────────────────────────
    YOLO_MODEL: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = Field(default=0.4, ge=0.1, le=1.0)
    USE_CUDA: bool = False
    TARGET_CLASSES: List[int] = Field(
        default=[0, 24, 26, 39, 67],
        description="COCO class IDs: person=0, backpack=24, handbag=26, bottle=39, cell_phone=67",
    )
    TRACKER_TYPE: str = Field(default="bytetrack.yaml", description="bytetrack.yaml or botsort.yaml")

    # ── Zones ────────────────────────────────────────────────
    ZONES_CONFIG_PATH: str = Field(default="config/zones.json")
    DWELL_ALERT_THRESHOLD_SECONDS: int = Field(default=120, description="Dwell time anomaly threshold")
    DWELL_UPDATE_INTERVAL_SECONDS: int = Field(default=5)
    CROWD_THRESHOLD: int = Field(default=10, description="Max people in a zone before crowd alert")

    # ── Business Hours ───────────────────────────────────────
    BUSINESS_HOURS_START: int = Field(default=9, description="Business hours start (24h format)")
    BUSINESS_HOURS_END: int = Field(default=21, description="Business hours end (24h format)")

    # ── Streaming ────────────────────────────────────────────
    STREAM_BACKEND: StreamBackendType = StreamBackendType.MEMORY

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_PREFIX: str = "store"
    KAFKA_MAX_RETRIES: int = 3
    KAFKA_RETRY_BACKOFF_MS: int = 500

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_STREAM_MAXLEN: int = 10000

    # ── Database ─────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./store_intelligence.db",
        description="SQLAlchemy async database URL",
    )

    # ── API ──────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_CORS_ORIGINS: List[str] = Field(default=["*"])
    API_RATE_LIMIT: str = "100/minute"

    # ── Analytics ────────────────────────────────────────────
    ANALYTICS_UPDATE_INTERVAL_SECONDS: int = 30
    FOOTFALL_SUMMARY_INTERVAL_SECONDS: int = 60
    STORE_SUMMARY_INTERVAL_SECONDS: int = 300
    ANOMALY_BASELINE_WINDOW_HOURS: int = 24
    ANOMALY_RETRAIN_INTERVAL_SECONDS: int = 3600

    # Anomaly thresholds
    LOITERING_THRESHOLD_SECONDS: int = 300
    ABANDONED_OBJECT_THRESHOLD_SECONDS: int = 60
    CHECKOUT_QUEUE_THRESHOLD: int = 8
    ZSCORE_THRESHOLD: float = 2.5

    # ── Staff Classification ─────────────────────────────────
    STAFF_DWELL_THRESHOLD_SECONDS: int = Field(
        default=1800, description="Session duration (seconds) after which a track is flagged as staff",
    )
    STAFF_ZONE_VISIT_THRESHOLD: int = Field(
        default=5, description="Distinct zone count to flag a track as staff",
    )

    # ── Re-entry Detection ───────────────────────────────────
    REENTRY_WINDOW_SECONDS: int = Field(
        default=300, description="Seconds after exit during which a new track is treated as re-entry",
    )

    # ── Dwell Event Interval ─────────────────────────────────
    DWELL_EVENT_INTERVAL_MS: int = Field(
        default=30000, description="Interval (ms) between periodic ZONE_DWELL spec events",
    )

    # ── Metrics ──────────────────────────────────────────────
    ENABLE_PROMETHEUS: bool = True
    METRICS_PORT: int = 9090

    # ── Store Intelligence (Spec Pipeline) ───────────────────
    STORE_LAYOUT_PATH: str = "config/store_layout.json"
    POS_DATA_PATH: str = "config/pos_transactions.csv"
    STAFF_DWELL_THRESHOLD_SECONDS: int = Field(
        default=1800, description="Cumulative dwell above this (30 min) → likely staff"
    )
    STAFF_ZONE_VISIT_THRESHOLD: int = Field(
        default=5, description="Visiting 5+ distinct zones in one session → likely staff"
    )
    REENTRY_WINDOW_SECONDS: int = Field(
        default=300, description="Same visitor re-entering within 5 min → REENTRY event"
    )
    DWELL_EVENT_INTERVAL_MS: int = Field(
        default=30000, description="Emit a ZONE_DWELL event every 30 s while in zone"
    )
    DEAD_ZONE_THRESHOLD_SECONDS: int = Field(
        default=1800, description="No visits for 30 min → dead-zone anomaly"
    )
    STALE_FEED_THRESHOLD_SECONDS: int = Field(
        default=600, description="No events for 10 min → stale camera feed alert"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
