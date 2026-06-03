"""
Store Intelligence System — Pydantic Schemas.

Defines all event types, API response envelopes, and query models.
These schemas are the contract between pipeline, streaming, analytics, and API layers.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════


class EventType(str, Enum):
    PERSON_DETECTED = "person_detected"
    PERSON_LOST = "person_lost"
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    DWELL_ALERT = "dwell_alert"
    DWELL_TIME_UPDATE = "dwell_time_update"
    CROWD_ALERT = "crowd_alert"
    FOOTFALL_SUMMARY = "footfall_summary"
    ANOMALY_DETECTED = "anomaly_detected"
    STORE_SUMMARY = "store_summary"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AnomalyType(str, Enum):
    CROWD_SURGE = "crowd_surge"
    ABANDONED_OBJECT = "abandoned_object"
    LOITERING = "loitering"
    CHECKOUT_QUEUE = "checkout_queue"
    AFTER_HOURS_ENTRY = "after_hours_entry"
    STATISTICAL_ANOMALY = "statistical_anomaly"


class ZoneType(str, Enum):
    ENTRY_EXIT = "entry_exit"
    PRODUCT = "product"
    CHECKOUT = "checkout"
    ENGAGEMENT = "engagement"


class CameraStatus(str, Enum):
    ACTIVE = "active"
    STOPPED = "stopped"
    ERROR = "error"


class PipelineStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"


# ═══════════════════════════════════════════════════════════
# Event Schemas
# ═══════════════════════════════════════════════════════════


class BaseEvent(BaseModel):
    """Base event emitted by the pipeline. All events share this schema."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    camera_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    track_id: Optional[int] = None
    bbox: Optional[List[float]] = Field(
        default=None, description="Normalized [x1, y1, x2, y2] in range 0-1"
    )
    zone_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat() + "Z"}}


class AnomalyEvent(BaseModel):
    """Anomaly detection result — embedded in BaseEvent.metadata or standalone."""

    anomaly_type: AnomalyType
    severity: Severity
    affected_zone: str
    evidence: Dict[str, Any]
    suggested_action: str


# ═══════════════════════════════════════════════════════════
# API Response Envelope
# ═══════════════════════════════════════════════════════════

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    message: str


class ResponseMeta(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    request_id: Optional[str] = None


class ApiResponse(BaseModel, Generic[T]):
    """Standard API response envelope."""

    status: str = "success"
    data: Optional[Any] = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    error: Optional[ErrorDetail] = None


# ═══════════════════════════════════════════════════════════
# Analytics Response Models
# ═══════════════════════════════════════════════════════════


class FootfallData(BaseModel):
    period: str
    count: int
    start_time: datetime
    end_time: datetime


class ZoneStats(BaseModel):
    zone_id: str
    zone_name: str
    zone_type: str
    visit_count: int
    avg_dwell_seconds: float
    peak_count: int
    current_occupancy: int


class HeatmapZone(BaseModel):
    zone_id: str
    zone_name: str
    intensity: float = Field(ge=0.0, le=1.0, description="Normalized 0-1 intensity")
    visit_count: int
    avg_dwell_seconds: float
    polygon: List[List[int]]
    color: str


class FunnelStage(BaseModel):
    stage: str
    count: int
    percentage: float


class PeakHourData(BaseModel):
    hour: int
    count: int


class TrackJourney(BaseModel):
    track_id: int
    first_seen: datetime
    last_seen: datetime
    total_duration_seconds: float
    zones_visited: List[str]
    zone_dwell_times: Dict[str, float]
    bbox_history: List[Dict[str, Any]]


class AnomalyRecord(BaseModel):
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: Severity
    affected_zone: str
    camera_id: str
    timestamp: datetime
    evidence: Dict[str, Any]
    suggested_action: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_notes: Optional[str] = None


class CameraInfo(BaseModel):
    camera_id: str
    source: str
    status: CameraStatus
    frames_processed: int = 0
    started_at: Optional[datetime] = None


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str
    pipeline_status: PipelineStatus
    active_cameras: int
    total_events: int
    total_anomalies: int


class ZoneLiveData(BaseModel):
    zone_id: str
    zone_name: str
    current_occupancy: int
    active_track_ids: List[int]
    avg_dwell_current: float


class StoreSummary(BaseModel):
    total_visitors_today: int
    current_in_store: int
    active_anomalies: int
    avg_dwell_seconds: float
    peak_hour: Optional[int] = None
    peak_hour_count: int = 0
    conversion_rate: float = 0.0


# ═══════════════════════════════════════════════════════════
# Query Parameters
# ═══════════════════════════════════════════════════════════


class EventFilter(BaseModel):
    event_types: Optional[List[EventType]] = None
    zone_id: Optional[str] = None
    camera_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    track_id: Optional[int] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)


class AnomalyFilter(BaseModel):
    severity: Optional[Severity] = None
    anomaly_type: Optional[AnomalyType] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    resolved: Optional[bool] = None


class ResolveRequest(BaseModel):
    notes: str = Field(default="", description="Resolution notes")


# ═══════════════════════════════════════════════════════════
# Zone Configuration
# ═══════════════════════════════════════════════════════════


class ZoneConfig(BaseModel):
    id: str
    name: str
    polygon: List[List[int]]
    type: ZoneType
    color: str = "#6366F1"


# ═══════════════════════════════════════════════════════════
# Spec-Compliant Event & Analytics Schemas
# ═══════════════════════════════════════════════════════════


class SpecEventType(str, Enum):
    """Event types defined by the external specification."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class SpecSeverity(str, Enum):
    """Anomaly severity levels for the spec-compliant API."""

    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class SpecEventMetadata(BaseModel):
    """Optional metadata attached to each spec event."""

    queue_depth: Optional[int] = Field(
        default=None, description="Current billing-queue depth at time of event"
    )
    sku_zone: Optional[str] = Field(
        default=None, description="SKU category / sub-zone label, e.g. MOISTURISER"
    )
    session_seq: int = Field(
        default=0, description="Monotonic event sequence within the visitor session"
    )


class SpecEvent(BaseModel):
    """Single event conforming to the external output specification.

    Every event emitted by the pipeline or accepted via the ingest
    endpoint must match this schema exactly.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID-v4 identifier for this event",
    )
    store_id: str = Field(..., description="Store identifier, e.g. STORE_BLR_002")
    camera_id: str = Field(..., description="Camera identifier, e.g. CAM_ENTRY_01")
    visitor_id: str = Field(
        ..., description="Visitor identifier, e.g. VIS_c8a2f1"
    )
    event_type: SpecEventType
    timestamp: datetime = Field(
        ..., description="ISO-8601 event timestamp"
    )
    zone_id: Optional[str] = Field(
        default=None, description="Zone where the event occurred"
    )
    dwell_ms: int = Field(
        default=0, ge=0, description="Dwell duration in milliseconds"
    )
    is_staff: bool = Field(default=False, description="True if the person is staff")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Detection / classification confidence"
    )
    metadata: SpecEventMetadata = Field(default_factory=SpecEventMetadata)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat() + "Z"}}


class IngestRequest(BaseModel):
    """Batch event ingestion request.

    Accepts up to 500 events per request to prevent oversized payloads.
    """

    events: List[SpecEvent] = Field(
        ..., max_length=500, description="Batch of spec-compliant events (max 500)"
    )


class IngestResponse(BaseModel):
    """Result of a batch event ingestion."""

    accepted: int = Field(default=0, description="Number of events successfully ingested")
    rejected: int = Field(default=0, description="Number of events that failed validation")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-event error details for rejected events",
    )


class StoreMetrics(BaseModel):
    """Aggregated KPI metrics for a single store."""

    unique_visitors: int = Field(default=0, description="Distinct visitor count")
    conversion_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Visitors who transacted / total visitors"
    )
    avg_dwell_per_zone: Dict[str, float] = Field(
        default_factory=dict,
        description="Zone-id → average dwell time in seconds",
    )
    queue_depth: int = Field(default=0, description="Current billing-queue depth")
    abandonment_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of queue-joiners who abandoned",
    )


class StoreFunnelStage(BaseModel):
    """One stage of the visitor conversion funnel."""

    stage: str = Field(..., description="Funnel stage name, e.g. 'Entered Store'")
    count: int = Field(default=0, ge=0, description="Number of visitors at this stage")
    drop_off_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of visitors who dropped off at this stage",
    )


class StoreHeatmapZone(BaseModel):
    """Zone-level heatmap data for the spec-compliant analytics API."""

    zone_id: str
    zone_name: str
    visit_frequency: int = Field(default=0, ge=0, description="Total visit count")
    avg_dwell_ms: int = Field(default=0, ge=0, description="Average dwell in milliseconds")
    intensity: int = Field(
        default=0, ge=0, le=100,
        description="Normalized heatmap intensity (0–100)",
    )
    data_confidence: bool = Field(
        default=True,
        description="False when sample size is too small for reliable metrics",
    )


class StoreAnomaly(BaseModel):
    """Anomaly record for the spec-compliant analytics API."""

    anomaly_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique anomaly identifier",
    )
    type: str = Field(..., description="Anomaly type, e.g. 'dead_zone', 'crowd_surge'")
    severity: SpecSeverity
    store_id: str
    zone_id: Optional[str] = None
    detected_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting data points for this anomaly",
    )
    suggested_action: str = Field(
        default="", description="Recommended remediation step"
    )


class SpecHealthStore(BaseModel):
    """Per-store health status in the spec health endpoint."""

    store_id: str
    last_event_ts: Optional[datetime] = Field(
        default=None, description="Timestamp of the most recent event from this store"
    )
    status: str = Field(default="unknown", description="Store feed status: ok | stale | offline")
    stale_feed: bool = Field(
        default=False,
        description="True when no events received within the stale-feed threshold",
    )


class SpecHealthResponse(BaseModel):
    """Top-level health / readiness probe response."""

    status: str = Field(default="ok", description="Overall system status")
    version: str = Field(default="1.0.0", description="API version string")
    uptime_seconds: float = Field(default=0.0, description="Process uptime in seconds")
    stores: List[SpecHealthStore] = Field(
        default_factory=list, description="Per-store health summaries"
    )
