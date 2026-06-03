"""
Store Intelligence System — SQLAlchemy Database Models.

Persists events and anomalies to SQLite (dev) or PostgreSQL (prod).
Uses SQLAlchemy 2.0 async patterns.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class EventRecord(Base):
    """Persisted event record."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), unique=True, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    track_id = Column(Integer, nullable=True, index=True)
    bbox = Column(JSON, nullable=True)
    zone_id = Column(String(50), nullable=True, index=True)
    metadata_json = Column(JSON, nullable=True)

    def __repr__(self) -> str:
        return f"<EventRecord(event_id={self.event_id}, type={self.event_type})>"


class AnomalyRecord(Base):
    """Persisted anomaly record."""

    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    anomaly_id = Column(String(36), unique=True, nullable=False, index=True)
    anomaly_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(10), nullable=False, index=True)
    affected_zone = Column(String(50), nullable=False)
    camera_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    evidence = Column(JSON, nullable=True)
    suggested_action = Column(Text, nullable=True)
    resolved = Column(Boolean, default=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    resolved_notes = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<AnomalyRecord(anomaly_id={self.anomaly_id}, type={self.anomaly_type}, severity={self.severity})>"


class FootfallRecord(Base):
    """Aggregated footfall data for time-series queries."""

    __tablename__ = "footfall"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_id = Column(String(50), nullable=False, index=True)
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    granularity = Column(String(10), nullable=False)  # '1h' or '1d'
    count = Column(Integer, nullable=False, default=0)
    avg_dwell_seconds = Column(Float, nullable=True)
    peak_count = Column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<FootfallRecord(zone={self.zone_id}, count={self.count})>"
