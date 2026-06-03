# DESIGN.md — Store Intelligence System Architecture

## Overview

The Store Intelligence System is a complete pipeline that transforms raw CCTV footage into actionable retail analytics. It processes video from up to 15 cameras (5 stores × 3 angles), detects and tracks people in real time, emits structured behavioural events, and exposes a queryable REST API backed by a live dashboard.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DETECTION LAYER                              │
│                                                                     │
│  ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐  │
│  │ VideoSrc  │──▶│ YOLOv8n  │──▶│ ByteTrack │──▶│ Zone Mapper  │  │
│  │ (CLAHE)   │   │ Detector │   │  Tracker  │   │ (Polygon)    │  │
│  └───────────┘   └──────────┘   └───────────┘   └──────────────┘  │
│        │                                                │           │
│        ▼                                                ▼           │
│  ┌───────────┐                                  ┌──────────────┐   │
│  │  Staff    │                                  │   Event      │   │
│  │Classifier │                                  │  Emitter     │   │
│  └───────────┘                                  └──────┬───────┘   │
└────────────────────────────────────────────────────────┬────────────┘
                                                         │
                              Structured JSON Events     │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         EVENT STREAM                                │
│                                                                     │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Kafka  │    │ Redis Stream │    │  In-Memory   │               │
│  │ (prod)  │    │  (staging)   │    │   (dev)      │               │
│  └─────────┘    └──────────────┘    └──────────────┘               │
└────────────────────────────────────────────────────────┬────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INTELLIGENCE API                               │
│                                                                     │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐               │
│  │ Multi-Store  │  │  Anomaly   │  │    POS       │               │
│  │ State Store  │  │  Detector  │  │  Correlator  │               │
│  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘               │
│         │                │                 │                        │
│         ▼                ▼                 ▼                        │
│  ┌─────────────────────────────────────────────────────┐           │
│  │              FastAPI REST Endpoints                  │           │
│  │  POST /events/ingest   GET /stores/{id}/metrics     │           │
│  │  GET /stores/{id}/funnel  GET /stores/{id}/heatmap  │           │
│  │  GET /stores/{id}/anomalies  GET /health            │           │
│  └─────────────────────────────┬───────────────────────┘           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LIVE DASHBOARD                                │
│                                                                     │
│  React + Vite + Recharts + WebSocket                               │
│  Real-time visitor count, zone heatmap, conversion funnel,         │
│  anomaly alerts, MJPEG camera feed                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. YOLOv8n + ByteTrack — Speed Over Accuracy

We chose YOLOv8n (nano) with ByteTrack for detection + tracking because:

- **Real-time constraint**: At 15 FPS × 3 cameras per store, we need ~45 inferences/second per store. YOLOv8n achieves ~8ms/frame on CPU, making it feasible without GPU.
- **ByteTrack over DeepSORT**: ByteTrack uses only bounding box IoU for association — no appearance features needed. This is 3× faster than DeepSORT and sufficient for single-camera tracking where people don't occlude heavily.
- **Trade-off**: We sacrifice some accuracy in crowded scenes (group handling) for throughput. The `confidence` field in events propagates this uncertainty downstream.

### 2. In-Memory State Store — Latency Over Durability

The StateStore uses `threading.RLock`-protected Python dicts rather than Redis or PostgreSQL for the hot path:

- **Sub-millisecond reads**: API endpoints that compute metrics (heatmap, funnel, anomalies) read directly from memory. No network hop.
- **Event-sourced**: All state is reconstructable from the event stream. The `POST /events/ingest` endpoint can replay events on restart.
- **Trade-off**: State is lost on restart. In production, a startup rehydration step from the database or Redis would be added.

### 3. Pluggable Streaming Backends — Same Code, Different Infra

The `StreamBackend` ABC allows swapping Kafka (production), Redis Streams (staging), and in-memory (development) without touching pipeline code:

- **Kafka for durability**: Partitioned by store_id, 24h retention, exactly-once semantics.
- **In-memory for tests**: No external dependencies for CI/CD.
- **Trade-off**: The ABC adds a thin abstraction layer. In-memory backend has no partitioning or replay.

### 4. Heuristic Staff Classification — Rules Over ML

Staff detection uses a rules-based classifier rather than a trained model:

- **Session duration > 30 min**: Staff are present for entire shifts.
- **Zone visit count > 5**: Staff patrol all zones; customers visit 1-3.
- **No training data needed**: Works immediately on any store without labelled data.
- **Trade-off**: May misclassify long-browsing customers as staff in edge cases. The `is_staff` flag is always present in events, allowing downstream correction.

### 5. Polygon-Based Zone Mapping — Flexibility Over Speed

Zone detection uses `cv2.pointPolygonTest` with arbitrary polygons from JSON config:

- **Store-specific layouts**: Each store defines its own zone polygons matching its floor plan.
- **No grid quantization**: Arbitrary polygon shapes (L-shaped zones, triangular displays).
- **Trade-off**: ~0.1ms per point per polygon. Negligible at <100 tracked people, but would need spatial indexing at scale.

## AI-Assisted Decisions

### 1. Event Schema Design
**What AI suggested**: A deeply nested event schema with separate sub-objects for spatial data, temporal data, and behavioural data.
**What we chose**: A flat schema with a single `metadata` object. The spec's schema is intentionally flat — `dwell_ms`, `is_staff`, and `confidence` are top-level fields, not buried in nested objects. This makes SQL queries and filtering trivial. We agreed with the spec's design over the AI's suggestion.

### 2. Re-entry Detection Strategy
**What AI suggested**: Full appearance-based Re-ID using OSNet embeddings to match people across camera views.
**What we chose**: Time-window-based re-entry detection — if a new track appears within 5 minutes of a prior exit from the same camera, it's flagged as a REENTRY with the same visitor_id. This is simpler, faster, and sufficient for single-entrance stores. We partially agreed — appearance Re-ID would be better but exceeds the time budget.

### 3. Anomaly Detection: Rules vs ML
**What AI suggested**: An autoencoder-based anomaly detector trained on historical traffic patterns.
**What we chose**: A hybrid approach — hard-coded business rules (queue spike, after-hours, loitering) combined with statistical Z-score on rolling windows. This avoids the cold-start problem and gives interpretable `suggested_action` strings. We overrode the AI's suggestion because the system needs to work on day one without historical data.
