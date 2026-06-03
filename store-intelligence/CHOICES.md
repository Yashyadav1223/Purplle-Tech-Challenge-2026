# CHOICES.md — Detection Model & Technical Decisions

## Decision 1: Detection Model Selection

### Options Considered

| Model | FPS (CPU) | mAP@50 | Size | Tracking Support |
|-------|-----------|--------|------|-----------------|
| YOLOv8n | ~120 FPS | 37.3 | 6.2 MB | Built-in ByteTrack/BoTSORT |
| YOLOv8s | ~65 FPS | 44.9 | 22.5 MB | Built-in ByteTrack/BoTSORT |
| YOLOv9t | ~90 FPS | 38.3 | 7.7 MB | Requires external tracker |
| RT-DETR-l | ~35 FPS | 53.0 | 32 MB | No built-in tracking |
| MediaPipe Pose | ~30 FPS | N/A | 10 MB | No multi-person tracking |

### What AI Suggested
The LLM recommended YOLOv8s (small) for better accuracy, noting that the 44.9 mAP would significantly improve group detection and partially occluded person counting.

### What We Chose: YOLOv8n (nano)

**Rationale:**
1. **Throughput requirement**: 3 cameras × 15 FPS = 45 frames/second per store. On CPU (no GPU assumed for retail edge deployment), YOLOv8n processes ~120 FPS while YOLOv8s drops to ~65 FPS. The nano model gives us 2.6× headroom.
2. **Built-in tracking**: Ultralytics YOLO has first-class ByteTrack integration via `model.track(persist=True)`. No external tracker library needed. This eliminates a major integration surface.
3. **Sufficient accuracy**: For person detection (COCO class 0), YOLOv8n achieves ~55% AP — significantly higher than the model-wide 37.3% mAP. People are large, distinctive objects; nano is adequate.
4. **Model size**: 6.2 MB loads in <1s. Critical for `docker compose up` cold-start time.

**What we'd change with more time**: Use YOLOv8s or YOLOv9 with TensorRT optimization for GPU-equipped stores.

---

## Decision 2: Event Schema Design

### Options Considered

**Option A — Fully Normalized (relational)**
```json
{
  "event": {"id": "...", "type": "ZONE_ENTER", "timestamp": "..."},
  "subject": {"visitor_id": "...", "is_staff": false},
  "location": {"store_id": "...", "camera_id": "...", "zone_id": "..."},
  "metrics": {"dwell_ms": 0, "confidence": 0.91},
  "context": {"queue_depth": null, "session_seq": 5}
}
```

**Option B — Flat with metadata bag (spec-compliant)**
```json
{
  "event_id": "uuid-v4",
  "store_id": "STORE_BLR_002",
  "camera_id": "CAM_ENTRY_01",
  "visitor_id": "VIS_c8a2f1",
  "event_type": "ZONE_DWELL",
  "timestamp": "2026-03-03T14:22:10Z",
  "zone_id": "SKINCARE",
  "dwell_ms": 8400,
  "is_staff": false,
  "confidence": 0.91,
  "metadata": {"queue_depth": null, "sku_zone": "MOISTURISER", "session_seq": 5}
}
```

**Option C — Event-type-specific schemas**
Different schema per event type (EntryEvent, ExitEvent, DwellEvent, etc.)

### What AI Suggested
The LLM initially proposed Option A (normalized) with separate sub-objects, arguing it would be more maintainable and queryable. It also suggested Option C for type safety.

### What We Chose: Option B — Flat with metadata bag

**Rationale:**
1. **Spec compliance**: The challenge spec explicitly defines this schema. Deviating would fail schema validation.
2. **Query simplicity**: A flat schema enables `SELECT * FROM events WHERE store_id='...' AND event_type='ZONE_ENTER' AND zone_id='SKINCARE'` without joins or JSON path extraction.
3. **Streaming compatibility**: Kafka/Redis consumers can filter on top-level fields without parsing nested objects.
4. **Metadata flexibility**: The `metadata` bag accommodates event-type-specific fields (queue_depth for billing events, sku_zone for zone events) without schema migration.

---

## Decision 3: API Architecture — FastAPI + In-Memory State

### Options Considered

| Architecture | Latency | Complexity | Scalability |
|-------------|---------|------------|-------------|
| FastAPI + In-Memory StateStore | <1ms reads | Low | Single process |
| FastAPI + Redis | ~2ms reads | Medium | Multi-process |
| FastAPI + PostgreSQL/TimescaleDB | ~10ms reads | High | Full horizontal |
| gRPC + Redis | <1ms reads | High | Multi-process |

### What AI Suggested
The LLM recommended FastAPI + Redis for the state store, arguing that Redis provides persistence across restarts and enables horizontal scaling via Redis Cluster.

### What We Chose: FastAPI + In-Memory StateStore

**Rationale:**
1. **Latency**: The spec requires real-time metrics "not cached from yesterday." An in-memory store with `threading.RLock` gives sub-millisecond reads — 10× faster than Redis.
2. **Simplicity**: No Redis connection management, no serialization overhead, no network failure modes. The `docker compose up` starts faster.
3. **Event sourcing**: The `POST /events/ingest` endpoint is the source of truth. On restart, events can be replayed to rebuild state. The state store is a materialized view, not the primary store.
4. **Idempotency**: The event_id deduplication set lives in memory. Duplicate ingest calls are O(1) set lookups.
5. **Trade-off**: State is lost on process restart. For production, we'd add:
   - Startup rehydration from SQLite/PostgreSQL
   - Periodic state snapshots to Redis
   - Or switch to Redis as primary store with connection pooling

### VLM Usage

We did **not** use a Vision-Language Model (VLM) in the pipeline. We evaluated using a VLM for:
- **Zone classification**: Asking a VLM "What zone of the store is this person in?" — rejected because polygon-based geometric testing is 1000× faster and deterministic.
- **Staff detection**: Asking a VLM "Is this person wearing a uniform?" — rejected because faces are blurred in the dataset, uniforms are not guaranteed visible, and a heuristic (dwell time + zone count) is more reliable.
- **Activity recognition**: "What is this person doing?" — rejected because the spec requires structured events (ENTRY/EXIT/DWELL), not free-text descriptions. The tracking + zone mapping pipeline produces these events directly.

VLMs would add value for open-ended analytics questions ("What's the most common customer behaviour pattern?") but are not needed for the structured event pipeline the spec requires.
