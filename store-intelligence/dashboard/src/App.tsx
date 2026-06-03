import { useState, useEffect, useRef, useCallback } from "react";
import {
  Activity, Users, ShoppingCart, Clock, AlertTriangle, LogOut,
  MonitorPlay, Square, UploadCloud, Video, Zap, MapPin,
  TrendingUp, Shield, CircleDot, ArrowRight, BarChart3,
  Eye, Cpu, FileVideo, X, Store, ChevronDown,
  Flame, Target,
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";

/* ─── Types ─── */
interface Stats {
  visitors: number; exits: number; conversion: number;
  dwell: string; queueDepth: number; anomalies: number;
  abandonmentRate: number;
}
interface FunnelStage { stage: string; count: number; percentage: number; drop_off_pct: number; }
interface ZoneData {
  zone_id: string; zone_name: string; visit_count: number;
  avg_dwell_seconds: number; current_occupancy: number;
}
interface HeatmapZone {
  zone_id: string; zone_name: string; visit_frequency: number;
  avg_dwell_ms: number; intensity: number; data_confidence: boolean;
}
interface FootfallPoint { period: string; count: number; }
interface EventItem {
  id: string; time: string; badge: string; badgeBg: string;
  badgeColor: string; text: string; vid?: string;
}
interface AnomalyItem {
  id: string; title: string; detail: string; severity: string;
  suggested_action?: string;
}
interface StoreOption { id: string; name: string; }

const STORES: StoreOption[] = [
  { id: "STORE_BLR_002", name: "Bangalore – Koramangala" },
];

const ZONE_COLORS: Record<string, string> = {
  entrance: "#3B82F6", ENTRY_EXIT: "#3B82F6",
  lipstick_aisle: "#EC4899", LIPSTICK: "#EC4899",
  skincare_aisle: "#8B5CF6", SKINCARE: "#8B5CF6",
  trial_area: "#F59E0B", CENTER_DISPLAY: "#F59E0B",
  checkout: "#10B981", CASH_COUNTER: "#10B981", BILLING: "#10B981",
  LEFT_SHELF: "#6366F1", RIGHT_SHELF: "#F97316",
};
const ZONE_ORDER = ["ENTRY_EXIT", "LEFT_SHELF", "RIGHT_SHELF", "CENTER_DISPLAY", "SKINCARE", "LIPSTICK", "CASH_COUNTER", "entrance", "lipstick_aisle", "skincare_aisle", "trial_area", "checkout"];
const DEFAULT_ZONES: ZoneData[] = [
  { zone_id: "ENTRY_EXIT", zone_name: "Store Entrance", visit_count: 0, avg_dwell_seconds: 0, current_occupancy: 0 },
  { zone_id: "LEFT_SHELF", zone_name: "Left Shelf", visit_count: 0, avg_dwell_seconds: 0, current_occupancy: 0 },
  { zone_id: "RIGHT_SHELF", zone_name: "Right Shelf", visit_count: 0, avg_dwell_seconds: 0, current_occupancy: 0 },
  { zone_id: "CENTER_DISPLAY", zone_name: "Center Display", visit_count: 0, avg_dwell_seconds: 0, current_occupancy: 0 },
  { zone_id: "SKINCARE", zone_name: "Skincare Aisle", visit_count: 0, avg_dwell_seconds: 0, current_occupancy: 0 },
  { zone_id: "LIPSTICK", zone_name: "Lipstick Aisle", visit_count: 0, avg_dwell_seconds: 0, current_occupancy: 0 },
  { zone_id: "CASH_COUNTER", zone_name: "Checkout Counter", visit_count: 0, avg_dwell_seconds: 0, current_occupancy: 0 },
];
const EV_CONFIG: Record<string, { badge: string; bg: string; color: string }> = {
  person_detected:   { badge: "ENTRY",    bg: "#ecfdf5", color: "#059669" },
  person_lost:       { badge: "EXIT",     bg: "#fef2f2", color: "#dc2626" },
  zone_enter:        { badge: "ZONE IN",  bg: "#eef2ff", color: "#4f46e5" },
  zone_exit:         { badge: "ZONE OUT", bg: "#f5f3ff", color: "#7c3aed" },
  dwell_time_update: { badge: "DWELL",    bg: "#fffbeb", color: "#d97706" },
  dwell_alert:       { badge: "ALERT",    bg: "#fef3c7", color: "#92400e" },
  anomaly_detected:  { badge: "ANOMALY",  bg: "#fee2e2", color: "#991b1b" },
  crowd_alert:       { badge: "CROWD",    bg: "#fee2e2", color: "#991b1b" },
  ENTRY:             { badge: "ENTRY",    bg: "#ecfdf5", color: "#059669" },
  EXIT:              { badge: "EXIT",     bg: "#fef2f2", color: "#dc2626" },
  ZONE_ENTER:        { badge: "ZONE IN",  bg: "#eef2ff", color: "#4f46e5" },
  ZONE_EXIT:         { badge: "ZONE OUT", bg: "#f5f3ff", color: "#7c3aed" },
  ZONE_DWELL:        { badge: "DWELL",    bg: "#fffbeb", color: "#d97706" },
  REENTRY:           { badge: "RE-ENTRY", bg: "#fef3c7", color: "#92400e" },
  BILLING_QUEUE_JOIN:    { badge: "QUEUE",   bg: "#f0fdf4", color: "#15803d" },
  BILLING_QUEUE_ABANDON: { badge: "ABANDON", bg: "#fef2f2", color: "#dc2626" },
};
const fmtTime = (ts: string) => {
  try { return new Date(ts).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
  catch { return ""; }
};

/* ── Store Selector Dropdown ── */
function StoreSelector({ stores, selected, onChange }: {
  stores: StoreOption[]; selected: string; onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const current = stores.find(s => s.id === selected) ?? stores[0];
  return (
    <div className="store-selector" onClick={() => setOpen(!open)}>
      <Store size={14} />
      <span className="store-name">{current?.name}</span>
      <ChevronDown size={14} className={open ? "flip" : ""} />
      {open && (
        <div className="store-dropdown">
          {stores.map(s => (
            <div key={s.id} className={`store-option ${s.id === selected ? "active" : ""}`}
              onClick={(e) => { e.stopPropagation(); onChange(s.id); setOpen(false); }}>
              <Store size={12} /> {s.name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Heatmap Grid ── */
function HeatmapGrid({ zones }: { zones: HeatmapZone[] }) {
  const getColor = (intensity: number) => {
    if (intensity >= 80) return "#ef4444";
    if (intensity >= 60) return "#f97316";
    if (intensity >= 40) return "#eab308";
    if (intensity >= 20) return "#22c55e";
    return "#3b82f6";
  };
  return (
    <div className="heatmap-grid">
      {zones.map(z => (
        <div className="heatmap-cell" key={z.zone_id} style={{ background: `${getColor(z.intensity)}20`, borderColor: getColor(z.intensity) }}>
          <div className="heatmap-intensity" style={{ background: getColor(z.intensity), width: `${z.intensity}%` }} />
          <div className="heatmap-label">
            <span className="heatmap-zone-name">{z.zone_name}</span>
            <span className="heatmap-stats">{z.visit_frequency} visits · {Math.round(z.avg_dwell_ms / 1000)}s</span>
          </div>
          <span className="heatmap-value" style={{ color: getColor(z.intensity) }}>{z.intensity}</span>
          {!z.data_confidence && <span className="low-confidence" title="Low confidence: fewer than 20 sessions">⚠</span>}
        </div>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════ */
/*  LANDING PAGE                                   */
/* ═══════════════════════════════════════════════ */
function LandingPage({ onStart }: { onStart: (file: File) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [apiOk, setApiOk] = useState(false);

  useEffect(() => {
    fetch("/api/v1/health").then(r => r.json())
      .then(d => { if (d.status === "success") setApiOk(true); })
      .catch(() => {});
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    if (e.dataTransfer.files?.length) setFile(e.dataTransfer.files[0]);
  };

  return (
    <div className="landing">
      {/* Background blobs */}
      <div className="landing-blob blob-1" />
      <div className="landing-blob blob-2" />
      <div className="landing-blob blob-3" />

      <div className="landing-content">
        {/* Header */}
        <nav className="landing-nav">
          <div className="landing-logo">
            <div className="logo-icon-lg"><Zap size={20} /></div>
            <span>Store Intelligence</span>
          </div>
          <div className={`status-chip ${apiOk ? "ok" : ""}`}>
            <span className="status-dot-sm" />
            {apiOk ? "API Connected" : "Connecting…"}
          </div>
        </nav>

        {/* Hero */}
        <div className="landing-hero">
          <div className="hero-badge">
            <Cpu size={13} /> AI-Powered Retail Analytics
          </div>
          <h1 className="hero-title">
            Turn CCTV footage into<br />
            <span className="hero-gradient">real-time store insights</span>
          </h1>
          <p className="hero-subtitle">
            Drop a video below to detect visitors, track movement across zones,
            measure dwell times, and monitor queue activity — all in real time.
          </p>
        </div>

        {/* Upload Card */}
        <div className="upload-card">
          <div
            className={`drop-area ${dragging ? "active" : ""} ${file ? "has-file" : ""}`}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => document.getElementById("landing-file")?.click()}
          >
            {file ? (
              <div className="file-preview">
                <div className="file-icon-wrap">
                  <FileVideo size={28} />
                </div>
                <div className="file-details">
                  <span className="file-name-lg">{file.name}</span>
                  <span className="file-size">{(file.size / (1024 * 1024)).toFixed(1)} MB</span>
                </div>
                <button className="file-remove" onClick={e => { e.stopPropagation(); setFile(null); }}>
                  <X size={16} />
                </button>
              </div>
            ) : (
              <div className="drop-prompt">
                <div className="drop-icon">
                  <UploadCloud size={32} />
                </div>
                <p className="drop-title">Drag & drop your CCTV video</p>
                <p className="drop-hint">or click to browse · MP4, AVI, MOV</p>
              </div>
            )}
            <input id="landing-file" type="file" accept="video/*" hidden
              onChange={e => { if (e.target.files?.length) setFile(e.target.files[0]); }} />
          </div>

          <button
            className="launch-btn"
            disabled={!file || !apiOk}
            onClick={() => file && onStart(file)}
          >
            <MonitorPlay size={18} />
            Start Analysis
            <ArrowRight size={16} />
          </button>
        </div>

        {/* Features */}
        <div className="feature-row">
          {[
            { icon: <Eye size={20} />, title: "Live Detection", desc: "YOLOv8 person detection with real-time bounding boxes" },
            { icon: <MapPin size={20} />, title: "Zone Tracking", desc: "Track visitors across entrance, aisles, trial area & checkout" },
            { icon: <BarChart3 size={20} />, title: "Instant Analytics", desc: "Conversion funnels, dwell times, footfall charts — live" },
            { icon: <Shield size={20} />, title: "Anomaly Detection", desc: "Crowd spikes, queue buildup, and dead zone alerts" },
          ].map(f => (
            <div className="feature-card" key={f.title}>
              <div className="feature-icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════ */
/*  DASHBOARD                                      */
/* ═══════════════════════════════════════════════ */
function Dashboard({ videoFile, onStop }: { videoFile: File; onStop: () => void }) {
  const [apiOnline, setApiOnline] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [wsLive, setWsLive] = useState(false);
  const [eventCount, setEventCount] = useState(0);
  const [selectedStore, setSelectedStore] = useState(STORES[0].id);
  const wsRef = useRef<WebSocket | null>(null);
  const started = useRef(false);
  const stoppingRef = useRef(false);

  const [stats, setStats] = useState<Stats>({
    visitors: 0, exits: 0, conversion: 0, dwell: "0s", queueDepth: 0, anomalies: 0, abandonmentRate: 0,
  });
  const [events, setEvents] = useState<EventItem[]>([]);
  const [funnel, setFunnel] = useState<FunnelStage[]>([]);
  const [zones, setZones] = useState<ZoneData[]>([]);
  const [heatmap, setHeatmap] = useState<HeatmapZone[]>([]);
  const [footfall, setFootfall] = useState<FootfallPoint[]>([]);
  const [anomalyList, setAnomalyList] = useState<AnomalyItem[]>([]);
  const checkoutRef = useRef(0);
  const totalRef = useRef(0);

  /* ── Health ── */
  useEffect(() => {
    const poll = () => {
      fetch("/api/v1/health").then(r => r.json())
        .then(d => { if (d.status === "success") { setApiOnline(true); setEventCount(d.data?.total_events ?? 0); } })
        .catch(() => setApiOnline(false));
    };
    poll();
    const id = setInterval(poll, 8000);
    return () => { clearInterval(id); wsRef.current?.close(); };
  }, []);

  /* ── Spec Endpoint Polling (multi-store) ── */
  useEffect(() => {
    if (!processing && eventCount === 0) return;
    const poll = async () => {
      try {
        const [metricsR, funnelR, heatmapR, anomaliesR] = await Promise.allSettled([
          fetch(`/api/v1/stores/${selectedStore}/metrics`).then(r => r.json()),
          fetch(`/api/v1/stores/${selectedStore}/funnel`).then(r => r.json()),
          fetch(`/api/v1/stores/${selectedStore}/heatmap`).then(r => r.json()),
          fetch(`/api/v1/stores/${selectedStore}/anomalies`).then(r => r.json()),
        ]);
        if (metricsR.status === "fulfilled" && metricsR.value?.status === "success") {
          const m = metricsR.value.data;
          setStats(prev => ({
            ...prev,
            visitors: m.unique_visitors ?? prev.visitors,
            conversion: Math.round((m.conversion_rate ?? 0) * 100),
            queueDepth: m.queue_depth ?? prev.queueDepth,
            abandonmentRate: Math.round((m.abandonment_rate ?? 0) * 100),
          }));
        }
        if (funnelR.status === "fulfilled" && funnelR.value?.status === "success") {
          setFunnel(funnelR.value.data ?? []);
        }
        if (heatmapR.status === "fulfilled" && heatmapR.value?.status === "success") {
          setHeatmap(heatmapR.value.data ?? []);
        }
        if (anomaliesR.status === "fulfilled" && anomaliesR.value?.status === "success") {
          const raw = anomaliesR.value.data ?? [];
          setAnomalyList(raw.map((a: any) => ({
            id: a.anomaly_id ?? crypto.randomUUID(),
            title: a.type ?? "Unknown",
            detail: a.suggested_action ?? a.zone_id ?? "",
            severity: (a.severity ?? "INFO").toLowerCase(),
            suggested_action: a.suggested_action,
          })));
        }
      } catch {}
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [processing, eventCount, selectedStore]);

  /* ── Legacy Analytics (zone occupancy, footfall) ── */
  useEffect(() => {
    if (!processing) return;
    const poll = async () => {
      try {
        const [zR, ffR] = await Promise.all([
          fetch("/api/v1/analytics/zones").then(r => r.json()),
          fetch("/api/v1/analytics/footfall").then(r => r.json()),
        ]);
        if (zR.status === "success") setZones(zR.data ?? []);
        if (ffR.status === "success") setFootfall((ffR.data ?? []).filter((p: FootfallPoint) => p.count > 0));
      } catch {}
    };
    poll();
    const id = setInterval(poll, 4000);
    return () => clearInterval(id);
  }, [processing]);

  /* ── Event handler ── */
  const handleEvent = useCallback((evt: Record<string, unknown>) => {
    if (!evt?.event_type) return;
    const type = evt.event_type as string;
    const meta = (evt.metadata ?? {}) as Record<string, unknown>;
    const vid = evt.visitor_id ? String(evt.visitor_id) : evt.track_id ? `VIS_${String(evt.track_id).padStart(4, "0")}` : "";
    const zoneId = (evt.zone_id ?? "") as string;
    const zoneName = (meta.zone_name ?? zoneId) as string;
    const ts = (evt.timestamp ?? new Date().toISOString()) as string;
    const cfg = EV_CONFIG[type];

    switch (type) {
      case "person_detected": case "ENTRY":
        totalRef.current++; setStats(p => ({ ...p, visitors: p.visitors + 1 })); break;
      case "person_lost": case "EXIT":
        setStats(p => ({ ...p, visitors: Math.max(0, p.visitors - 1), exits: p.exits + 1 })); break;
      case "REENTRY":
        setStats(p => ({ ...p, visitors: p.visitors + 1 })); break;
      case "zone_enter": case "ZONE_ENTER":
        if (zoneId === "checkout" || zoneId === "CASH_COUNTER" || zoneId === "BILLING") { checkoutRef.current++; setStats(p => ({ ...p, queueDepth: p.queueDepth + 1 })); } break;
      case "zone_exit": case "ZONE_EXIT":
        if (zoneId === "checkout" || zoneId === "CASH_COUNTER" || zoneId === "BILLING") {
          const c = totalRef.current > 0 ? Math.round((checkoutRef.current / totalRef.current) * 100) : 0;
          setStats(p => ({ ...p, queueDepth: Math.max(0, p.queueDepth - 1), conversion: c }));
        } break;
      case "dwell_time_update": case "ZONE_DWELL": {
        const s = meta.dwell_seconds as number | undefined;
        const ms = evt.dwell_ms as number | undefined;
        const val = s ?? (ms ? ms / 1000 : undefined);
        if (val != null) setStats(p => ({ ...p, dwell: val >= 60 ? `${Math.round(val / 60)}m ${Math.round(val % 60)}s` : `${Math.round(val)}s` }));
        break;
      }
      case "BILLING_QUEUE_JOIN":
        setStats(p => ({ ...p, queueDepth: p.queueDepth + 1 })); break;
      case "BILLING_QUEUE_ABANDON":
        setStats(p => ({ ...p, queueDepth: Math.max(0, p.queueDepth - 1), anomalies: p.anomalies + 1 })); break;
      case "anomaly_detected": case "crowd_alert": case "dwell_alert":
        setStats(p => ({ ...p, anomalies: p.anomalies + 1 }));
        setAnomalyList(prev => [{
          id: crypto.randomUUID(),
          title: type === "crowd_alert" ? "Crowd Spike" : type === "dwell_alert" ? "Dwell Alert" : "Anomaly",
          detail: zoneName ? `Zone: ${zoneName}` : "Store-wide",
          severity: type === "crowd_alert" ? "critical" : "warn",
        }, ...prev].slice(0, 8)); break;
      case "pipeline_stopped":
        if (!stoppingRef.current) {
          stoppingRef.current = true;
          setTimeout(() => {
            setProcessing(false);
            wsRef.current?.close();
            onStop();
          }, 1500);
        }
        break;
    }
    if (cfg) {
      let text = "";
      if (type.includes("zone") || type.includes("ZONE")) text = zoneName || zoneId;
      else if (type === "dwell_time_update" || type === "ZONE_DWELL") text = `${meta.dwell_seconds ?? Math.round((evt.dwell_ms as number ?? 0) / 1000)}s in ${zoneName || zoneId}`;
      else if (type === "BILLING_QUEUE_JOIN") text = `Joined queue (depth: ${meta.queue_depth ?? "?"})`;
      else if (type === "BILLING_QUEUE_ABANDON") text = `Abandoned queue`;
      else if (type === "REENTRY") text = "Re-entered store";
      setEvents(prev => [{
        id: crypto.randomUUID(), time: fmtTime(ts),
        badge: cfg.badge, badgeBg: cfg.bg, badgeColor: cfg.color, text, vid,
      }, ...prev].slice(0, 50));
    }
  }, [onStop]);

  /* ── WebSocket ── */
  const connectWs = useCallback(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${window.location.host}/api/v1/events/stream`);
    ws.onopen = () => setWsLive(true);
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "backfill" && Array.isArray(msg.events)) msg.events.forEach(handleEvent);
        else if (msg.type === "event" && msg.event) handleEvent(msg.event);
      } catch {}
    };
    ws.onerror = () => setWsLive(false);
    ws.onclose = () => { setWsLive(false); wsRef.current = null; };
    wsRef.current = ws;
  }, [handleEvent]);

  /* ── Auto-start ── */
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    const go = async () => {
      const fd = new FormData();
      fd.append("file", videoFile);
      try {
        const r = await fetch("/api/v1/cameras/cam-1/start", { method: "POST", body: fd });
        if (r.ok) { setProcessing(true); connectWs(); }
      } catch (e) { console.error(e); }
    };
    go();
  }, [videoFile, connectWs]);

  const handleStop = async () => {
    try {
      await fetch("/api/v1/cameras/cam-1/stop", { method: "POST" });
      setProcessing(false);
      wsRef.current?.close();
      onStop();
    } catch (e) { console.error(e); }
  };

  const funnelMax = funnel.length > 0 ? Math.max(...funnel.map(f => f.count), 1) : 1;
  const funnelColors = ["#3B82F6", "#8B5CF6", "#F59E0B", "#10B981"];
  const sortedZones = (zones.length > 0 ? zones : DEFAULT_ZONES)
    .slice().sort((a, b) => ZONE_ORDER.indexOf(a.zone_id) - ZONE_ORDER.indexOf(b.zone_id));

  const kpis = [
    { label: "Visitors", val: stats.visitors, icon: <Users size={14} />, bg: "#eef2ff", color: "#4f46e5" },
    { label: "Conversion", val: `${stats.conversion}%`, icon: <ShoppingCart size={14} />, bg: "#ecfdf5", color: "#059669" },
    { label: "Avg Dwell", val: stats.dwell, icon: <Clock size={14} />, bg: "#f5f3ff", color: "#7c3aed" },
    { label: "Queue", val: stats.queueDepth, icon: <Activity size={14} />, bg: "#fffbeb", color: "#d97706" },
    { label: "Anomalies", val: stats.anomalies, icon: <AlertTriangle size={14} />, bg: "#fef2f2", color: "#dc2626" },
    { label: "Exits", val: stats.exits, icon: <LogOut size={14} />, bg: "#ecfeff", color: "#0891b2" },
  ];

  return (
    <div className="dashboard">
      <header className="card header">
        <div className="header-left">
          <div className="header-logo">
            <div className="logo-icon"><Zap size={16} /></div>
            Store Intelligence
          </div>
          <StoreSelector stores={STORES} selected={selectedStore} onChange={setSelectedStore} />
          <span className={`status-pill ${apiOnline ? "online" : "offline"}`}>
            <span className="status-dot" />{apiOnline ? "Online" : "Offline"}
          </span>
          {processing && <span className={`status-pill ${wsLive ? "live" : "offline"}`}><span className="status-dot" />{wsLive ? "Live" : "…"}</span>}
          <span className="header-meta">{eventCount > 0 && <>{eventCount.toLocaleString()} events · </>}{selectedStore}</span>
        </div>
        <div className="header-controls">
          <div className="now-playing"><FileVideo size={14} /> {videoFile.name}</div>
          <button className="btn btn-stop" onClick={handleStop}><Square size={12} /> Stop Analysis</button>
        </div>
      </header>

      <div className="main-grid">
        {/* ── Left Column: Video + KPIs + Stats ── */}
        <div className="left-col">
          <div className="card video-panel">
            {processing ? (
              <><img src="/api/v1/cameras/cam-1/stream" alt="Live" /><div className="video-badge"><span className="rec-dot" /> LIVE</div></>
            ) : (
              <div className="video-placeholder"><Video size={44} /><p>Starting…</p></div>
            )}
          </div>

          {/* KPI Cards */}
          <div className="kpi-grid">
            {kpis.map(k => (
              <div className="card kpi-card" key={k.label}>
                <div className="kpi-label"><span className="kpi-icon" style={{ background: k.bg, color: k.color }}>{k.icon}</span>{k.label}</div>
                <div className="kpi-value">{k.val}</div>
              </div>
            ))}
          </div>

          {/* Processing Info */}
          <div className="card process-strip">
            <div className="strip-item">
              <Cpu size={14} />
              <span className="strip-label">Source</span>
              <span className="strip-val">{videoFile.name}</span>
            </div>
            <div className="strip-item">
              <Activity size={14} />
              <span className="strip-label">Events</span>
              <span className="strip-val">{eventCount.toLocaleString()}</span>
            </div>
            <div className="strip-item">
              <MapPin size={14} />
              <span className="strip-label">Zones</span>
              <span className="strip-val">{sortedZones.filter(z => z.current_occupancy > 0).length} / {sortedZones.length} active</span>
            </div>
            <div className="strip-item">
              <Target size={14} />
              <span className="strip-label">Abandon</span>
              <span className="strip-val">{stats.abandonmentRate}%</span>
            </div>
          </div>
        </div>

        {/* ── Right Column: Funnel + Events ── */}
        <div className="right-col">
          <div className="card funnel-panel">
            <div className="section-title"><TrendingUp size={14} /> Conversion Funnel</div>
            {(funnel.length > 0 ? funnel : [
              { stage: "Entry", count: stats.visitors, percentage: 100, drop_off_pct: 0 },
              { stage: "Zone Visit", count: 0, percentage: 0, drop_off_pct: 0 },
              { stage: "Billing Queue", count: 0, percentage: 0, drop_off_pct: 0 },
              { stage: "Purchase", count: checkoutRef.current, percentage: stats.conversion, drop_off_pct: 0 },
            ]).map((f, i) => (
              <div className="funnel-row" key={f.stage}>
                <span className="funnel-label">{f.stage.charAt(0).toUpperCase() + f.stage.slice(1)}</span>
                <div className="funnel-track"><div className="funnel-fill" style={{ width: `${Math.max((f.count / funnelMax) * 100, 2)}%`, background: funnelColors[i % 4] }} /></div>
                <span className="funnel-stat">{f.count} <span className="funnel-pct">({Math.round(f.percentage)}%)</span></span>
                {f.drop_off_pct > 0 && <span className="funnel-drop">↓{Math.round(f.drop_off_pct)}%</span>}
              </div>
            ))}
          </div>

          <div className="card event-panel">
            <div className="section-title"><CircleDot size={14} /> Live Events</div>
            <div className="event-scroll">
              {events.length === 0 ? (
                <div className="empty-state"><Activity size={24} /> Waiting for events…</div>
              ) : events.map(ev => (
                <div className="event-row" key={ev.id}>
                  <span className="ev-time">{ev.time}</span>
                  <span className="ev-badge" style={{ background: ev.badgeBg, color: ev.badgeColor }}>{ev.badge}</span>
                  <span className="ev-text">{ev.text}</span>
                  {ev.vid && <span className="ev-vid">{ev.vid}</span>}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="bottom-grid">
        <div className="card zone-panel">
          <div className="section-title"><MapPin size={14} /> Zone Occupancy</div>
          <div className="zone-list">
            {sortedZones.map(z => {
              const col = ZONE_COLORS[z.zone_id] ?? "#6366F1";
              return (
                <div className="zone-row" key={z.zone_id}>
                  <span className="zone-dot" style={{ background: col }} />
                  <div className="zone-info"><div className="zone-name">{z.zone_name}</div><div className="zone-meta">{z.visit_count} visits · {Math.round(z.avg_dwell_seconds)}s avg</div></div>
                  <span className="zone-occ" style={{ color: col }}>{z.current_occupancy}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="card chart-panel">
          <div className="section-title"><Zap size={14} /> Footfall Timeline</div>
          <ResponsiveContainer width="100%" height={210}>
            <AreaChart data={footfall.length > 0 ? footfall : [{ period: "Now", count: 0 }]}>
              <defs><linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#6366F1" stopOpacity={0.2} /><stop offset="100%" stopColor="#6366F1" stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="period" axisLine={false} tickLine={false} />
              <YAxis axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 8, fontSize: 12, boxShadow: "0 2px 8px rgba(0,0,0,0.08)" }} />
              <Area type="monotone" dataKey="count" stroke="#6366F1" strokeWidth={2} fill="url(#areaGrad)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* ── Heatmap Panel (from spec endpoint) ── */}
        {heatmap.length > 0 ? (
          <div className="card heatmap-panel">
            <div className="section-title"><Flame size={14} /> Zone Heatmap</div>
            <HeatmapGrid zones={heatmap} />
          </div>
        ) : (
          <div className="card anomaly-panel">
            <div className="section-title"><Shield size={14} /> Active Anomalies</div>
            <div className="anomaly-list">
              {anomalyList.length === 0 ? (
                <div className="empty-state"><Shield size={24} /> No anomalies</div>
              ) : anomalyList.map(a => (
                <div className={`anomaly-card ${a.severity}`} key={a.id}>
                  <div className="anomaly-head"><span className="anomaly-title">{a.title}</span><span className={`sev-badge ${a.severity}`}>{a.severity}</span></div>
                  <span className="anomaly-desc">{a.detail}</span>
                  {a.suggested_action && <span className="anomaly-action">{a.suggested_action}</span>}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════ */
/*  APP ROOT                                       */
/* ═══════════════════════════════════════════════ */
export default function App() {
  const [view, setView] = useState<"landing" | "dashboard">("landing");
  const [videoFile, setVideoFile] = useState<File | null>(null);

  const handleStart = (file: File) => {
    setVideoFile(file);
    setView("dashboard");
  };

  const handleStop = () => {
    setView("landing");
    setVideoFile(null);
  };

  if (view === "dashboard" && videoFile) {
    return <Dashboard videoFile={videoFile} onStop={handleStop} />;
  }
  return <LandingPage onStart={handleStart} />;
}
