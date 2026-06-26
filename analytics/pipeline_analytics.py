"""
analytics/pipeline_analytics.py

Complete analytics engine for pipeline inspection sessions.
Reads a session's detections.csv and produces:

  1.  Unique defect counts (by class)
  2.  Pipeline Integrity Index (0-100 score)
  3.  Severity breakdown
  4.  Zone-based risk analysis
  5.  Defect size distribution
  6.  Activity timeline (defects over time)
  7.  Defect heatmap
  8.  Anomaly detection (IsolationForest)
  9.  Maintenance alerts (rule-based)
  10. Comparative trend (reads sessions_summary.csv)
  11. Per-defect profile table
  12. All charts as Plotly JSON (sent to frontend)

Color palette is designed for the dark slate background
(slate-950 / indigo theme from the frontend).
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import IsolationForest

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── palette (dark theme) ──────────────────────────────────────
C_CRACK       = "#f87171"   # red-400
C_CORROSION   = "#fb923c"   # orange-400
C_RUST        = "#facc15"   # yellow-400
C_DEFORMATION = "#e879f9"   # fuchsia-400
C_CRITICAL    = "#ef4444"   # red-500
C_HIGH        = "#f97316"   # orange-500
C_MODERATE    = "#eab308"   # yellow-500
C_LOW         = "#22c55e"   # green-500
C_ACCENT      = "#6366f1"   # indigo-500
C_GOOD        = "#10b981"   # emerald-500
C_DEGRADED    = "#f59e0b"   # amber-500
C_BG          = "#020617"   # slate-950
C_SURFACE     = "#0f172a"   # slate-900
C_BORDER      = "#1e293b"   # slate-800
C_TEXT        = "#e2e8f0"   # slate-200
C_MUTED       = "#64748b"   # slate-500

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C_BG,
    plot_bgcolor=C_SURFACE,
    font=dict(family="Plus Jakarta Sans, sans-serif", color=C_TEXT, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER, linecolor=C_BORDER),
    yaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER, linecolor=C_BORDER),
)

CLASS_COLORS = {
    "crack": C_CRACK,
    "corrosion": C_CORROSION,
    "rust": C_RUST,
    "deformation": C_DEFORMATION,
}

SEVERITY_COLORS = {
    "CRITICAL": C_CRITICAL,
    "HIGH": C_HIGH,
    "MODERATE": C_MODERATE,
    "LOW": C_LOW,
}


# ─────────────────────────────────────────────────────────────
# INTEGRITY INDEX
# ─────────────────────────────────────────────────────────────

def compute_integrity_index(unique_defects: dict, anomaly_count: int = 0) -> dict:
    """
    Compute a 0-100 pipeline integrity score.
    Lower score = worse pipeline health.
    """
    score = 100.0

    score -= unique_defects.get("crack", 0) * 3.0
    score -= unique_defects.get("corrosion", 0) * 2.0
    score -= unique_defects.get("rust", 0) * 1.0
    score -= unique_defects.get("deformation", 0) * 4.0
    score -= anomaly_count * 10.0

    score = max(0.0, min(100.0, score))

    if score >= 80:
        status, color = "GOOD", C_GOOD
    elif score >= 50:
        status, color = "DEGRADED", C_DEGRADED
    else:
        status, color = "CRITICAL", C_CRITICAL

    return {"score": round(score, 1), "status": status, "color": color}


# ─────────────────────────────────────────────────────────────
# RULE-BASED ALERTS
# ─────────────────────────────────────────────────────────────

def generate_alerts(unique_defects: dict, integrity: dict,
                    zone_df: pd.DataFrame, df: pd.DataFrame) -> list:
    alerts = []

    def add(alert_type, message, severity="WARNING", action=True):
        alerts.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "action_required": action,
        })

    if unique_defects.get("crack", 0) > 10:
        add("STRUCTURAL", f"High crack count ({unique_defects['crack']} unique cracks). Structural inspection required.", "CRITICAL")

    if unique_defects.get("crack", 0) > 0:
        crit_cracks = df[(df["class_name"] == "crack") & (df["severity"] == "CRITICAL")]["track_id"].nunique()
        if crit_cracks > 0:
            add("CRITICAL_DEFECT", f"{crit_cracks} CRITICAL severity crack(s) detected. Immediate action required.", "CRITICAL")

    if integrity["score"] < 50:
        add("INTEGRITY", f"Pipeline Integrity Index is {integrity['score']}/100 (CRITICAL). Urgent maintenance required.", "CRITICAL")
    elif integrity["score"] < 80:
        add("INTEGRITY", f"Pipeline Integrity Index is {integrity['score']}/100 (DEGRADED). Schedule maintenance.", "WARNING", False)

    if unique_defects.get("deformation", 0) > 0:
        add("PHYSICAL_DAMAGE", f"{unique_defects['deformation']} deformation(s) detected. Physical damage confirmed.", "HIGH")

    if not zone_df.empty and "crack" in zone_df.columns and "debris" not in zone_df.columns:
        try:
            worst_zone = zone_df["crack"].idxmax()
            count = int(zone_df.loc[worst_zone, "crack"])
            if count > 5:
                add("ZONE_RISK", f"Zone {worst_zone} has highest crack concentration ({count} cracks). Prioritise this section.", "HIGH")
        except Exception:
            pass

    total = sum(unique_defects.values())
    if total == 0:
        alerts.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": "CLEAR",
            "message": "No defects detected in this session. Pipeline appears healthy.",
            "severity": "OK",
            "action_required": False,
        })

    return alerts


# ─────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────

def run_anomaly_detection(df: pd.DataFrame, fps: float = 30.0) -> dict:
    """
    Groups detections by second, builds a per-second feature vector,
    runs IsolationForest, and returns flagged anomalous seconds.

    Returns empty result if not enough data to train.
    """
    if df.empty:
        return {"anomalies": [], "anomaly_count": 0, "chart": None}

    df = df.copy()
    df["second"] = (df["frame_number"] / fps).astype(int)

    # Per-second feature matrix
    per_second = df.groupby("second").agg(
        total=("track_id", "count"),
        crack=("class_name", lambda x: (x == "crack").sum()),
        corrosion=("class_name", lambda x: (x == "corrosion").sum()),
        rust=("class_name", lambda x: (x == "rust").sum()),
        deformation=("class_name", lambda x: (x == "deformation").sum()),
        critical=("severity", lambda x: (x == "CRITICAL").sum()),
        avg_conf=("confidence", "mean"),
    ).reset_index()

    if len(per_second) < 10:
        return {"anomalies": [], "anomaly_count": 0, "chart": None}

    features = per_second[["total", "crack", "corrosion", "rust", "deformation", "critical", "avg_conf"]].values

    model = IsolationForest(contamination=0.1, random_state=42)
    labels = model.fit_predict(features)

    per_second["anomaly"] = labels == -1
    anomalous = per_second[per_second["anomaly"]]

    # Chart: total detections per second, anomalies highlighted
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=per_second["second"],
        y=per_second["total"],
        mode="lines",
        name="Detections/sec",
        line=dict(color=C_ACCENT, width=2),
        fill="tozeroy",
        opacity=0.08,
    ))

    if not anomalous.empty:
        fig.add_trace(go.Scatter(
            x=anomalous["second"],
            y=anomalous["total"],
            mode="markers",
            name="Anomaly",
            marker=dict(color=C_CRITICAL, size=10, symbol="x",
                        line=dict(color=C_CRITICAL, width=2)),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Anomaly Detection — Detection Rate per Second", font=dict(size=13)),
        xaxis_title="Time (seconds)",
        yaxis_title="Detections",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER),
        showlegend=True,
    )

    return {
        "anomalies": anomalous[["second", "total", "crack", "corrosion"]].to_dict("records"),
        "anomaly_count": len(anomalous),
        "chart": json.loads(fig.to_json()),
    }


# ─────────────────────────────────────────────────────────────
# CHART 1 — Defect counts (unique track_id per class)
# ─────────────────────────────────────────────────────────────

def chart_defect_counts(unique_defects: dict) -> dict:
    if not unique_defects:
        return {}
    classes = list(unique_defects.keys())
    counts = list(unique_defects.values())
    colors = [CLASS_COLORS.get(c, C_ACCENT) for c in classes]

    fig = go.Figure(go.Bar(
        x=classes,
        y=counts,
        marker=dict(
            color=colors,
            line=dict(color=colors, width=1),
        ),
        text=counts,
        textposition="outside",
        textfont=dict(color=C_TEXT, size=13, family="JetBrains Mono"),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Unique Defect Counts (by Track ID)", font=dict(size=13)),
        xaxis_title="Defect Class",
        yaxis_title="Unique Count",
        bargap=0.35,
    )
    return json.loads(fig.to_json())


# ─────────────────────────────────────────────────────────────
# CHART 2 — Severity breakdown (donut)
# ─────────────────────────────────────────────────────────────

def chart_severity_donut(df: pd.DataFrame) -> dict:
    # Use max severity per unique track_id
    if df.empty or "track_id" not in df.columns:
        return {}

    severity_order = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}
    df2 = df.copy()
    df2["sev_rank"] = df2["severity"].map(severity_order).fillna(0)
    per_defect = df2.sort_values("sev_rank", ascending=False).groupby("track_id").first().reset_index()
    sev_counts = per_defect["severity"].value_counts()

    labels = sev_counts.index.tolist()
    values = sev_counts.values.tolist()
    colors = [SEVERITY_COLORS.get(s, C_MUTED) for s in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color=C_BG, width=3)),
        textfont=dict(size=12, color=C_TEXT),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Severity Distribution (per Unique Defect)", font=dict(size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER,
                    orientation="h", yanchor="bottom", y=-0.2),
        showlegend=True,
        annotations=[dict(
            text=f"<b>{sum(values)}</b><br>defects",
            x=0.5, y=0.5, font_size=14, showarrow=False,
            font=dict(color=C_TEXT),
        )],
    )
    return json.loads(fig.to_json())


# ─────────────────────────────────────────────────────────────
# CHART 3 — Zone analysis (grouped bar)
# ─────────────────────────────────────────────────────────────

def chart_zone_analysis(df: pd.DataFrame) -> tuple:
    """Returns (chart_json, zone_summary_df)"""
    if df.empty:
        return {}, pd.DataFrame()

    zone_df = df.groupby(["zone", "class_name"])["track_id"].nunique().unstack(fill_value=0)

    fig = go.Figure()
    for cls in zone_df.columns:
        fig.add_trace(go.Bar(
            name=cls,
            x=zone_df.index.tolist(),
            y=zone_df[cls].tolist(),
            marker=dict(color=CLASS_COLORS.get(cls, C_ACCENT),
                        line=dict(color=C_BG, width=1)),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Zone-Based Defect Distribution", font=dict(size=13)),
        barmode="group",
        xaxis_title="Pipeline Zone",
        yaxis_title="Unique Defect Count",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER),
        bargap=0.2,
        bargroupgap=0.05,
    )
    return json.loads(fig.to_json()), zone_df.reset_index()


# ─────────────────────────────────────────────────────────────
# CHART 4 — Activity timeline
# ─────────────────────────────────────────────────────────────

def chart_activity_timeline(df: pd.DataFrame, fps: float = 30.0) -> dict:
    if df.empty:
        return {}

    df2 = df.copy()
    df2["second"] = (df2["frame_number"] / fps).astype(int)

    timeline = df2.groupby(["second", "class_name"]).size().unstack(fill_value=0).reset_index()

    fig = go.Figure()
    for cls in [c for c in timeline.columns if c != "second"]:
        fig.add_trace(go.Scatter(
            x=timeline["second"],
            y=timeline[cls],
            mode="lines",
            name=cls,
            line=dict(color=CLASS_COLORS.get(cls, C_ACCENT), width=2),
            fill="tozeroy",
            fillcolor=CLASS_COLORS.get(cls, C_ACCENT),
            opacity=0.08,
            hovertemplate=f"<b>{cls}</b><br>Second: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Defect Activity Timeline (Detections per Second)", font=dict(size=13)),
        xaxis_title="Time (seconds)",
        yaxis_title="Detections",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER),
        hovermode="x unified",
    )
    return json.loads(fig.to_json())


# ─────────────────────────────────────────────────────────────
# CHART 5 — Defect size distribution (box plot per class)
# ─────────────────────────────────────────────────────────────

def chart_defect_size_distribution(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    # Use max area per unique defect
    per_defect = df.groupby(["track_id", "class_name"])["area"].max().reset_index()

    fig = go.Figure()
    for cls in per_defect["class_name"].unique():
        subset = per_defect[per_defect["class_name"] == cls]["area"]
        fig.add_trace(go.Box(
            y=subset,
            name=cls,
            marker_color=CLASS_COLORS.get(cls, C_ACCENT),
            line_color=CLASS_COLORS.get(cls, C_ACCENT),
            fillcolor=CLASS_COLORS.get(cls, C_ACCENT),
            opacity=0.15,
            boxmean="sd",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Defect Size Distribution (Bounding Box Area px²)", font=dict(size=13)),
        yaxis_title="Area (px²)",
        xaxis_title="Defect Class",
        showlegend=False,
    )
    return json.loads(fig.to_json())


# ─────────────────────────────────────────────────────────────
# CHART 6 — Confidence distribution (violin)
# ─────────────────────────────────────────────────────────────

def chart_confidence_distribution(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    fig = go.Figure()
    for cls in df["class_name"].unique():
        subset = df[df["class_name"] == cls]["confidence"]
        fig.add_trace(go.Violin(
            y=subset,
            name=cls,
            box_visible=True,
            meanline_visible=True,
            fillcolor=CLASS_COLORS.get(cls, C_ACCENT),
            line_color=CLASS_COLORS.get(cls, C_ACCENT),
            opacity=0.3,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Model Confidence Distribution by Class", font=dict(size=13)),
        yaxis_title="Confidence Score",
        xaxis_title="Class",
        showlegend=False,
    )
    return json.loads(fig.to_json())


# ─────────────────────────────────────────────────────────────
# CHART 7 — Heatmap (spatial density)
# ─────────────────────────────────────────────────────────────

def chart_heatmap(df: pd.DataFrame, class_name: str,
                  fw: int = 640, fh: int = 640) -> dict:
    if df.empty:
        return {}

    subset = df[df["class_name"] == class_name]
    if subset.empty:
        return {}

    grid = np.zeros((fh, fw), dtype=float)
    for _, row in subset.iterrows():
        x = int(np.clip(row["center_x"], 0, fw - 1))
        y = int(np.clip(row["center_y"], 0, fh - 1))
        grid[y, x] += 1

    grid = gaussian_filter(grid, sigma=20)

    fig = go.Figure(go.Heatmap(
        z=grid,
        colorscale=[
            [0.0,  C_BG],
            [0.25, "#1e1b4b"],
            [0.5,  "#4338ca"],
            [0.75, "#f97316"],
            [1.0,  "#ef4444"],
        ],
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=C_TEXT),
            outlinecolor=C_BORDER,
        ),
        hovertemplate="x: %{x}<br>y: %{y}<br>density: %{z:.2f}<extra></extra>",
    ))

    # Add zone dividers
    third = fw // 3
    for x_pos, label in [(third, "Zone A│B"), (2 * third, "Zone B│C")]:
        fig.add_vline(x=x_pos, line=dict(color=C_MUTED, width=1, dash="dot"))

    fig.update_layout(
        paper_bgcolor=C_BG,
        plot_bgcolor=C_SURFACE,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=C_TEXT, size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        title=dict(text=f"{class_name.capitalize()} Spatial Density Heatmap", font=dict(size=13)),
        xaxis=dict(title="Frame Width (px)", gridcolor=C_BORDER, linecolor=C_BORDER),
        yaxis=dict(title="Frame Height (px)", autorange="reversed",
                   gridcolor=C_BORDER, linecolor=C_BORDER),
    )
    return json.loads(fig.to_json())


# ─────────────────────────────────────────────────────────────
# CHART 8 — Integrity index gauge
# ─────────────────────────────────────────────────────────────

def chart_integrity_gauge(integrity: dict) -> dict:
    score = integrity["score"]
    color = integrity["color"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 80, "increasing": {"color": C_GOOD}, "decreasing": {"color": C_CRITICAL}},
        number={"suffix": "/100", "font": {"size": 36, "color": C_TEXT}},
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickcolor=C_TEXT,
                tickfont=dict(color=C_TEXT),
            ),
            bar=dict(color=color, thickness=0.25),
            bgcolor=C_SURFACE,
            bordercolor=C_BORDER,
            steps=[
                dict(range=[0, 50],  color="#1f0606"),
                dict(range=[50, 80], color="#1a1500"),
                dict(range=[80, 100], color="#061a0e"),
            ],
            threshold=dict(
                line=dict(color=C_TEXT, width=2),
                thickness=0.75,
                value=80,
            ),
        ),
        title=dict(
            text=f"Pipeline Integrity Index<br><span style='font-size:14px;color:{color}'>{integrity['status']}</span>",
            font=dict(size=14, color=C_TEXT),
        ),
    ))

    fig.update_layout(
        paper_bgcolor=C_BG,
        plot_bgcolor=C_SURFACE,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=C_TEXT, size=12),
        margin=dict(l=30, r=30, t=80, b=20),
    )
    return json.loads(fig.to_json())


# ─────────────────────────────────────────────────────────────
# CHART 9 — Comparative trend (across sessions)
# ─────────────────────────────────────────────────────────────

def chart_comparative_trend(sessions_summary_path: str, module: str = "pipeline") -> dict:
    if not os.path.exists(sessions_summary_path):
        return {}

    df = pd.read_csv(sessions_summary_path)
    df = df[df["module"] == module].copy()

    if len(df) < 2:
        return {}

    df["session_label"] = df["created_at"].str[:16].str.replace("T", " ")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Pipeline Integrity Index Over Sessions",
                        "Defect Counts Over Sessions"),
        vertical_spacing=0.12,
    )

    # Row 1 — integrity index trend
    # We recompute it from columns if available, else show total_unique_defects proxy
    if "total_unique_defects" in df.columns:
        proxy_score = (100 - df["total_unique_defects"] * 2).clip(0, 100)
        fig.add_trace(go.Scatter(
            x=df["session_label"], y=proxy_score,
            mode="lines+markers",
            name="Integrity Index",
            line=dict(color=C_GOOD, width=2),
            marker=dict(size=7, color=C_GOOD),
            fill="tozeroy",
            opacity=0.08,
        ), row=1, col=1)

    # Row 2 — per class defect trend
    for cls, color in [("crack_count", C_CRACK), ("corrosion_count", C_CORROSION),
                       ("rust_count", C_RUST), ("deformation_count", C_DEFORMATION)]:
        if cls in df.columns:
            fig.add_trace(go.Scatter(
                x=df["session_label"], y=df[cls],
                mode="lines+markers",
                name=cls.replace("_count", ""),
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
            ), row=2, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=480,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER,
                    orientation="h", yanchor="bottom", y=-0.15),
        hovermode="x unified",
    )
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=10, color=C_MUTED))
    return json.loads(fig.to_json())


# ─────────────────────────────────────────────────────────────
# PER-DEFECT PROFILE TABLE
# ─────────────────────────────────────────────────────────────

def build_defect_profile_table(df: pd.DataFrame) -> list:
    """
    One row per unique track_id with:
    class, max_severity, max_area, avg_confidence,
    first_seen_sec, last_seen_sec, duration_sec, primary_zone
    """
    if df.empty:
        return []

    severity_order = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}
    df2 = df.copy()
    df2["sev_rank"] = df2["severity"].map(severity_order).fillna(0)

    profile = df2.groupby("track_id").agg(
        class_name=("class_name", "first"),
        max_severity=("severity", lambda x: x.iloc[df2.loc[x.index, "sev_rank"].values.argmax()]),
        max_area=("area", "max"),
        avg_confidence=("confidence", "mean"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        primary_zone=("zone", lambda x: x.mode()[0] if not x.empty else "unknown"),
    ).reset_index()

    profile["duration_sec"] = (profile["last_seen"] - profile["first_seen"]).round(2)
    profile["avg_confidence"] = profile["avg_confidence"].round(3)
    profile["max_area"] = profile["max_area"].round(1)

    return profile.to_dict("records")


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — run_full_analytics()
# ─────────────────────────────────────────────────────────────

def run_full_analytics(detections_csv_path: str,
                       sessions_summary_path: str,
                       fps: float = 30.0) -> dict:
    """
    Master function called by the FastAPI endpoint.
    Returns a single JSON-serialisable dict with every metric and chart.
    """
    if not os.path.exists(detections_csv_path):
        return {"error": f"Detections CSV not found: {detections_csv_path}"}

    df = pd.read_csv(detections_csv_path)

    # ── Basic counts ─────────────────────────────────────────
    if df.empty:
        unique_defects = {}
    else:
        unique_defects = df.groupby("class_name")["track_id"].nunique().to_dict()

    total_frames = int(df["frame_number"].max()) if not df.empty else 0
    total_detections = len(df)

    # ── Anomaly detection ─────────────────────────────────────
    anomaly_result = run_anomaly_detection(df, fps)

    # ── Integrity index ───────────────────────────────────────
    integrity = compute_integrity_index(unique_defects, anomaly_result["anomaly_count"])

    # ── Zone data ─────────────────────────────────────────────
    zone_chart, zone_df = chart_zone_analysis(df)
    zone_summary = zone_df.to_dict("records") if not zone_df.empty else []

    # ── Alerts ────────────────────────────────────────────────
    alerts = generate_alerts(unique_defects, integrity, zone_df, df)

    # ── Heatmaps (one per class that appears) ─────────────────
    heatmaps = {}
    for cls in unique_defects.keys():
        heatmaps[cls] = chart_heatmap(df, cls)

    # ── All charts ────────────────────────────────────────────
    charts = {
        "defect_counts":      chart_defect_counts(unique_defects),
        "severity_donut":     chart_severity_donut(df),
        "zone_analysis":      zone_chart,
        "activity_timeline":  chart_activity_timeline(df, fps),
        "size_distribution":  chart_defect_size_distribution(df),
        "confidence_violin":  chart_confidence_distribution(df),
        "integrity_gauge":    chart_integrity_gauge(integrity),
        "anomaly_timeline":   anomaly_result["chart"],
        "comparative_trend":  chart_comparative_trend(sessions_summary_path),
        "heatmaps":           heatmaps,
    }

    return {
        "unique_defects":    unique_defects,
        "total_detections":  total_detections,
        "total_frames":      total_frames,
        "integrity":         integrity,
        "alerts":            alerts,
        "anomaly_count":     anomaly_result["anomaly_count"],
        "anomalies":         anomaly_result["anomalies"],
        "zone_summary":      zone_summary,
        "defect_profiles":   build_defect_profile_table(df),
        "charts":            charts,
    }