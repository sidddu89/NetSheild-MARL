#!/usr/bin/python3
"""
NetShield-MARL: Phase 5 - Real-Time Security Visualizer Dashboard
Module Path: user_space/dashboard.py

A Streamlit + PyVis + NetworkX dashboard that reads directly from the
live SQLite audit ledger and renders:

  1. Top KPI metric cards   -- nodes, events, threat score, ledger integrity
  2. Live network graph     -- colour-coded PyVis topology (green/orange/red)
  3. Anomaly & action feed  -- scrolling MARL decision log
  4. Ledger block explorer  -- paginated table + one-click hash verification

Run:
    streamlit run user_space/dashboard.py

The dashboard polls the SQLite database every REFRESH_INTERVAL_SEC seconds
using Streamlit's built-in rerun so it tracks the live pipeline without
requiring any WebSocket or inter-process connection.

Dependencies:
    streamlit, pyvis, networkx, pandas, sqlite3 (stdlib)
"""

import os
import json
import math
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import streamlit as st
from pyvis.network import Network

# ---------------------------------------------------------------------------
# Path bootstrap -- allows running from any working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent   # user_space/
_ROOT = _HERE.parent                       # project root

import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from user_space.audit_ledger import CryptographicAuditLedger

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH: str = os.environ.get(
    "NETSHIELD_LEDGER_DB",
    str(_ROOT / "netshield_audit.db"),
)

REFRESH_INTERVAL_SEC: int = int(os.environ.get("NETSHIELD_DASH_REFRESH", "4"))
MAX_FEED_ROWS: int = 30

CONTAINER_REGISTRY: Dict[str, str] = {
    "netshield_frontend": "172.18.0.2",
    "netshield_api":      "172.18.0.3",
    "netshield_db":       "172.18.0.4",
    "netshield_attacker": "172.18.0.5",
}

ACTION_COLOURS: Dict[str, str] = {
    "ALLOW":             "#22c55e",
    "THROTTLE":          "#f97316",
    "BLOCK_PORT":        "#ef4444",
    "ISOLATE_CONTAINER": "#dc2626",
    "GENESIS_BLOCK":     "#6366f1",
    "UNKNOWN":           "#94a3b8",
}

# =============================================================================
# PAGE CONFIG  (must be the FIRST Streamlit call in the script)
# =============================================================================
st.set_page_config(
    page_title="NetShield-MARL Visualizer",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# DARK CYBERPUNK THEME CSS
# =============================================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg-base:      #020617;
  --bg-card:      #0f172a;
  --bg-card-2:    #1e293b;
  --border:       #334155;
  --text-primary: #f1f5f9;
  --text-muted:   #94a3b8;
  --accent:       #38bdf8;
  --accent-glow:  #0ea5e9;
  --green:        #22c55e;
  --orange:       #f97316;
  --red:          #ef4444;
  --indigo:       #818cf8;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg-base) !important;
  color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 100% !important; }

/* ---- Header ---- */
.ns-header {
  background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
  border: 1px solid var(--border);
  border-bottom: 2px solid var(--accent);
  border-radius: 12px;
  padding: 1.1rem 2rem;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  box-shadow: 0 0 40px rgba(56,189,248,0.10);
}
.ns-header h1 {
  font-size: 1.55rem;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 4s linear infinite;
}
@keyframes shimmer { to { background-position: 200% center; } }
.ns-header .subtitle {
  font-size: 0.78rem;
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
  margin: 0;
}
.ns-live-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse-dot 1.5s ease-in-out infinite;
  flex-shrink: 0;
}
@keyframes pulse-dot {
  0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(34,197,94,0.7); }
  50%      { opacity:0.6; box-shadow:0 0 0 6px rgba(34,197,94,0); }
}

/* ---- Metric cards ---- */
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.2rem 1.4rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
  border-color: var(--accent);
  box-shadow: 0 0 20px rgba(56,189,248,0.12);
}
.metric-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: var(--accent-stripe, var(--accent));
}
.metric-label {
  font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-muted); margin-bottom: 0.4rem;
}
.metric-value {
  font-size: 2.1rem; font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  line-height: 1; margin-bottom: 0.25rem;
}
.metric-sub { font-size: 0.72rem; color: var(--text-muted); }

/* ---- Section titles ---- */
.ns-section-title {
  font-size: 0.8rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--accent);
  border-left: 3px solid var(--accent);
  padding-left: 0.6rem; margin-bottom: 0.75rem;
}

/* ---- Feed items ---- */
.feed-item {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--item-accent, var(--accent));
  border-radius: 6px;
  padding: 0.55rem 0.9rem; margin-bottom: 0.4rem;
  font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;
  animation: slide-in 0.3s ease;
}
@keyframes slide-in { from { opacity:0; transform:translateX(-8px); } }
.feed-allow            { --item-accent: var(--green); }
.feed-throttle         { --item-accent: var(--orange); }
.feed-block_port       { --item-accent: var(--red); }
.feed-isolate_container{ --item-accent: #dc2626; }
.feed-genesis          { --item-accent: var(--indigo); }

/* ---- Badges ---- */
.badge {
  display: inline-block; font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase;
  border-radius: 999px; padding: 2px 10px;
  font-family: 'JetBrains Mono', monospace;
}
.badge-green  { background:#14532d; color:#4ade80; border:1px solid #16a34a; }
.badge-red    { background:#450a0a; color:#f87171; border:1px solid #dc2626; }
.badge-orange { background:#431407; color:#fb923c; border:1px solid #ea580c; }
.badge-indigo { background:#1e1b4b; color:#a5b4fc; border:1px solid #6366f1; }
.badge-slate  { background:#1e293b; color:#94a3b8; border:1px solid #475569; }

/* ---- Buttons ---- */
.stButton > button {
  background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
  color: white !important; border: none !important;
  font-weight: 600 !important; letter-spacing: 0.05em !important;
  border-radius: 8px !important;
  transition: opacity 0.2s, box-shadow 0.2s !important;
}
.stButton > button:hover {
  opacity: 0.88 !important;
  box-shadow: 0 0 16px rgba(14,165,233,0.4) !important;
}

/* ---- Divider ---- */
.ns-divider { border:none; border-top:1px solid var(--border); margin:1rem 0; }

/* ---- Hash text ---- */
.hash-text {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem; color: var(--accent); word-break: break-all;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# =============================================================================
# DATA LAYER
# =============================================================================

@st.cache_resource
def _get_ledger() -> CryptographicAuditLedger:
    """Cache one CryptographicAuditLedger for the Streamlit server lifetime."""
    return CryptographicAuditLedger(db_path=DB_PATH)


def _db_exists() -> bool:
    return Path(DB_PATH).exists()


def load_all_blocks() -> pd.DataFrame:
    """
    Read every block from audit_ledger into a DataFrame.
    Returns an empty DataFrame with correct columns if DB is unavailable.
    """
    empty = pd.DataFrame(columns=[
        "block_index", "timestamp", "event_payload",
        "previous_hash", "current_hash",
    ])
    if not _db_exists():
        return empty
    try:
        # Read-only URI prevents dashboard writes from corrupting the live DB
        conn = sqlite3.connect(
            f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False
        )
        df = pd.read_sql_query(
            "SELECT block_index, timestamp, event_payload, "
            "previous_hash, current_hash "
            "FROM audit_ledger ORDER BY block_index ASC",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return empty


def parse_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the JSON event_payload column into flat scalar columns.
    Enriches the DataFrame with: action_taken, anomaly_score, source_ip,
    target_port, container_agent, shaped_reward, dry_run, execution_result.
    """
    if df.empty:
        return df

    def _parse(raw: str) -> pd.Series:
        try:
            p = json.loads(raw)
            return pd.Series({
                "action_taken":    p.get("action_taken",           "UNKNOWN"),
                "anomaly_score":   float(p.get("detected_anomaly_score", 0.0)),
                "source_ip":       p.get("source_ip",              ""),
                "target_port":     p.get("target_port",            ""),
                "container_agent": p.get("container_agent",
                                         p.get("event_type",       "GENESIS")),
                "shaped_reward":   float(p.get("shaped_reward",    0.0)),
                "dry_run":         bool(p.get("dry_run",           True)),
                "execution_result":p.get("execution_result",       ""),
            })
        except Exception:
            return pd.Series({
                "action_taken": "UNKNOWN", "anomaly_score": 0.0,
                "source_ip": "", "target_port": "", "container_agent": "GENESIS",
                "shaped_reward": 0.0, "dry_run": True, "execution_result": "",
            })

    parsed = df["event_payload"].apply(_parse)
    return pd.concat([df.drop(columns=["event_payload"]), parsed], axis=1)


# =============================================================================
# DERIVED METRICS
# =============================================================================

def compute_metrics(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "total_blocks": 0, "active_nodes": len(CONTAINER_REGISTRY),
            "total_events": 0, "avg_score": 0.0, "max_score": 0.0,
            "anomaly_pct": 0.0, "defensive_actions": 0,
        }

    defensive = int(df["action_taken"].isin(
        ["THROTTLE", "BLOCK_PORT", "ISOLATE_CONTAINER"]
    ).sum())
    scored = df[df["anomaly_score"] > 0]["anomaly_score"]

    return {
        "total_blocks":      int(len(df)),
        "active_nodes":      int(df["container_agent"].nunique()),
        "total_events":      max(0, int(len(df)) - 1),
        "avg_score":         float(scored.mean()) if not scored.empty else 0.0,
        "max_score":         float(df["anomaly_score"].max()),
        "anomaly_pct":       float(scored.mean() * 100) if not scored.empty else 0.0,
        "defensive_actions": defensive,
    }


# =============================================================================
# NETWORK GRAPH  (PyVis + NetworkX)
# =============================================================================

NODE_ICONS = {
    "netshield_frontend": "FRONTEND",
    "netshield_api":      "API",
    "netshield_db":       "DB",
    "netshield_attacker": "ATTACKER",
}

SEVERITY = {"ALLOW": 0, "THROTTLE": 1, "BLOCK_PORT": 2,
            "ISOLATE_CONTAINER": 3, "UNKNOWN": 0}


def build_network_html(df: pd.DataFrame) -> str:
    """
    Build and return a self-contained PyVis HTML string.

    Node colour = worst MARL action ever applied to that agent:
      green (#22c55e)   ALLOW
      orange (#f97316)  THROTTLE
      red (#ef4444)     BLOCK_PORT
      deep-red (#dc2626) ISOLATE_CONTAINER

    Edge weight = log1p(event count), colour mirrors target action.
    Falls back to the docker-compose topology skeleton when no real edges exist.
    """
    G = nx.DiGraph()

    # -- Worst action per container node --
    node_worst: Dict[str, str] = {}
    if not df.empty:
        for _, row in df.iterrows():
            agent  = str(row.get("container_agent", ""))
            action = str(row.get("action_taken", "ALLOW"))
            if SEVERITY.get(action, 0) > SEVERITY.get(node_worst.get(agent, "ALLOW"), 0):
                node_worst[agent] = action

    # -- Add nodes --
    all_nodes = set(CONTAINER_REGISTRY.keys())
    if not df.empty:
        for a in df["container_agent"].dropna().unique():
            all_nodes.add(str(a))

    for node in all_nodes:
        worst  = node_worst.get(node, "ALLOW")
        colour = ACTION_COLOURS.get(worst, "#94a3b8")
        label  = NODE_ICONS.get(node, node.replace("netshield_", "").upper())
        ip     = CONTAINER_REGISTRY.get(node, "")
        count  = int(df[df["container_agent"] == node].shape[0]) if not df.empty else 0
        title  = (
            f"<b>{node}</b><br>"
            f"IP: {ip}<br>"
            f"Events: {count}<br>"
            f"Status: <b style='color:{colour}'>{worst}</b>"
        )
        G.add_node(
            node, label=label, color=colour, title=title,
            size=28 + min(count * 3, 30), font={"color": "#f1f5f9", "size": 13},
        )

    # -- Add edges --
    ip_to_node = {v: k for k, v in CONTAINER_REGISTRY.items()}
    if not df.empty:
        edge_counts: Dict[Tuple[str, str], int] = {}
        for _, row in df.iterrows():
            src_ip = str(row.get("source_ip", ""))
            tgt    = str(row.get("container_agent", ""))
            src    = ip_to_node.get(src_ip, src_ip)
            if src and tgt and src != tgt:
                edge_counts[(src, tgt)] = edge_counts.get((src, tgt), 0) + 1

        for (src, tgt), weight in edge_counts.items():
            if src not in G.nodes:
                G.add_node(src, label=src, color="#94a3b8", size=18,
                           font={"color": "#f1f5f9", "size": 11})
            best_action = (
                df[df["container_agent"] == tgt]["action_taken"]
                .value_counts().idxmax()
                if tgt in df["container_agent"].values else "ALLOW"
            )
            G.add_edge(
                src, tgt,
                width=1.0 + math.log1p(weight),
                color=ACTION_COLOURS.get(best_action, "#64748b"),
                title=f"{weight} events | {best_action}",
                arrows="to",
            )

    # Fall back to topology skeleton when no real edges exist
    if G.number_of_edges() == 0:
        for src, tgt in [
            ("netshield_api",      "netshield_frontend"),
            ("netshield_api",      "netshield_db"),
            ("netshield_attacker", "netshield_db"),
            ("netshield_attacker", "netshield_frontend"),
        ]:
            if src in G.nodes and tgt in G.nodes:
                G.add_edge(src, tgt, width=1.5, color="#334155",
                           title="topology", arrows="to")

    # -- Render with PyVis --
    net = Network(
        height="490px", width="100%",
        bgcolor="#020617", font_color="#94a3b8",
        directed=True, notebook=False,
    )
    net.from_nx(G)
    net.set_options("""{
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -6500,
          "centralGravity": 0.3,
          "springLength": 160,
          "springConstant": 0.04,
          "damping": 0.12
        },
        "stabilization": {"iterations": 160}
      },
      "edges": {
        "smooth": {"enabled": true, "type": "dynamic"},
        "font":   {"size": 11, "color": "#64748b"}
      },
      "nodes": {
        "borderWidth": 2,
        "shadow": {"enabled": true, "size": 8}
      },
      "interaction": {"hover": true, "tooltipDelay": 150}
    }""")

    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        net.save_graph(f.name)
        tmp = f.name
    with open(tmp, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp)
    return html


# =============================================================================
# UI HELPERS
# =============================================================================

def _badge(action: str) -> str:
    css = {
        "ALLOW":             "badge-green",
        "THROTTLE":          "badge-orange",
        "BLOCK_PORT":        "badge-red",
        "ISOLATE_CONTAINER": "badge-red",
        "GENESIS_BLOCK":     "badge-indigo",
    }.get(action.upper(), "badge-slate")
    return f'<span class="badge {css}">{action}</span>'


def _score_colour(score: float) -> str:
    if score >= 0.85: return "var(--red)"
    if score >= 0.60: return "var(--orange)"
    if score >= 0.30: return "#facc15"
    return "var(--green)"


# =============================================================================
# RENDER FUNCTIONS
# =============================================================================

def render_header(ts: str) -> None:
    st.markdown(f"""
    <div class="ns-header">
      <div class="ns-live-dot"></div>
      <div>
        <h1>NetShield-MARL &bull; Real-Time Zero-Trust Security Visualizer</h1>
        <p class="subtitle">
          B.Tech AIML &nbsp;|&nbsp;
          eBPF &rarr; GIN &rarr; IPPO-MARL &rarr; SHA-256 Ledger
          &nbsp;|&nbsp; Refreshed: {ts}
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(metrics: Dict, chain_ok: Optional[bool]) -> None:
    c1, c2, c3, c4 = st.columns(4, gap="small")

    with c1:
        st.markdown(f"""
        <div class="metric-card" style="--accent-stripe:#38bdf8">
          <div class="metric-label">Active Microservice Nodes</div>
          <div class="metric-value" style="color:#38bdf8">{metrics['active_nodes']}</div>
          <div class="metric-sub">{len(CONTAINER_REGISTRY)} topology containers</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card" style="--accent-stripe:#818cf8">
          <div class="metric-label">Ingested Kernel Events</div>
          <div class="metric-value" style="color:#818cf8">{metrics['total_events']}</div>
          <div class="metric-sub">{metrics['total_blocks']} ledger blocks total</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        sc  = _score_colour(metrics["avg_score"])
        pct = metrics["anomaly_pct"]
        st.markdown(f"""
        <div class="metric-card" style="--accent-stripe:{sc}">
          <div class="metric-label">Avg Anomaly Threat Score</div>
          <div class="metric-value" style="color:{sc}">{pct:.1f}<span style="font-size:1.1rem">%</span></div>
          <div class="metric-sub">Peak: {metrics['max_score']:.4f} &nbsp;|&nbsp; {metrics['defensive_actions']} defenses</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        if chain_ok is None:
            bdg, sub, stripe = '<span class="badge badge-slate">NO DB</span>', "Database not found", "#475569"
        elif chain_ok:
            bdg, sub, stripe = '<span class="badge badge-green">&#10003; PASS</span>', "SHA-256 chain intact", "#22c55e"
        else:
            bdg, sub, stripe = '<span class="badge badge-red">&#10007; TAMPERED</span>', "Chain integrity broken!", "#ef4444"
        st.markdown(f"""
        <div class="metric-card" style="--accent-stripe:{stripe}">
          <div class="metric-label">Cryptographic Ledger</div>
          <div class="metric-value" style="font-size:1.35rem;padding-top:.3rem">{bdg}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)


def render_feed(df: pd.DataFrame) -> None:
    st.markdown('<div class="ns-section-title">&#9889; Live Anomaly &amp; Mitigation Feed</div>',
                unsafe_allow_html=True)

    if df.empty:
        st.markdown("""<div class="feed-item" style="color:var(--text-muted)">
          Waiting for pipeline events &mdash; start <code>main_pipeline.py</code> to begin.
        </div>""", unsafe_allow_html=True)
        return

    feed = df[df["action_taken"] != "GENESIS_BLOCK"].tail(MAX_FEED_ROWS).iloc[::-1]

    for _, row in feed.iterrows():
        action  = str(row.get("action_taken", "UNKNOWN")).upper()
        score   = float(row.get("anomaly_score", 0.0))
        agent   = str(row.get("container_agent", ""))
        src_ip  = str(row.get("source_ip", ""))
        port    = str(row.get("target_port", ""))
        blk     = int(row.get("block_index", 0))
        ts_raw  = str(row.get("timestamp", ""))
        reward  = float(row.get("shaped_reward", 0.0))

        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).strftime("%H:%M:%S")
        except Exception:
            ts = ts_raw[:19]

        fcls   = f"feed-{action.lower().replace(' ','_')}"
        scol   = _score_colour(score)
        rcol   = "var(--green)" if reward >= 0 else "var(--red)"

        st.markdown(f"""
        <div class="feed-item {fcls}">
          <span style="color:var(--text-muted)">[{ts}]</span>
          &nbsp; Block <span style="color:var(--accent)">#{blk}</span>
          &nbsp; {_badge(action)}
          &nbsp; <span style="color:{scol}">score={score:.4f}</span>
          &nbsp; agent=<b style="color:#e2e8f0">{agent}</b>
          &nbsp; src={src_ip}:{port}
          &nbsp; R=<span style="color:{rcol}">{reward:+.2f}</span>
        </div>""", unsafe_allow_html=True)


def render_explorer(df: pd.DataFrame) -> None:
    st.markdown('<div class="ns-section-title">&#128272; Cryptographic Ledger Block Explorer</div>',
                unsafe_allow_html=True)

    btn_col, result_col = st.columns([1, 3], gap="small")

    with btn_col:
        clicked = st.button("&#128269; Verify Hash Integrity", use_container_width=True)

    with result_col:
        if clicked:
            if not _db_exists():
                st.error("Database not found. Start the pipeline first.", icon="X")
            else:
                with st.spinner("Recalculating all SHA-256 hashes..."):
                    ledger = _get_ledger()
                    valid, bad = ledger.verify_chain_integrity()
                if valid:
                    st.success(
                        f"Chain VALID - all {len(df)} blocks cryptographically intact.",
                        icon="ok"
                    )
                else:
                    st.error(
                        f"TAMPER DETECTED at Block #{bad}! Hash chain broken.",
                        icon="warning"
                    )

    if df.empty:
        st.info("No ledger blocks to display yet.", icon="i")
        return

    display = df[[
        "block_index", "timestamp", "container_agent", "action_taken",
        "anomaly_score", "source_ip", "target_port", "shaped_reward",
        "current_hash", "previous_hash",
    ]].copy()

    display["current_hash"]  = display["current_hash"].str[:22] + "..."
    display["previous_hash"] = display["previous_hash"].str[:22] + "..."
    display["anomaly_score"] = display["anomaly_score"].apply(lambda x: f"{x:.4f}")
    display["shaped_reward"] = display["shaped_reward"].apply(
        lambda x: f"{float(x):+.2f}" if x != "" else ""
    )
    display = display.rename(columns={
        "block_index": "#", "timestamp": "Timestamp (UTC)",
        "container_agent": "Agent", "action_taken": "Action",
        "anomaly_score": "Score", "source_ip": "Src IP",
        "target_port": "Port", "shaped_reward": "Reward",
        "current_hash": "Current Hash", "previous_hash": "Prev Hash",
    })

    st.dataframe(
        display.iloc[::-1],
        use_container_width=True,
        height=340,
        column_config={
            "#":             st.column_config.NumberColumn(width="small"),
            "Action":        st.column_config.TextColumn(width="medium"),
            "Score":         st.column_config.TextColumn(width="small"),
            "Reward":        st.column_config.TextColumn(width="small"),
            "Current Hash":  st.column_config.TextColumn(width="large"),
            "Prev Hash":     st.column_config.TextColumn(width="large"),
        },
        hide_index=True,
    )

    latest = df.iloc[-1]
    with st.expander("Latest Block Detail", expanded=False):
        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown(f"**Block Index:** `{latest['block_index']}`")
            st.markdown(f"**Timestamp:** `{latest['timestamp']}`")
            st.markdown(f"**Agent:** `{latest['container_agent']}`")
            st.markdown(f"**Action:** {_badge(latest['action_taken'])}",
                        unsafe_allow_html=True)
        with lc2:
            st.markdown(f"**Anomaly Score:** `{float(latest['anomaly_score']):.6f}`")
            st.markdown(f"**Shaped Reward:** `{float(latest['shaped_reward']):+.4f}`")
        st.markdown("**Current Hash:**")
        st.markdown(f'<p class="hash-text">{latest["current_hash"]}</p>',
                    unsafe_allow_html=True)
        st.markdown("**Previous Hash:**")
        st.markdown(f'<p class="hash-text">{latest["previous_hash"]}</p>',
                    unsafe_allow_html=True)


def render_sidebar(metrics: Dict) -> int:
    with st.sidebar:
        st.markdown("### Dashboard Settings")
        refresh = st.slider("Auto-refresh (sec)", 2, 30,
                            REFRESH_INTERVAL_SEC, step=1)
        st.markdown("---")
        st.markdown("### Session Totals")
        st.metric("Total Blocks",      metrics["total_blocks"])
        st.metric("Defensive Actions", metrics["defensive_actions"])
        st.metric("Max Threat Score",  f"{metrics['max_score']:.4f}")
        st.markdown("---")
        st.markdown("### Database")
        st.markdown(f'<span class="hash-text">{DB_PATH}</span>',
                    unsafe_allow_html=True)
        if _db_exists():
            kb = Path(DB_PATH).stat().st_size / 1024
            st.caption(f"Size: {kb:.1f} KB")
            st.markdown('<span class="badge badge-green">Connected</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-slate">Not Found</span>',
                        unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Pipeline Layers")
        for lid, name, path in [
            ("L1", "eBPF Collector",  "kernel_space/loader.py"),
            ("L2", "GIN Detector",    "user_space/gnn_detector.py"),
            ("L3", "MARL Engine",     "user_space/marl_mitigation.py"),
            ("L4", "Audit Ledger",    "user_space/audit_ledger.py"),
            ("L5", "This Dashboard",  "user_space/dashboard.py"),
        ]:
            st.markdown(
                f'<span class="badge badge-indigo">{lid}</span> '
                f'**{name}**<br>'
                f'<span style="font-size:.7rem;color:var(--text-muted)">{path}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")
    return refresh


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    import streamlit.components.v1 as components

    # -- Load & parse --
    raw_df    = load_all_blocks()
    parsed_df = parse_blocks(raw_df)
    metrics   = compute_metrics(parsed_df)
    chain_ok: Optional[bool] = (True if (_db_exists() and not raw_df.empty) else None)
    now_str   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -- Sidebar --
    refresh_sec = render_sidebar(metrics)

    # -- Header --
    render_header(now_str)

    # -- KPI Cards --
    render_kpi_cards(metrics, chain_ok)
    st.markdown('<hr class="ns-divider">', unsafe_allow_html=True)

    # -- Graph | Feed --
    graph_col, feed_col = st.columns([3, 2], gap="medium")

    with graph_col:
        st.markdown('<div class="ns-section-title">&#127758; Live Microservice Network Graph</div>',
                    unsafe_allow_html=True)
        legend = " &nbsp; ".join([
            '<span class="badge badge-green">Healthy / ALLOW</span>',
            '<span class="badge badge-orange">THROTTLE</span>',
            '<span class="badge badge-red">BLOCK / ISOLATE</span>',
            '<span class="badge badge-indigo">Genesis</span>',
        ])
        st.markdown(f'<div style="margin-bottom:.5rem">{legend}</div>',
                    unsafe_allow_html=True)
        graph_html = build_network_html(parsed_df)
        components.html(graph_html, height=510, scrolling=False)

    with feed_col:
        render_feed(parsed_df)

    st.markdown('<hr class="ns-divider">', unsafe_allow_html=True)

    # -- Block Explorer --
    render_explorer(parsed_df)

    # -- Auto-refresh countdown --
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.monotonic()

    elapsed   = time.monotonic() - st.session_state.last_refresh
    remaining = max(0.0, refresh_sec - elapsed)

    st.markdown(
        f'<p style="text-align:right;font-size:.72rem;color:var(--text-muted);'
        f'font-family:JetBrains Mono,monospace;margin-top:.5rem">'
        f'Next refresh in {remaining:.0f}s &nbsp;|&nbsp; DB: {DB_PATH}</p>',
        unsafe_allow_html=True,
    )

    if elapsed >= refresh_sec:
        st.session_state.last_refresh = time.monotonic()
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()