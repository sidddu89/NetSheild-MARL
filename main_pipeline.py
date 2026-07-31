#!/usr/bin/python3
"""
NetShield-MARL: Main Pipeline Orchestrator
File Path: main_pipeline.py

Master Event Loop & Inter-Layer Telemetry Router for the NetShield-MARL system.

Threading Model (Producer-Consumer):
    ┌─────────────────────────────────┐
    │  eBPF Producer Thread           │
    │  (kernel_space/loader.py)       │
    │  EBPFLoader.start_listener()    │
    │  Intercepts sys_enter_connect   │
    │  Puts payloads → event_queue    │
    └──────────────┬──────────────────┘
                   │ queue.Queue (thread-safe)
    ┌──────────────▼──────────────────┐
    │  Main Consumer Thread           │
    │  GraphBuilder.add_interaction() │  ← drains queue every cycle
    │  Every N seconds:               │
    │    GINAnomalyDetector.infer()   │
    │    MARLMitigationEngine.apply() │
    │    CryptographicAuditLedger     │
    └─────────────────────────────────┘

Dry-Run / Demo Mode:
    By default NETSHIELD_DRY_RUN=true — the pipeline runs with a synthetic
    event injector instead of the live eBPF loader, making it safe to
    demonstrate on any machine without root privileges or BCC.
    Set NETSHIELD_DRY_RUN=false and run with `sudo` for live kernel hooking.

Dependencies:
    Standard library: threading, queue, signal, time, os, sys, logging, json
    User space:       torch, torch_geometric (via layer modules)
"""

import os
import sys
import time
import json
import queue
import signal
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import torch

# ---------------------------------------------------------------------------
# Logging — module-level, structured, timestamped
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("NetShield.Pipeline")

# ---------------------------------------------------------------------------
# Runtime Configuration — all tunable via environment variables
# ---------------------------------------------------------------------------
# Time window (seconds) between GNN batch evaluations.
GRAPH_WINDOW_SEC: float = float(os.environ.get("NETSHIELD_WINDOW_SEC", "3.0"))

# Anomaly score threshold above which MARL mitigation is triggered.
ANOMALY_TRIGGER_THRESHOLD: float = float(os.environ.get("NETSHIELD_THRESHOLD", "0.30"))

# Maximum events the inter-thread queue will buffer before back-pressuring.
QUEUE_MAX_SIZE: int = int(os.environ.get("NETSHIELD_QUEUE_SIZE", "2048"))

# Dry-run: True = safe demo mode (no eBPF, no iptables). False = live kernel mode.
DRY_RUN: bool = os.environ.get("NETSHIELD_DRY_RUN", "true").lower() != "false"

# Path to the SQLite audit ledger database file.
LEDGER_DB_PATH: str = os.environ.get("NETSHIELD_LEDGER_DB", "netshield_audit.db")

# Path to optional pre-trained MARL policy weights (.pt file).
POLICY_WEIGHTS_PATH: Optional[str] = os.environ.get("NETSHIELD_POLICY_WEIGHTS", None)

# Docker/container names observed by the docker-compose topology (for ISOLATE actions).
KNOWN_CONTAINER_NODES = {
    "netshield_frontend",
    "netshield_api",
    "netshield_db",
    "netshield_attacker",
}


# ===========================================================================
# Conditional Layer Imports
# ===========================================================================
# EBPFLoader requires root + BCC (only available on Linux with kernel headers).
# In dry-run mode we skip the import entirely so the pipeline can run on
# any machine (Windows WSL2, CI, development laptops) without sudo.

if not DRY_RUN:
    try:
        from kernel_space.loader import EBPFLoader
    except ImportError as exc:
        log.critical(
            f"Failed to import EBPFLoader (live mode requires BCC + root): {exc}"
        )
        sys.exit(1)

# User-space layers are always imported.
from user_space.graph_builder import GraphBuilder
from user_space.gnn_detector import GINAnomalyDetector
from user_space.marl_mitigation import MARLMitigationEngine, MARLMitigationAgent
from user_space.audit_ledger import CryptographicAuditLedger


# ===========================================================================
# Synthetic Event Injector (Dry-Run / Demo Mode)
# ===========================================================================

class SyntheticEventInjector(threading.Thread):
    """
    Simulates a stream of eBPF socket telemetry events in dry-run mode.
    Generates a mix of normal traffic and high-anomaly attack bursts so the
    full pipeline can be demonstrated end-to-end without kernel privileges.

    Runs as a daemon thread; stops when `stop_event` is set.
    """

    # Simulated microservice topology matching docker-compose.yml
    _NORMAL_TRAFFIC = [
        {"pid": 101, "tgid": 101, "uid": 1000, "comm": "netshield_api",    "dest_ip": "172.18.0.2", "dest_port": 80},
        {"pid": 102, "tgid": 102, "uid": 1000, "comm": "netshield_api",    "dest_ip": "172.18.0.3", "dest_port": 5432},
        {"pid": 103, "tgid": 103, "uid": 1000, "comm": "netshield_db",     "dest_ip": "172.18.0.2", "dest_port": 80},
    ]
    _ATTACK_TRAFFIC = [
        {"pid": 999, "tgid": 999, "uid": 0,    "comm": "netshield_attacker", "dest_ip": "172.18.0.3", "dest_port": 5432},
        {"pid": 999, "tgid": 999, "uid": 0,    "comm": "netshield_attacker", "dest_ip": "172.18.0.2", "dest_port": 80},
        {"pid": 999, "tgid": 999, "uid": 0,    "comm": "netshield_attacker", "dest_ip": "172.18.0.4", "dest_port": 8080},
        {"pid": 998, "tgid": 998, "uid": 0,    "comm": "netshield_attacker", "dest_ip": "172.18.0.3", "dest_port": 5432},
        {"pid": 997, "tgid": 997, "uid": 0,    "comm": "netshield_attacker", "dest_ip": "172.18.0.3", "dest_port": 5432},
        {"pid": 996, "tgid": 996, "uid": 0,    "comm": "netshield_attacker", "dest_ip": "172.18.0.3", "dest_port": 5432},
    ]

    def __init__(self, event_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="SyntheticInjector", daemon=True)
        self._queue      = event_queue
        self._stop       = stop_event
        self._cycle      = 0

    def run(self) -> None:
        log.info("[DRY-RUN] Synthetic event injector started.")
        while not self._stop.is_set():
            self._cycle += 1
            # Every 3rd cycle simulate an attack burst; otherwise normal traffic.
            traffic = self._ATTACK_TRAFFIC if (self._cycle % 3 == 0) else self._NORMAL_TRAFFIC
            for event in traffic:
                if self._stop.is_set():
                    break
                try:
                    self._queue.put_nowait(event)
                except queue.Full:
                    log.warning("[DRY-RUN] Event queue full — dropping synthetic event.")
            # Emit events at ~1 event per 150ms to fill each 3-second window
            time.sleep(0.15)

        log.info("[DRY-RUN] Synthetic event injector stopped.")


# ===========================================================================
# eBPF Producer Thread (Live Mode Only)
# ===========================================================================

class EBPFProducerThread(threading.Thread):
    """
    Wraps EBPFLoader in a background daemon thread.
    Injects raw socket telemetry payloads from the BCC perf ring buffer
    into the shared inter-thread event_queue.

    Inherits from threading.Thread so the main orchestrator loop can run
    independently on the main thread, consuming from the queue at its own pace.
    """

    def __init__(self, event_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="EBPFProducer", daemon=True)
        self._queue      = event_queue
        self._stop       = stop_event
        self._bpf_loader: Optional["EBPFLoader"] = None  # Lazy-init in run()

    def _on_ebpf_event(self, payload: Dict[str, Any]) -> None:
        """BCC callback: enqueue telemetry event for main-thread consumption."""
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            log.warning(
                f"[eBPF] Event queue full (size={self._queue.maxsize}) — "
                f"dropping event from PID {payload.get('pid', '?')}. "
                f"Consider increasing NETSHIELD_QUEUE_SIZE."
            )

    def run(self) -> None:
        """
        Compiles and loads the eBPF probe, then polls the perf ring buffer
        in a tight loop until the stop_event is signalled.
        """
        log.info("[eBPF] Compiling and loading probe into kernel...")
        try:
            self._bpf_loader = EBPFLoader(callback_fn=self._on_ebpf_event)
        except Exception as exc:
            log.critical(f"[eBPF] Failed to load BPF probe: {exc}")
            self._stop.set()
            return

        log.info("[eBPF] Probe active — intercepting sys_enter_connect syscalls.")

        # Open the perf ring buffer. BCC will invoke _on_ebpf_event() per event.
        self._bpf_loader.bpf["socket_events"].open_perf_buffer(
            self._bpf_loader._event_handler
        )

        while not self._stop.is_set():
            # Poll with 100ms timeout — yields CPU between events, avoids 100% spin.
            self._bpf_loader.bpf.perf_buffer_poll(timeout=100)

        log.info("[eBPF] Probe detaching from kernel.")

    def stop(self) -> None:
        """Signals the polling loop to exit cleanly."""
        self._stop.set()


# ===========================================================================
# NetShield MARL Orchestrator — Core Pipeline
# ===========================================================================

class NetShieldOrchestrator:
    """
    NetShield-MARL Master Pipeline Orchestrator.

    Implements a producer-consumer threading model:
      - Producer:   EBPFProducerThread (live) or SyntheticEventInjector (dry-run)
                    writes socket telemetry dicts into self._event_queue.
      - Consumer:   _pipeline_loop() (runs on the calling thread) drains the queue,
                    feeds events into GraphBuilder, and on each timed window boundary
                    runs the full GNN → MARL → Ledger decision cycle.

    The orchestrator owns all layer instances and manages their lifetimes.
    """

    def __init__(
        self,
        graph_window_sec:  float         = GRAPH_WINDOW_SEC,
        anomaly_threshold: float         = ANOMALY_TRIGGER_THRESHOLD,
        ledger_db_path:    str           = LEDGER_DB_PATH,
        policy_weights:    Optional[str] = POLICY_WEIGHTS_PATH,
        dry_run:           bool          = DRY_RUN,
    ):
        """
        Args:
            graph_window_sec:  Seconds between GNN batch evaluations.
            anomaly_threshold: GNN score above which MARL mitigation fires.
            ledger_db_path:    File path to the SQLite audit database.
            policy_weights:    Optional path to saved MARL policy .pt file.
            dry_run:           If True, use synthetic events instead of live eBPF.
        """
        self.graph_window_sec  = graph_window_sec
        self.anomaly_threshold = anomaly_threshold
        self.dry_run           = dry_run

        # ---------------------------------------------------------------
        # Thread synchronisation primitives
        # ---------------------------------------------------------------
        # Thread-safe queue: eBPF/synthetic producer → main consumer loop.
        self._event_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        # Cooperative stop signal: set to True to terminate all threads cleanly.
        self._stop_event: threading.Event = threading.Event()
        # Producer thread reference (set in start()).
        self._producer_thread: Optional[threading.Thread] = None

        # ---------------------------------------------------------------
        # Pipeline metrics (all guarded by _metrics_lock)
        # ---------------------------------------------------------------
        self._metrics_lock = threading.Lock()
        self._total_events_ingested: int = 0
        self._total_windows_evaluated: int = 0
        self._total_anomalies_detected: int = 0
        self._total_mitigations_applied: int = 0
        self._pipeline_start_time: float = 0.0

        # ---------------------------------------------------------------
        # Layer 2: Graph Builder
        # ---------------------------------------------------------------
        log.info(f"[Init] GraphBuilder: window={graph_window_sec}s")
        self._graph_builder = GraphBuilder(time_window=graph_window_sec)

        # ---------------------------------------------------------------
        # Layer 2: GNN Anomaly Detector
        # ---------------------------------------------------------------
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"[Init] GINAnomalyDetector: device={self._device}")
        self._gnn_model = GINAnomalyDetector(num_node_features=3).to(self._device)
        self._gnn_model.eval()

        # ---------------------------------------------------------------
        # Layer 3: MARL Mitigation Engine
        # ---------------------------------------------------------------
        log.info(f"[Init] MARLMitigationEngine: dry_run={dry_run}")
        self._marl_engine = MARLMitigationEngine(
            policy_weights_path=policy_weights,
            dry_run=dry_run,
        )
        # Pre-register agents matching the docker-compose topology.
        for node_id in KNOWN_CONTAINER_NODES:
            self._marl_engine.register_agent(node_id)

        # ---------------------------------------------------------------
        # Layer 4: Cryptographic Audit Ledger
        # ---------------------------------------------------------------
        log.info(f"[Init] CryptographicAuditLedger: db={ledger_db_path}")
        self._ledger = CryptographicAuditLedger(db_path=ledger_db_path)

        log.info("=" * 65)
        log.info("🛡️  NetShield-MARL Orchestrator initialised successfully.")
        log.info(f"    Mode:      {'DRY-RUN (safe demo)' if dry_run else '🔴 LIVE (kernel eBPF)'}")
        log.info(f"    Window:    {graph_window_sec}s | Threshold: {anomaly_threshold}")
        log.info(f"    Ledger:    {ledger_db_path}")
        log.info(f"    Device:    {self._device}")
        log.info("=" * 65)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Starts the pipeline:
          1. Registers OS signal handlers for graceful Ctrl+C shutdown.
          2. Launches the eBPF producer (or synthetic injector) as a daemon thread.
          3. Enters the main consumer loop on the calling thread.
        """
        # --- Signal handlers for graceful shutdown ---
        # signal.signal() only works in the main thread of the main interpreter.
        # Guard the registration so the orchestrator can also be instantiated from
        # test harnesses and sub-threads without raising ValueError.
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT,  self._handle_shutdown_signal)
            signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
            log.debug("[Pipeline] Signal handlers registered (SIGINT, SIGTERM).")
        else:
            log.debug(
                "[Pipeline] Skipping signal registration — not in main thread. "
                "Call orchestrator.shutdown() manually to stop."
            )

        self._pipeline_start_time = time.monotonic()

        # --- Start producer thread ---
        if self.dry_run:
            self._producer_thread = SyntheticEventInjector(
                event_queue=self._event_queue,
                stop_event=self._stop_event,
            )
        else:
            self._producer_thread = EBPFProducerThread(
                event_queue=self._event_queue,
                stop_event=self._stop_event,
            )

        self._producer_thread.start()
        log.info(f"[Pipeline] Producer thread '{self._producer_thread.name}' started.")

        # --- Main consumer loop (blocks until _stop_event is set) ---
        self._pipeline_loop()

    def shutdown(self) -> None:
        """
        Gracefully shuts down the pipeline:
          1. Sets the cooperative stop event (signals threads to exit).
          2. Joins the producer thread with a timeout.
          3. Prints final audit ledger integrity verification.
          4. Prints session metrics summary.
        """
        if self._stop_event.is_set():
            return  # Already shutting down

        log.info("\n[Shutdown] Graceful shutdown initiated...")
        self._stop_event.set()

        # Wait for producer thread to exit cleanly
        if self._producer_thread and self._producer_thread.is_alive():
            log.info(f"[Shutdown] Joining producer thread '{self._producer_thread.name}'...")
            self._producer_thread.join(timeout=3.0)
            if self._producer_thread.is_alive():
                log.warning("[Shutdown] Producer thread did not exit in time — continuing.")

        # Final chain integrity check
        log.info("\n[Shutdown] Running final cryptographic chain integrity audit...")
        try:
            is_valid, bad_block = self._ledger.verify_chain_integrity()
            if is_valid:
                log.info("[Shutdown] ✅ Ledger chain integrity: VALID.")
            else:
                log.error(f"[Shutdown] ❌ Ledger chain integrity: TAMPERED at block #{bad_block}!")
        except Exception as exc:
            log.error(f"[Shutdown] Ledger verification failed: {exc}")

        # Session metrics
        self._print_session_summary()
        log.info("[Shutdown] NetShield-MARL pipeline stopped. Goodbye.")

    # ------------------------------------------------------------------
    # Core Consumer Loop
    # ------------------------------------------------------------------

    def _pipeline_loop(self) -> None:
        """
        Main consumer loop. Runs on the calling (main) thread.

        Cycle:
          1. Drain the event queue into GraphBuilder (non-blocking, bulk drain).
          2. Check if the graph time-window has elapsed.
          3. If yes: export PyG graph, run GNN inference, MARL + ledger if anomaly.
          4. Sleep briefly to yield CPU between drain cycles.
        """
        log.info("[Pipeline] Main consumer loop started.")
        _window_deadline = time.monotonic() + self.graph_window_sec

        while not self._stop_event.is_set():
            # ----------------------------------------------------------
            # Step 1: Drain event queue (non-blocking batch drain)
            # ----------------------------------------------------------
            drained_count = 0
            while True:
                try:
                    event = self._event_queue.get_nowait()
                    self._ingest_event(event)
                    drained_count += 1
                except queue.Empty:
                    break  # Queue exhausted for this cycle

            if drained_count > 0:
                log.debug(f"[Pipeline] Drained {drained_count} events from queue.")

            # ----------------------------------------------------------
            # Step 2: Check time-window boundary
            # ----------------------------------------------------------
            now = time.monotonic()
            if now >= _window_deadline:
                self._evaluate_window()
                _window_deadline = time.monotonic() + self.graph_window_sec

            # ----------------------------------------------------------
            # Step 3: Yield CPU — 50ms sleep prevents a busy-wait spin
            # ----------------------------------------------------------
            time.sleep(0.05)

        log.info("[Pipeline] Consumer loop exited.")

    def _ingest_event(self, event: Dict[str, Any]) -> None:
        """
        Receives a single eBPF telemetry event dict and feeds it into the
        GraphBuilder's temporal aggregation window.

        Args:
            event: Dict with keys: pid, tgid, uid, comm, dest_ip, dest_port.
        """
        try:
            pid       = event.get("pid",       0)
            comm      = event.get("comm",      "unknown")
            dest_ip   = event.get("dest_ip",   "0.0.0.0")
            dest_port = event.get("dest_port", 0)

            self._graph_builder.add_interaction(
                source_pid=pid,
                source_comm=comm,
                dest_ip=dest_ip,
                dest_port=dest_port,
            )

            with self._metrics_lock:
                self._total_events_ingested += 1

        except Exception as exc:
            log.error(f"[Ingest] Error processing event: {exc} | Event: {event}")

    def _evaluate_window(self) -> None:
        """
        Triggered every GRAPH_WINDOW_SEC seconds.
        Exports the PyG graph from the current window, runs GNN inference,
        and dispatches MARL mitigation if an anomaly is detected.
        """
        with self._metrics_lock:
            self._total_windows_evaluated += 1
            window_num = self._total_windows_evaluated

        # --- Export PyG Data object from aggregated window ---
        pyg_data = self._graph_builder.get_graph_data(force_export=True)
        if pyg_data is None or pyg_data.num_nodes == 0:
            log.debug(f"[Window #{window_num}] No graph data in this window. Skipping.")
            return

        log.info(
            f"[Window #{window_num}] Graph exported: "
            f"{pyg_data.num_nodes} nodes, {pyg_data.edge_index.shape[1]} edges."
        )

        # --- GNN Inference ---
        try:
            is_anomaly, anomaly_score = self._gnn_model.predict_anomaly(
                pyg_data.to(self._device)
            )
        except Exception as exc:
            log.error(f"[Window #{window_num}] GNN inference error: {exc}")
            return

        # --- Decision Gate ---
        if is_anomaly or anomaly_score > self.anomaly_threshold:
            log.warning(
                f"[Window #{window_num}] 🚨 ANOMALY DETECTED | "
                f"score={anomaly_score:.4f} | nodes={pyg_data.num_nodes}"
            )
            with self._metrics_lock:
                self._total_anomalies_detected += 1
            self._run_defense_cycle(
                window_num=window_num,
                pyg_data=pyg_data,
                anomaly_score=anomaly_score,
            )
        else:
            log.info(
                f"[Window #{window_num}] ✅ NORMAL | "
                f"score={anomaly_score:.4f} | nodes={pyg_data.num_nodes}"
            )

    def _run_defense_cycle(
        self,
        window_num:    int,
        pyg_data:      Any,
        anomaly_score: float,
    ) -> None:
        """
        Full defense execution cycle triggered on anomaly detection.

        For each suspicious node in the graph (nodes with high out-degree acting
        as traffic sources), the MARL engine selects and executes a mitigation
        action, then the result is committed to the cryptographic audit ledger.

        Args:
            window_num:    Current window index (for logging).
            pyg_data:      PyTorch Geometric Data object from GraphBuilder.
            anomaly_score: GNN-produced anomaly confidence score.
        """
        # ------------------------------------------------------------------
        # Identify candidate threat nodes from the graph topology.
        # Strategy: nodes with the highest out-degree are likely traffic sources
        # (originators of anomalous connections). We process the top-N.
        # ------------------------------------------------------------------
        num_nodes  = pyg_data.num_nodes
        edge_index = pyg_data.edge_index          # shape: [2, E]
        x_features = pyg_data.x                   # shape: [N, 3] — [act, in_deg, out_deg]

        # Build per-node out-degree from edge_index source indices
        if edge_index.shape[1] > 0:
            # Count outgoing edges per node
            src_nodes = edge_index[0]
            out_degrees_tensor = torch.zeros(num_nodes, dtype=torch.float)
            for src in src_nodes:
                out_degrees_tensor[src.item()] += 1

            # Top-N suspicious nodes by out-degree (max 3 per window)
            n_candidates = min(3, num_nodes)
            _, top_indices = torch.topk(out_degrees_tensor, k=n_candidates)
            candidate_indices = top_indices.tolist()
        else:
            # No edges — treat all nodes as candidates (small graph)
            candidate_indices = list(range(min(3, num_nodes)))

        # ------------------------------------------------------------------
        # Process each candidate node
        # ------------------------------------------------------------------
        for node_idx in candidate_indices:
            # Extract node features: [activity_count, in_degree, out_degree]
            node_feat   = x_features[node_idx]
            act_count   = float(node_feat[0].item())
            in_degree   = float(node_feat[1].item())
            out_degree  = float(node_feat[2].item())

            # Synthesise target IP and port from edge_index.
            # In live operation these come from event_data["dest_ip"] / ["dest_port"];
            # here we reconstruct a representative target from the graph structure.
            # The GraphBuilder encodes dest nodes as "dest_ip:port" string ids.
            target_ip, target_port = self._resolve_target(
                node_idx=node_idx,
                edge_index=edge_index,
                num_nodes=num_nodes,
            )

            # Map node index → container agent identifier.
            # Uses a round-robin over registered container names so that each
            # suspicious node is assigned to an independent MARL agent.
            container_list = list(KNOWN_CONTAINER_NODES)
            agent_id = container_list[node_idx % len(container_list)]

            log.info(
                f"[Defense] Window #{window_num} | Node[{node_idx}] "
                f"act={act_count:.0f} out={out_degree:.0f} → "
                f"agent='{agent_id}' target={target_ip}:{target_port}"
            )

            # --- MARL: select action + execute OS command ---
            try:
                audit_payload = self._marl_engine.apply_mitigation(
                    node_id=agent_id,
                    anomaly_score=anomaly_score,
                    target_ip=target_ip,
                    target_port=target_port,
                    activity_count=act_count,
                    in_degree=in_degree,
                    out_degree=out_degree,
                )
            except Exception as exc:
                log.error(
                    f"[Defense] MARL apply_mitigation failed for node {node_idx}: {exc}"
                )
                continue

            # Enrich audit payload with orchestrator metadata
            audit_payload["window_index"] = window_num
            audit_payload["node_index"]   = node_idx
            audit_payload["graph_nodes"]  = num_nodes
            audit_payload["graph_edges"]  = int(edge_index.shape[1])

            # --- Ledger: commit to SHA-256 cryptographic chain ---
            try:
                block_hash = self._ledger.append_event(audit_payload)
                log.info(
                    f"[Ledger] Block committed | "
                    f"action={audit_payload['action_taken']} | "
                    f"agent='{agent_id}' | hash={block_hash[:16]}..."
                )
            except Exception as exc:
                log.error(f"[Ledger] Failed to commit block: {exc}")
                continue

            with self._metrics_lock:
                self._total_mitigations_applied += 1

            # Print a prominent defense summary line to stdout
            self._print_defense_banner(
                window_num=window_num,
                agent_id=agent_id,
                anomaly_score=anomaly_score,
                action_taken=audit_payload["action_taken"],
                target_ip=target_ip,
                target_port=target_port,
                block_hash=block_hash,
            )

    # ------------------------------------------------------------------
    # Helper: Target Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_target(
        node_idx: int,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple:
        """
        Resolves a representative target IP and port for a given graph node.
        In live mode these values come directly from the eBPF event payload;
        here we use the edge_index to identify the most common destination
        node connected to node_idx and decode from docker-compose IP ranges.

        Args:
            node_idx:   Source node index to resolve.
            edge_index: [2, E] tensor (source, dest).
            num_nodes:  Total number of nodes.

        Returns:
            Tuple of (target_ip: str, target_port: int).
        """
        # Default fallback (used when no outgoing edges found)
        _DEFAULT_SUBNET = "172.18.0"
        _COMMON_PORTS = [80, 5432, 8080, 443]

        if edge_index.shape[1] == 0:
            return f"{_DEFAULT_SUBNET}.{(node_idx % 4) + 2}", _COMMON_PORTS[0]

        # Find all destination nodes that node_idx connects to
        src_mask   = (edge_index[0] == node_idx)
        dest_nodes = edge_index[1][src_mask]

        if len(dest_nodes) == 0:
            return f"{_DEFAULT_SUBNET}.{(node_idx % 4) + 2}", _COMMON_PORTS[0]

        # Most frequently targeted destination node
        dest_idx   = int(dest_nodes.mode().values.item())
        target_ip  = f"{_DEFAULT_SUBNET}.{(dest_idx % 4) + 2}"
        target_port = _COMMON_PORTS[dest_idx % len(_COMMON_PORTS)]
        return target_ip, target_port

    # ------------------------------------------------------------------
    # Signal Handler
    # ------------------------------------------------------------------

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """Handles SIGINT (Ctrl+C) and SIGTERM for graceful shutdown."""
        sig_name = signal.Signals(signum).name
        log.info(f"\n[Signal] Received {sig_name} — initiating graceful shutdown...")
        self.shutdown()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Display Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_defense_banner(
        window_num:    int,
        agent_id:      str,
        anomaly_score: float,
        action_taken:  str,
        target_ip:     str,
        target_port:   int,
        block_hash:    str,
    ) -> None:
        """Prints a structured, human-readable defense action banner."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        banner = (
            f"\n{'━' * 65}\n"
            f"  🚨 ZERO-TRUST DEFENSE ACTION EXECUTED\n"
            f"{'━' * 65}\n"
            f"  Window    : #{window_num}\n"
            f"  Timestamp : {ts}\n"
            f"  Agent     : {agent_id}\n"
            f"  GNN Score : {anomaly_score:.6f}\n"
            f"  Action    : {action_taken}\n"
            f"  Target    : {target_ip}:{target_port}\n"
            f"  Ledger    : Block committed — SHA-256: {block_hash[:32]}...\n"
            f"{'━' * 65}\n"
        )
        print(banner, flush=True)

    def _print_session_summary(self) -> None:
        """Prints aggregate pipeline metrics at session end."""
        elapsed = time.monotonic() - self._pipeline_start_time
        with self._metrics_lock:
            summary = (
                f"\n{'═' * 65}\n"
                f"  📊 NetShield-MARL SESSION SUMMARY\n"
                f"{'═' * 65}\n"
                f"  Runtime              : {elapsed:.1f}s\n"
                f"  Events Ingested      : {self._total_events_ingested}\n"
                f"  Windows Evaluated    : {self._total_windows_evaluated}\n"
                f"  Anomalies Detected   : {self._total_anomalies_detected}\n"
                f"  Mitigations Applied  : {self._total_mitigations_applied}\n"
                f"  MARL Agents Active   : {len(self._marl_engine.list_agents())}\n"
                f"  Mode                 : {'DRY-RUN' if self.dry_run else 'LIVE'}\n"
                f"{'═' * 65}\n"
            )
        print(summary, flush=True)

        # Print per-agent summary
        for agent_id in self._marl_engine.list_agents():
            summary_data = self._marl_engine.get_agent_summary(agent_id)
            if summary_data and summary_data["step_count"] > 0:
                log.info(
                    f"[Agent] '{agent_id}': "
                    f"steps={summary_data['step_count']} | "
                    f"episode_R={summary_data['episode_reward']:+.2f} | "
                    f"last_action={summary_data['last_action']}"
                )


# ===========================================================================
# Entry Point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  🛡️  NetShield-MARL: Autonomous Zero-Trust Security System")
    print("  B.Tech AIML Major Project — Phase 3+4 Integration")
    print("=" * 65)
    print(f"  DRY_RUN      : {DRY_RUN}")
    print(f"  Window       : {GRAPH_WINDOW_SEC}s")
    print(f"  Threshold    : {ANOMALY_TRIGGER_THRESHOLD}")
    print(f"  Ledger DB    : {LEDGER_DB_PATH}")
    print(f"  Queue Size   : {QUEUE_MAX_SIZE}")
    print("=" * 65)

    if not DRY_RUN and os.geteuid() != 0:
        print("\n❌  ERROR: Live mode (NETSHIELD_DRY_RUN=false) requires root privileges.")
        print("   Rerun with: sudo -E python3 main_pipeline.py")
        sys.exit(1)

    orchestrator = NetShieldOrchestrator(
        graph_window_sec=GRAPH_WINDOW_SEC,
        anomaly_threshold=ANOMALY_TRIGGER_THRESHOLD,
        ledger_db_path=LEDGER_DB_PATH,
        policy_weights=POLICY_WEIGHTS_PATH,
        dry_run=DRY_RUN,
    )

    # start() blocks on the main consumer loop until Ctrl+C / SIGTERM
    orchestrator.start()
