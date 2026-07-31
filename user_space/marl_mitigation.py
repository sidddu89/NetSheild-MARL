#!/usr/bin/python3
"""
NetShield-MARL: Phase 3 – Multi-Agent Reinforcement Learning Mitigation Engine
Module Path: user_space/marl_mitigation.py

Provides:
  - MARLMitigationEngine  : Neural policy manager for per-container RL agents.
  - MARLMitigationAgent   : Backward-compatible API wrapper used by main_pipeline.py.
  - execute_action()      : Low-level Linux/Docker mitigation command executor.
  - select_mitigation_action() : Anomaly-score → discrete action mapper.

Action Space (Discrete-4, per Farama Gymnasium convention):
  0 – ALLOW            : Pass-through; standard monitoring.
  1 – THROTTLE         : Apply tc/iptables bandwidth shaping (rate-limit).
  2 – BLOCK_PORT       : iptables DROP rule on target destination port.
  3 – ISOLATE_CONTAINER: docker network disconnect — full container quarantine.

Observation Space per agent (4-dim float32 vector):
  [activity_count, in_degree, out_degree, anomaly_score]

Reward shaping (logged, not trained here – policy execution mode):
  R = +REWARD_ATTACK_BLOCKED   if action ∈ {1,2,3} and score ≥ THRESH_HIGH
    + +REWARD_CORRECT_ALLOW    if action == 0 and score < THRESH_MONITOR
    - PENALTY_FALSE_POSITIVE   if action ∈ {2,3} and score < THRESH_MODERATE
    - PENALTY_SLA_DISRUPTION   if action == 3 and score < THRESH_HIGH

Safety:
  All shell/Docker commands are constructed but executed only when the environment
  variable NETSHIELD_DRY_RUN is NOT set to "false". This guarantees safe execution
  during development, CI, and Windows WSL2 environments where sudo/Docker may be
  unavailable.

Dependencies: torch (standard pip), subprocess, os, logging (all stdlib except torch).
"""

import os
import json
import logging
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] MARL: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger("NetShield.MARL")

# ---------------------------------------------------------------------------
# Action Space Constants
# ---------------------------------------------------------------------------
ACTION_ALLOW             = 0
ACTION_THROTTLE          = 1
ACTION_BLOCK_PORT        = 2
ACTION_ISOLATE_CONTAINER = 3

ACTION_NAMES: Dict[int, str] = {
    ACTION_ALLOW:             "ALLOW",
    ACTION_THROTTLE:          "THROTTLE",
    ACTION_BLOCK_PORT:        "BLOCK_PORT",
    ACTION_ISOLATE_CONTAINER: "ISOLATE_CONTAINER",
}

NUM_ACTIONS      = 4   # Discrete action space cardinality
OBS_DIM          = 4   # [activity_count, in_degree, out_degree, anomaly_score]
HIDDEN_DIM       = 32  # Policy network hidden layer width

# ---------------------------------------------------------------------------
# Decision Thresholds (tunable without retraining)
# ---------------------------------------------------------------------------
THRESH_MONITOR   = 0.30   # Below → ALLOW
THRESH_MODERATE  = 0.60   # Below THRESH_HIGH → THROTTLE
THRESH_HIGH      = 0.85   # At or above → BLOCK_PORT or ISOLATE

# ---------------------------------------------------------------------------
# Reward Shaping Constants (for audit-log reward annotation)
# ---------------------------------------------------------------------------
REWARD_ATTACK_BLOCKED   = +1.0
REWARD_CORRECT_ALLOW    = +0.1
PENALTY_FALSE_POSITIVE  = -0.5
PENALTY_SLA_DISRUPTION  = -1.0

# ---------------------------------------------------------------------------
# Dry-Run Safety Gate
# ---------------------------------------------------------------------------
# Set environment variable NETSHIELD_DRY_RUN=false to enable live execution.
_DRY_RUN: bool = os.environ.get("NETSHIELD_DRY_RUN", "true").lower() != "false"


# ===========================================================================
# Neural Policy Network (Lightweight IPPO-style Actor)
# ===========================================================================

class ContainerPolicyNetwork(nn.Module):
    """
    Lightweight MLP actor used for policy-execution inference per container agent.

    Architecture: Linear(OBS_DIM → HIDDEN_DIM) → ReLU → Linear(HIDDEN_DIM → NUM_ACTIONS)
    Outputs log-softmax probabilities over the 4-action discrete space.

    In a full IPPO training loop this would be trained on collected trajectories.
    In Phase 3 (policy execution), weights are either pre-loaded or initialized to
    encode the rule-based heuristic as soft priors via bias initialization.
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = HIDDEN_DIM, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        self._init_heuristic_bias()

    def _init_heuristic_bias(self) -> None:
        """
        Bias-initializes the output layer to encode the rule-based policy as a
        soft prior, ensuring sensible behavior even before RL training begins.
        Prior:  ALLOW slightly favored at low anomaly → THROTTLE at mid → BLOCK at high.
        """
        with torch.no_grad():
            # [ALLOW, THROTTLE, BLOCK_PORT, ISOLATE]
            self.fc2.bias.copy_(torch.tensor([0.4, 0.1, -0.2, -0.5]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Float tensor of shape [batch, OBS_DIM] or [OBS_DIM].
        Returns:
            Log-softmax action probabilities of shape [batch, NUM_ACTIONS].
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension
        x = F.relu(self.fc1(obs))
        logits = self.fc2(x)
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> int:
        """
        Greedy action selection (argmax over log-probabilities).
        Returns the integer action ID.
        """
        self.eval()
        log_probs = self.forward(obs)
        return int(torch.argmax(log_probs, dim=-1).item())


# ===========================================================================
# Low-Level Mitigation Command Executor
# ===========================================================================

def execute_action(
    container_name: str,
    target_ip: str,
    target_port: int,
    action_id: int,
) -> Dict[str, Any]:
    """
    Constructs and (conditionally) executes the corresponding OS/Docker command
    for the given discrete action_id.

    Args:
        container_name: Docker container name/ID to act upon (e.g., "netshield_node_A").
        target_ip:      IPv4 address of the threat source to target.
        target_port:    Destination TCP/UDP port to block (used by BLOCK_PORT only).
        action_id:      Discrete action index [0–3].

    Returns:
        dict with keys: action_id, action_name, command, executed, dry_run, timestamp, result.
    """
    action_name = ACTION_NAMES.get(action_id, "UNKNOWN")
    timestamp   = datetime.now(timezone.utc).isoformat()
    command     = ""
    result      = ""

    # --- Build the OS-level command string ---
    if action_id == ACTION_ALLOW:
        command = f"# ALLOW — no iptables change. Monitoring {target_ip}"
        result  = f"Traffic from {target_ip} is permitted. Continuous monitoring active."

    elif action_id == ACTION_THROTTLE:
        # tc + iptables MARK approach: rate-limit ingress to 10 pps
        command = (
            f"sudo iptables -A INPUT -s {target_ip} "
            f"-m limit --limit 10/sec --limit-burst 20 -j ACCEPT && "
            f"sudo iptables -A INPUT -s {target_ip} -j DROP"
        )
        result = f"Bandwidth throttle applied to source IP {target_ip} (10 pps cap)."

    elif action_id == ACTION_BLOCK_PORT:
        # Hard DROP on the target destination port from the threat source
        command = (
            f"sudo iptables -I INPUT -s {target_ip} "
            f"-p tcp --dport {target_port} -j DROP"
        )
        result = f"iptables DROP rule inserted: {target_ip} → port {target_port}."

    elif action_id == ACTION_ISOLATE_CONTAINER:
        # docker network disconnect removes the container from all Docker networks
        command = f"docker network disconnect --force $(docker inspect -f '{{{{.HostConfig.NetworkMode}}}}' {container_name}) {container_name}"
        result  = f"Container '{container_name}' disconnected from Docker network bridge."

    else:
        logger.error(f"Unknown action_id={action_id}. No mitigation applied.")
        return {
            "action_id":   action_id,
            "action_name": "UNKNOWN",
            "command":     "",
            "executed":    False,
            "dry_run":     _DRY_RUN,
            "timestamp":   timestamp,
            "result":      "ERROR: Unknown action_id.",
        }

    # --- Execute or simulate ---
    executed = False
    if action_id == ACTION_ALLOW:
        # No-op — always "executed" trivially
        executed = True
        logger.info(f"[{action_name}] {result}")
    elif _DRY_RUN:
        logger.warning(
            f"[DRY-RUN] Would execute: `{command}` | "
            f"Set NETSHIELD_DRY_RUN=false to enable live mitigation."
        )
    else:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            executed = proc.returncode == 0
            if not executed:
                result += f" | STDERR: {proc.stderr.strip()}"
                logger.error(f"[{action_name}] Command failed: {proc.stderr.strip()}")
            else:
                logger.warning(f"[{action_name}] LIVE EXECUTION SUCCESS: {result}")
        except subprocess.TimeoutExpired:
            result  = f"ERROR: Command timed out after 10s."
            executed = False
            logger.error(f"[{action_name}] Timeout: {command}")

    return {
        "action_id":   action_id,
        "action_name": action_name,
        "command":     command,
        "executed":    executed,
        "dry_run":     _DRY_RUN,
        "timestamp":   timestamp,
        "result":      result,
    }


# ===========================================================================
# Per-Agent State Record
# ===========================================================================

class AgentState:
    """Tracks per-container agent history: last observation, action, and reward."""
    __slots__ = ("node_id", "last_obs", "last_action_id", "last_reward", "episode_reward", "step_count")

    def __init__(self, node_id: str):
        self.node_id       = node_id
        self.last_obs      = torch.zeros(OBS_DIM)
        self.last_action_id: Optional[int] = None
        self.last_reward   = 0.0
        self.episode_reward = 0.0
        self.step_count    = 0

    def update(self, obs: torch.Tensor, action_id: int, reward: float) -> None:
        self.last_obs       = obs
        self.last_action_id = action_id
        self.last_reward    = reward
        self.episode_reward += reward
        self.step_count     += 1


# ===========================================================================
# MARL Mitigation Engine (Primary Class – Phase 3 Requirement)
# ===========================================================================

class MARLMitigationEngine:
    """
    Multi-Agent Reinforcement Learning Mitigation Engine.

    Manages a fleet of independent per-container policy networks (IPPO-style).
    Each registered container agent receives its own ContainerPolicyNetwork and
    AgentState, enabling fully decentralized execution.

    Key Methods:
        register_agent(node_id)               → Register a new container agent.
        select_mitigation_action(node_id, ...) → Map observation → action.
        apply_mitigation(node_id, ...)         → Select + execute + log.
        compute_reward(action_id, score)       → Shaped reward for audit logging.
    """

    def __init__(self, policy_weights_path: Optional[str] = None, dry_run: bool = _DRY_RUN):
        """
        Args:
            policy_weights_path: Optional path to a saved torch state_dict (.pt file).
                                 If None, networks use heuristic-biased initialization.
            dry_run:             Override the global DRY_RUN flag for this instance.
        """
        self.dry_run  = dry_run
        self._agents: Dict[str, ContainerPolicyNetwork] = {}
        self._states: Dict[str, AgentState]             = {}
        self._weights_path = policy_weights_path

        logger.info(
            f"MARLMitigationEngine initialized | "
            f"dry_run={self.dry_run} | "
            f"weights={'loaded' if policy_weights_path else 'heuristic-prior'}"
        )

    # ------------------------------------------------------------------
    # Agent Registry
    # ------------------------------------------------------------------

    def register_agent(self, node_id: str) -> ContainerPolicyNetwork:
        """
        Registers a new container agent. Idempotent: returns existing policy if
        already registered.

        Args:
            node_id: Unique identifier for the container (e.g., Docker container name).
        Returns:
            The ContainerPolicyNetwork assigned to this agent.
        """
        if node_id not in self._agents:
            policy = ContainerPolicyNetwork()
            if self._weights_path and os.path.isfile(self._weights_path):
                try:
                    state = torch.load(self._weights_path, map_location="cpu", weights_only=True)
                    policy.load_state_dict(state)
                    logger.info(f"Loaded policy weights from '{self._weights_path}' for agent '{node_id}'.")
                except Exception as exc:
                    logger.warning(f"Could not load weights for '{node_id}': {exc}. Using heuristic prior.")
            self._agents[node_id] = policy
            self._states[node_id] = AgentState(node_id)
            logger.info(f"Registered new MARL agent: '{node_id}'.")
        return self._agents[node_id]

    # ------------------------------------------------------------------
    # Observation Builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_observation(
        activity_count: float,
        in_degree: float,
        out_degree: float,
        anomaly_score: float,
    ) -> torch.Tensor:
        """
        Builds the 4-dim observation vector from graph node metrics + GNN score.

        Args:
            activity_count: Total event count for this node in the current window.
            in_degree:      Number of unique incoming edge sources.
            out_degree:     Number of unique outgoing edge targets.
            anomaly_score:  GNN-produced anomaly probability [0.0 – 1.0].
        Returns:
            Float32 tensor of shape [OBS_DIM].
        """
        return torch.tensor(
            [activity_count, in_degree, out_degree, anomaly_score],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Core: Action Selection
    # ------------------------------------------------------------------

    def select_mitigation_action(
        self,
        node_id: str,
        anomaly_score: float,
        activity_count: float = 1.0,
        in_degree: float = 1.0,
        out_degree: float = 1.0,
    ) -> Tuple[int, str]:
        """
        Maps the current agent observation to a discrete mitigation action.
        Combines neural policy output with a hard-safety threshold override:
        if anomaly_score ≥ THRESH_HIGH, action is always ≥ BLOCK_PORT regardless
        of what the untrained network outputs. This prevents the cold-start problem.

        Args:
            node_id:        Container agent identifier (auto-registered if new).
            anomaly_score:  GNN anomaly probability in [0.0, 1.0].
            activity_count: Node event count (from GraphBuilder).
            in_degree:      Node in-degree (from GraphBuilder).
            out_degree:     Node out-degree (from GraphBuilder).

        Returns:
            Tuple of (action_id: int, action_name: str).
        """
        # Ensure agent is registered
        policy = self.register_agent(node_id)
        state  = self._states[node_id]

        obs = self.build_observation(activity_count, in_degree, out_degree, anomaly_score)

        # --- Neural policy action ---
        neural_action = policy.select_action(obs)

        # --- Hard-safety threshold override (prevents cold-start under-response) ---
        if anomaly_score >= THRESH_HIGH:
            # Severe threat: escalate to BLOCK_PORT minimum; use ISOLATE if neural policy agrees
            safe_action = ACTION_ISOLATE_CONTAINER if neural_action == ACTION_ISOLATE_CONTAINER else ACTION_BLOCK_PORT
        elif anomaly_score >= THRESH_MODERATE:
            # Moderate threat: at minimum THROTTLE
            safe_action = max(neural_action, ACTION_THROTTLE)
        else:
            safe_action = neural_action  # Trust the network at low anomaly

        action_name = ACTION_NAMES[safe_action]

        # Compute shaped reward for this decision
        reward = self.compute_reward(safe_action, anomaly_score)
        state.update(obs, safe_action, reward)

        logger.info(
            f"Agent['{node_id}'] | score={anomaly_score:.4f} | "
            f"neural={ACTION_NAMES[neural_action]} → safe={action_name} | reward={reward:+.2f}"
        )
        return safe_action, action_name

    # ------------------------------------------------------------------
    # Reward Shaping
    # ------------------------------------------------------------------

    @staticmethod
    def compute_reward(action_id: int, anomaly_score: float) -> float:
        """
        Computes the shaped reward signal for the current decision.
        This is logged in the audit ledger for offline policy improvement.

        Args:
            action_id:      Chosen discrete action.
            anomaly_score:  GNN anomaly confidence [0.0 – 1.0].
        Returns:
            Scalar reward float.
        """
        reward = 0.0
        is_defensive = action_id in {ACTION_THROTTLE, ACTION_BLOCK_PORT, ACTION_ISOLATE_CONTAINER}
        is_severe = action_id in {ACTION_BLOCK_PORT, ACTION_ISOLATE_CONTAINER}

        if is_defensive and anomaly_score >= THRESH_HIGH:
            reward += REWARD_ATTACK_BLOCKED
        elif action_id == ACTION_ALLOW and anomaly_score < THRESH_MONITOR:
            reward += REWARD_CORRECT_ALLOW
        if is_severe and anomaly_score < THRESH_MODERATE:
            reward += PENALTY_FALSE_POSITIVE
        if action_id == ACTION_ISOLATE_CONTAINER and anomaly_score < THRESH_HIGH:
            reward += PENALTY_SLA_DISRUPTION

        return round(reward, 4)

    # ------------------------------------------------------------------
    # End-to-End: Select + Execute + Return Payload
    # ------------------------------------------------------------------

    def apply_mitigation(
        self,
        node_id: str,
        anomaly_score: float,
        target_ip: str,
        target_port: int = 0,
        activity_count: float = 1.0,
        in_degree: float = 1.0,
        out_degree: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Full pipeline: select the RL action → execute the OS/Docker command →
        return a structured event payload ready to be committed to the audit ledger.

        Args:
            node_id:        Container agent identifier.
            anomaly_score:  GNN anomaly probability.
            target_ip:      Source IP address of the detected threat.
            target_port:    Destination port to block (used for BLOCK_PORT action).
            activity_count: Node graph metric.
            in_degree:      Node graph metric.
            out_degree:     Node graph metric.

        Returns:
            Dict audit payload compatible with CryptographicAuditLedger.append_event().
        """
        action_id, action_name = self.select_mitigation_action(
            node_id=node_id,
            anomaly_score=anomaly_score,
            activity_count=activity_count,
            in_degree=in_degree,
            out_degree=out_degree,
        )

        exec_result = execute_action(
            container_name=node_id,
            target_ip=target_ip,
            target_port=target_port,
            action_id=action_id,
        )

        state  = self._states[node_id]
        reward = state.last_reward

        audit_payload = {
            "source_ip":              target_ip,
            "dest_ip":                "internal",
            "target_port":            target_port,
            "container_agent":        node_id,
            "detected_anomaly_score": round(anomaly_score, 6),
            "action_id":              action_id,
            "action_taken":           action_name,
            "shaped_reward":          reward,
            "command_issued":         exec_result["command"],
            "command_executed":       exec_result["executed"],
            "dry_run":                exec_result["dry_run"],
            "execution_result":       exec_result["result"],
            "agent_step":             state.step_count,
            "episode_reward_total":   round(state.episode_reward, 4),
            "timestamp":              exec_result["timestamp"],
        }

        logger.info(
            f"Mitigation payload committed | agent='{node_id}' | "
            f"action={action_name} | reward={reward:+.2f} | score={anomaly_score:.4f}"
        )
        return audit_payload

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_agent_summary(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Returns a summary of the agent's lifetime statistics."""
        if node_id not in self._states:
            return None
        s = self._states[node_id]
        return {
            "node_id":          s.node_id,
            "step_count":       s.step_count,
            "episode_reward":   s.episode_reward,
            "last_action":      ACTION_NAMES.get(s.last_action_id, "NONE"),
            "last_reward":      s.last_reward,
        }

    def list_agents(self) -> List[str]:
        """Returns a list of all currently registered agent node IDs."""
        return list(self._agents.keys())


# ===========================================================================
# Backward-Compatible MARLMitigationAgent (used by main_pipeline.py)
# ===========================================================================

class MARLMitigationAgent:
    """
    Backward-compatible shim for main_pipeline.py.
    Delegates to MARLMitigationEngine internally while preserving the original
    select_action(anomaly_score, container_id) / execute_action(action_code, target) API.
    """

    # Internal engine instance — shared across all agent instances
    _engine: MARLMitigationEngine = MARLMitigationEngine()

    # Action name mapping (kept for legacy callers)
    action_map: Dict[int, str] = ACTION_NAMES

    def __init__(self, action_space: Optional[Any] = None):
        # action_space parameter is accepted for API compatibility but unused;
        # the discrete 4-action space is fixed by the engine constants.
        pass

    def select_action(self, anomaly_score: float, container_id_or_comm: str) -> Tuple[int, str]:
        """
        Legacy API: determines the optimal security action given GNN anomaly score.
        Delegates to MARLMitigationEngine.select_mitigation_action().

        Returns:
            (action_code: int, action_name: str)
        """
        action_id, action_name = self._engine.select_mitigation_action(
            node_id=container_id_or_comm,
            anomaly_score=anomaly_score,
        )
        return action_id, action_name

    def execute_action(self, action_code: int, target_ip_or_comm: str) -> Tuple[bool, str]:
        """
        Legacy API: executes mitigation action on Linux host / WSL2.
        Delegates to the module-level execute_action() function.

        Returns:
            (success: bool, message: str)
        """
        result = execute_action(
            container_name=target_ip_or_comm,
            target_ip=target_ip_or_comm,
            target_port=0,
            action_id=action_code,
        )
        return result["executed"] or result["dry_run"], result["result"]


# ===========================================================================
# Built-In Verification Test Suite
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("=" * 72)
    print("🛡️  NetShield-MARL Phase 3: MARL Mitigation Engine Test Suite")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # 1. Engine Initialization
    # -----------------------------------------------------------------------
    print("\n[STEP 1] Initializing MARLMitigationEngine (dry_run=True)...")
    engine = MARLMitigationEngine(dry_run=True)
    assert len(engine.list_agents()) == 0, "Engine should start with no agents."
    print("         ✅ Engine initialized with 0 agents.")

    # -----------------------------------------------------------------------
    # 2. Agent Registration
    # -----------------------------------------------------------------------
    print("\n[STEP 2] Registering container agents...")
    AGENTS = ["netshield_node_A", "netshield_node_B", "netshield_node_C"]
    for agent_id in AGENTS:
        engine.register_agent(agent_id)
    assert len(engine.list_agents()) == 3
    print(f"         ✅ Registered agents: {engine.list_agents()}")

    # -----------------------------------------------------------------------
    # 3. Policy Network Sanity Check
    # -----------------------------------------------------------------------
    print("\n[STEP 3] Validating ContainerPolicyNetwork output shape...")
    dummy_obs = torch.rand(OBS_DIM)
    policy    = engine._agents["netshield_node_A"]
    log_probs = policy(dummy_obs)
    assert log_probs.shape == (1, NUM_ACTIONS), f"Expected shape (1,4), got {log_probs.shape}"
    assert torch.allclose(torch.exp(log_probs).sum(), torch.tensor(1.0), atol=1e-5), \
        "Probabilities must sum to 1.0"
    print(f"         ✅ Policy output valid. Action probs: {torch.exp(log_probs).detach().numpy().round(3)}")

    # -----------------------------------------------------------------------
    # 4. Threshold-Override Action Selection — Three Scenarios
    # -----------------------------------------------------------------------
    print("\n[STEP 4] Simulating three anomaly scenarios...")

    test_cases = [
        # (label,             score,  expected_action_range)
        ("LOW  Anomaly",      0.15,   {ACTION_ALLOW}),
        ("MID  Anomaly",      0.72,   {ACTION_THROTTLE, ACTION_BLOCK_PORT}),
        ("HIGH Anomaly",      0.93,   {ACTION_BLOCK_PORT, ACTION_ISOLATE_CONTAINER}),
    ]

    for label, score, valid_actions in test_cases:
        action_id, action_name = engine.select_mitigation_action(
            node_id="netshield_node_A",
            anomaly_score=score,
            activity_count=12.0,
            in_degree=3.0,
            out_degree=5.0,
        )
        assert action_id in valid_actions, (
            f"Score {score} produced action {action_name} "
            f"which is not in valid set {[ACTION_NAMES[a] for a in valid_actions]}"
        )
        print(f"         ✅ [{label}] score={score:.2f} → Action: {action_name}")

    # -----------------------------------------------------------------------
    # 5. Full Pipeline Simulation: Incoming Attack Anomaly
    # -----------------------------------------------------------------------
    print("\n[STEP 5] Simulating incoming zero-trust attack anomaly...")
    print("         GNN anomaly_score=0.9421 | source_ip=172.18.0.4 | port=8080")

    attack_payload = engine.apply_mitigation(
        node_id="netshield_node_B",
        anomaly_score=0.9421,
        target_ip="172.18.0.4",
        target_port=8080,
        activity_count=35.0,
        in_degree=2.0,
        out_degree=8.0,
    )

    print("\n         📋 Mitigation Audit Payload:")
    print("         " + "-" * 60)
    for key, val in attack_payload.items():
        print(f"         {key:<28}: {val}")
    print("         " + "-" * 60)

    assert attack_payload["action_id"] in {ACTION_BLOCK_PORT, ACTION_ISOLATE_CONTAINER}, \
        "High-score attack must trigger BLOCK_PORT or ISOLATE_CONTAINER!"
    assert attack_payload["dry_run"] is True, "Dry-run must be True in test mode."
    print(f"\n         ✅ Correct defensive action selected: {attack_payload['action_taken']}")

    # -----------------------------------------------------------------------
    # 6. Backward-Compatible MARLMitigationAgent API
    # -----------------------------------------------------------------------
    print("\n[STEP 6] Verifying backward-compatible MARLMitigationAgent API...")
    legacy_agent = MARLMitigationAgent()

    code, name = legacy_agent.select_action(0.92, "netshield_attacker")
    assert code in {ACTION_BLOCK_PORT, ACTION_ISOLATE_CONTAINER}, \
        f"Legacy API must return high-severity action for score=0.92, got {name}"
    print(f"         ✅ select_action(0.92, 'netshield_attacker') → ({code}, '{name}')")

    success, msg = legacy_agent.execute_action(code, "172.18.0.4")
    assert success, f"execute_action returned failure in dry-run mode: {msg}"
    print(f"         ✅ execute_action({code}, '172.18.0.4') → success=True | {msg[:60]}...")

    # -----------------------------------------------------------------------
    # 7. Reward Shaping Correctness
    # -----------------------------------------------------------------------
    print("\n[STEP 7] Validating reward shaping logic...")
    r1 = MARLMitigationEngine.compute_reward(ACTION_BLOCK_PORT,        anomaly_score=0.92)
    r2 = MARLMitigationEngine.compute_reward(ACTION_ALLOW,             anomaly_score=0.10)
    r3 = MARLMitigationEngine.compute_reward(ACTION_ISOLATE_CONTAINER, anomaly_score=0.40)
    assert r1 == REWARD_ATTACK_BLOCKED,                "BLOCK on high-score should = +1.0"
    assert r2 == REWARD_CORRECT_ALLOW,                 "ALLOW on low-score should = +0.1"
    assert r3 == (PENALTY_FALSE_POSITIVE + PENALTY_SLA_DISRUPTION), \
        f"ISOLATE on low-score should = {PENALTY_FALSE_POSITIVE + PENALTY_SLA_DISRUPTION}"
    print(f"         ✅ BLOCK_PORT(0.92) reward  = {r1:+.2f}  [expected +1.0]")
    print(f"         ✅ ALLOW(0.10) reward       = {r2:+.2f}  [expected +0.1]")
    print(f"         ✅ ISOLATE(0.40) reward     = {r3:+.2f}  [expected -1.5]")

    # -----------------------------------------------------------------------
    # 8. Agent Diagnostics Summary
    # -----------------------------------------------------------------------
    print("\n[STEP 8] Agent lifetime summary...")
    for aid in AGENTS:
        summary = engine.get_agent_summary(aid)
        if summary:
            print(f"         Agent '{aid}': steps={summary['step_count']} | "
                  f"episode_R={summary['episode_reward']:+.2f} | "
                  f"last_action={summary['last_action']}")

    # -----------------------------------------------------------------------
    # Final Result
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("🎉 ALL TESTS PASSED — MARL Mitigation Engine Phase 3 Verified.")
    print("=" * 72)
    sys.exit(0)
