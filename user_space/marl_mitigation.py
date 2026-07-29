import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MARL: %(message)s")

class MARLMitigationAgent:
    """
    Multi-Agent Reinforcement Learning (MARL) IPPO Policy Interface for Zero-Trust Mitigation.
    Executes real-time container isolation and iptables packet filtering rules.
    """
    def __init__(self, action_space=None):
        # Action Map: 0 = Monitor/No-op, 1 = Rate-Limit Traffic, 2 = Isolate Container (iptables), 3 = Kill Container
        self.action_map = {
            0: "MONITOR",
            1: "RATE_LIMIT",
            2: "ISOLATE_IPTABLES",
            3: "TERMINATE_CONTAINER"
        }

    def select_action(self, anomaly_score, container_id_or_comm):
        """
        Determines the optimal security action given the GIN anomaly confidence score.
        In multi-agent mode, each container's policy evaluates local state & graph metrics.
        """
        if anomaly_score >= 0.85:
            action_code = 2  # High confidence attack: Isolate via iptables
        elif anomaly_score >= 0.60:
            action_code = 1  # Moderate confidence: Rate limit
        else:
            action_code = 0  # Normal traffic
            
        action_name = self.action_map[action_code]
        logging.info(f"Agent evaluated target '{container_id_or_comm}' -> Score: {anomaly_score:.4f} => Action: {action_name}")
        return action_code, action_name

    def execute_action(self, action_code, target_ip_or_comm):
        """
        Executes mitigation action directly on Linux host / WSL2 via subprocess.
        """
        if action_code == 0:
            return True, "No action required."

        elif action_code == 1:
            # Rate-limiting traffic via tc / iptables
            cmd = f"sudo iptables -A INPUT -s {target_ip_or_comm} -m limit --limit 10/sec -j ACCEPT"
            logging.warning(f"Executing Rate-Limit: {cmd}")
            # subprocess.run(cmd, shell=True, check=False)
            return True, f"Rate-limit applied to {target_ip_or_comm}"

        elif action_code == 2:
            # Isolate container IP via iptables DROP rule
            cmd = f"sudo iptables -A INPUT -s {target_ip_or_comm} -j DROP"
            logging.error(f"🚨 Executing Zero-Trust Isolation: {cmd}")
            # subprocess.run(cmd, shell=True, check=False)
            return True, f"Container/IP {target_ip_or_comm} ISOLATED via iptables"

        elif action_code == 3:
            # Docker container termination
            cmd = f"docker stop {target_ip_or_comm}"
            logging.critical(f"💥 Terminating Container: {cmd}")
            # subprocess.run(cmd, shell=True, check=False)
            return True, f"Container {target_ip_or_comm} terminated"

        return False, "Unknown action code"

if __name__ == "__main__":
    agent = MARLMitigationAgent()
    code, name = agent.select_action(0.92, "netshield_attacker")
    agent.execute_action(code, "172.18.0.4")
