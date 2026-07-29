#!/usr/bin/python3
import os
import sys
import time
import torch

from kernel_space.loader import EBPFLoader
from user_space.graph_builder import GraphBuilder
from user_space.gnn_detector import GINAnomalyDetector
from user_space.marl_mitigation import MARLMitigationAgent
from user_space.audit_ledger import CryptographicAuditLedger

class NetShieldOrchestrator:
    """
    NetShield-MARL Orchestrator: Connects eBPF Data Ingestion -> PyG Graph Aggregation -> 
    GIN Anomaly Detection -> MARL Mitigation Policy -> Cryptographic SQLite Audit Ledger.
    """
    def __init__(self, time_window=3):
        print("🛡️ Initializing NetShield-MARL Autonomous Zero-Trust Defense Pipeline...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Layer 2: Graph Builder & PyG GIN Detector
        self.graph_builder = GraphBuilder(time_window=time_window)
        self.gnn_model = GINAnomalyDetector(num_node_features=3).to(self.device)
        self.gnn_model.eval()
        
        # 2. Layer 3: MARL Agent
        self.marl_agent = MARLMitigationAgent()
        
        # 3. Layer 4: Cryptographic Ledger
        self.ledger = CryptographicAuditLedger("audit_ledger.db")
        
        # 4. Layer 1: eBPF Loader
        self.ebpf_loader = EBPFLoader(callback_fn=self.handle_ebpf_event)

    def handle_ebpf_event(self, event_data):
        """Callback invoked on each eBPF ring buffer event."""
        pid = event_data["pid"]
        comm = event_data["comm"]
        dest_ip = event_data["dest_ip"]
        dest_port = event_data["dest_port"]

        # Add event to real-time graph aggregator
        self.graph_builder.add_interaction(pid, comm, dest_ip, dest_port)
        
        # Check if time window elapsed to run inference pipeline
        pyg_graph = self.graph_builder.get_graph_data()
        if pyg_graph is not None:
            self.run_pipeline_cycle(pyg_graph, comm, f"{dest_ip}:{dest_port}")

    def run_pipeline_cycle(self, pyg_graph, source_comm, target_info):
        """Runs GNN classification -> MARL Action -> Ledger Logging."""
        is_anomaly, anomaly_score = self.gnn_model.predict_anomaly(pyg_graph)
        
        if is_anomaly:
            print(f"\n🚨 [ANOMALY DETECTED] Score: {anomaly_score:.4f} | Target: {source_comm} -> {target_info}")
            action_code, action_name = self.marl_agent.select_action(anomaly_score, source_comm)
            success, msg = self.marl_agent.execute_action(action_code, target_info)
            
            # Commit to SHA-256 Cryptographic Ledger
            self.ledger.log_event(
                event_type="ZERO_TRUST_ANOMALY",
                source_comm=source_comm,
                target_info=target_info,
                action_taken=action_name,
                anomaly_score=anomaly_score
            )
        else:
            print(f"✅ [NORMAL TRAFFIC] Score: {anomaly_score:.4f} | Graph Nodes: {pyg_graph.num_nodes}")

    def start(self):
        """Starts eBPF listener and pipeline loop."""
        self.ebpf_loader.start_listener()

if __name__ == "__main__":
    orchestrator = NetShieldOrchestrator()
    orchestrator.start()
