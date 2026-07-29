import time
import torch
from collections import defaultdict
from torch_geometric.data import Data

class GraphBuilder:
    def __init__(self, time_window=5, num_node_features=3):
        """
        Aggregates eBPF socket events into PyTorch Geometric Data graphs over a temporal window.
        """
        self.time_window = time_window
        self.num_node_features = num_node_features
        self.edges = defaultdict(int)
        self.node_activity = defaultdict(int)
        self.nodes = set()
        self.start_time = time.time()

    def add_interaction(self, source_pid, source_comm, dest_ip, dest_port):
        """Maps an eBPF socket event into graph topology."""
        source_node = f"{source_comm}:{source_pid}"
        dest_node = f"{dest_ip}:{dest_port}"
        
        self.nodes.add(source_node)
        self.nodes.add(dest_node)
        
        self.node_activity[source_node] += 1
        self.node_activity[dest_node] += 1
        
        edge = (source_node, dest_node)
        self.edges[edge] += 1

    def get_graph_data(self, force_export=False):
        """
        Converts buffered interaction events into a PyTorch Geometric Data object.
        Returns PyG Data instance if time window elapsed or force_export=True, else None.
        """
        current_time = time.time()
        if not force_export and (current_time - self.start_time < self.time_window):
            return None

        if len(self.nodes) == 0:
            # Reset window timer even on empty activity
            self.start_time = current_time
            return None

        # Build consecutive zero-indexed mapping for PyG
        node_to_idx = {node: i for i, node in enumerate(sorted(list(self.nodes)))}
        num_nodes = len(node_to_idx)

        # Build edge_index tensor [2, E] and edge_attr [E, 1]
        src_indices = []
        dst_indices = []
        edge_weights = []

        for (src, dst), weight in self.edges.items():
            src_indices.append(node_to_idx[src])
            dst_indices.append(node_to_idx[dst])
            edge_weights.append([float(weight)])

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        # Construct Node Feature Matrix x [N, num_node_features]
        # Feature 1: Total Activity Count, Feature 2: In-Degree, Feature 3: Out-Degree
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)
        for (src, dst) in self.edges.keys():
            out_degrees[src] += 1
            in_degrees[dst] += 1

        x_features = []
        for node in sorted(list(self.nodes)):
            act = float(self.node_activity[node])
            in_deg = float(in_degrees[node])
            out_deg = float(out_degrees[node])
            x_features.append([act, in_deg, out_deg])

        x = torch.tensor(x_features, dtype=torch.float)

        # Construct PyTorch Geometric Data Object
        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )

        # Clear buffer for next window
        self.edges.clear()
        self.nodes.clear()
        self.node_activity.clear()
        self.start_time = current_time

        return pyg_data

if __name__ == "__main__":
    gb = GraphBuilder(time_window=1)
    gb.add_interaction(101, "wget", "172.18.0.2", 80)
    gb.add_interaction(101, "wget", "172.18.0.3", 5432)
    time.sleep(1.1)
    graph = gb.get_graph_data()
    print("✅ PyTorch Geometric Graph Generated Successfully:")
    print(graph)
