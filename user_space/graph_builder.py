import time
from collections import defaultdict

class GraphBuilder:
    def __init__(self, time_window=5):
        # time_window in seconds
        self.time_window = time_window
        self.edges = defaultdict(int)
        self.nodes = set()
        self.start_time = time.time()

    def add_interaction(self, source_pid, source_comm, dest_info="unknown"):
        """Maps incoming traffic to graph nodes and weighted edges."""
        source_node = f"{source_comm}_{source_pid}"
        self.nodes.add(source_node)
        self.nodes.add(dest_info)
        
        # Edge format: (Source, Dest) -> Weight (Frequency)
        edge = (source_node, dest_info)
        self.edges[edge] += 1

    def get_graph_data(self):
        """Returns the current graph topology and resets the time window."""
        current_time = time.time()
        if current_time - self.start_time >= self.time_window:
            # Package data for PyTorch
            graph_data = {
                "nodes": list(self.nodes),
                "edges": list(self.edges.keys()),
                "weights": list(self.edges.values())
            }
            
            # Reset for the next time window
            self.edges.clear()
            self.nodes.clear()
            self.start_time = current_time
            
            return graph_data
        return None

if __name__ == "__main__":
    print("Graph Builder initialized. Ready to aggregate stream into topological graphs.")
