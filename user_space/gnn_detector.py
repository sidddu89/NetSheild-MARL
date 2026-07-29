import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool

class GINAnomalyDetector(torch.nn.Module):
    def __init__(self, num_node_features=3, hidden_dim=32):
        """
        Graph Isomorphism Network (GIN) for graph-level microservice anomaly detection.
        """
        super(GINAnomalyDetector, self).__init__()
        
        # Layer 1 MLP
        nn1 = Sequential(
            Linear(num_node_features, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn1, train_eps=True)
        
        # Layer 2 MLP
        nn2 = Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(nn2, train_eps=True)
        
        # Output Linear Classifier: 0 (Normal Behavior), 1 (Zero-Trust Anomaly/Attack)
        self.fc = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 2)
        )

    def forward(self, x, edge_index, batch=None):
        # Auto-create batch vector if evaluating a single graph object
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Global Add Pooling aggregates node features into graph embedding
        graph_embed = global_add_pool(x, batch)
        
        # Classification logits
        out = self.fc(graph_embed)
        return F.log_softmax(out, dim=-1)

    @torch.no_grad()
    def predict_anomaly(self, pyg_data, threshold=0.5):
        """
        Inference helper taking a PyG Data object and returning (is_anomaly, anomaly_score).
        """
        self.eval()
        device = next(self.parameters()).device
        x = pyg_data.x.to(device)
        edge_index = pyg_data.edge_index.to(device)
        
        log_probs = self.forward(x, edge_index)
        probs = torch.exp(log_probs)
        anomaly_score = probs[0, 1].item()
        
        is_anomaly = anomaly_score >= threshold
        return is_anomaly, anomaly_score

if __name__ == "__main__":
    from torch_geometric.data import Data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GINAnomalyDetector(num_node_features=3).to(device)
    
    # Mock single graph test
    x = torch.tensor([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    sample_data = Data(x=x, edge_index=edge_index)
    
    is_anomaly, score = model.predict_anomaly(sample_data.to(device))
    print(f"✅ GIN Detector test successful on {device}. Anomaly: {is_anomaly}, Score: {score:.4f}")
