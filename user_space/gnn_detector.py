import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool

class GINAnomalyDetector(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=32):
        super(GINAnomalyDetector, self).__init__()
        
        # Optimized for low VRAM consumption
        nn1 = Sequential(Linear(num_node_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        
        nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        
        # Output layer for binary classification: 0 (Normal) or 1 (Anomaly)
        self.fc = Linear(hidden_dim, 2)

    def forward(self, x, edge_index, batch):
        # Pass node features and edge indices into the GIN convolutional layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Pool the nodes into a single graph-level representation
        x = global_add_pool(x, batch)
        
        # Classify the graph
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

if __name__ == "__main__":
    # Test the model initialization on the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GINAnomalyDetector(num_node_features=3).to(device)
    print(f"✅ GIN Model initialized successfully on: {device}")
