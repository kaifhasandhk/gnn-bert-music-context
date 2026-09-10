import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool

class AudioGraphSAGE(nn.Module):
    """Task 2: 2-Layer GraphSAGE Spatial GNN Encoder."""
    def __init__(self, in_dim: int = 152, hidden_dim: int = 256, out_dim: int = 256, num_tags: int = 50):
        super(AudioGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr='mean')
        self.classifier = nn.Linear(out_dim, num_tags)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        g = global_mean_pool(h, batch) # Mean graph pooling
        probs = torch.sigmoid(self.classifier(g))
        return g, probs

class AudioGAT(nn.Module):
    """Optional Graph Attention Network variant."""
    def __init__(self, in_dim: int = 152, hidden_dim: int = 256, out_dim: int = 256, heads: int = 4):
        super(AudioGAT, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        g = global_mean_pool(h, batch)
        return g
