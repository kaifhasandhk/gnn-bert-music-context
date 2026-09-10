import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGEClassifier(nn.Module):
    """Task 2: 2-Layer GraphSAGE GNN Model."""
    def __init__(self, in_channels: int = 104, hidden_dim: int = 128, num_tags: int = 50):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head  = nn.Linear(hidden_dim, num_tags)
        
    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        g = global_mean_pool(h, batch)
        logits = self.head(g)
        return logits, g
