import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import DistilBertModel

class CrossAttentionFusionModel(nn.Module):
    """Task 3: Cross-Attention Multimodal Fusion (Query = GNN, Key/Value = BERT)."""
    def __init__(self, num_tags: int = 50, gnn_in_dim: int = 104, gnn_hidden: int = 128, bert_dim: int = 768):
        super().__init__()
        self.gnn_hidden = gnn_hidden

        self.conv1 = SAGEConv(gnn_in_dim, gnn_hidden)
        self.conv2 = SAGEConv(gnn_hidden, gnn_hidden)

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.q_proj = nn.Linear(gnn_hidden, gnn_hidden)
        self.k_proj = nn.Linear(bert_dim, gnn_hidden)
        self.v_proj = nn.Linear(bert_dim, gnn_hidden)

        self.head = nn.Sequential(
            nn.Linear(gnn_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_tags)
        )

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        g = global_mean_pool(h, batch)

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_text = bert_out.last_hidden_state

        Q = self.q_proj(g).unsqueeze(1)
        K = self.k_proj(h_text)
        V = self.v_proj(h_text)

        attn_weights = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.gnn_hidden), dim=-1)
        context = torch.bmm(attn_weights, V).squeeze(1)

        z = torch.cat([g, context], dim=-1)
        logits = self.head(z)
        return logits, z
