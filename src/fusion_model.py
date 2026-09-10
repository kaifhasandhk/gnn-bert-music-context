import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """Task 3: Cross-Attention Multimodal Fusion (GNN Query x BERT Key/Value)."""
    def __init__(self, d_gnn: int = 256, d_bert: int = 768, d_k: int = 128, num_tags: int = 50):
        super(CrossAttentionFusion, self).__init__()
        self.W_q = nn.Linear(d_gnn, d_k)
        self.W_k = nn.Linear(d_bert, d_k)
        self.W_v = nn.Linear(d_bert, d_k)
        self.scale = d_k ** 0.5
        
        self.mlp = nn.Sequential(
            nn.Linear(d_gnn + d_k, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_tags)
        )

    def forward(self, g, h_text):
        # g: Graph vector [B, 256]
        # h_text: BERT tokens [B, L, 768]
        Q = self.W_q(g).unsqueeze(1)    # [B, 1, 128]
        K = self.W_k(h_text)            # [B, L, 128]
        V = self.W_v(h_text)            # [B, L, 128]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        c = torch.matmul(attn, V).squeeze(1) # Context vector [B, 128]
        
        z = torch.cat([g, c], dim=-1)         # Fused embedding z [B, 384]
        out = self.mlp(z)
        return z, torch.sigmoid(out)
