import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """Task 4: Symmetric InfoNCE Loss for Audio Graph <-> Text Contrastive Alignment."""
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, g_emb, t_emb):
        # g_emb: [B, D], t_emb: [B, D]
        g_norm = F.normalize(g_emb, dim=-1)
        t_norm = F.normalize(t_emb, dim=-1)
        
        similarity_matrix = torch.matmul(g_norm, t_norm.T) / self.temperature
        labels = torch.arange(g_emb.size(0)).to(g_emb.device)
        
        loss_a2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2a = F.cross_entropy(similarity_matrix.T, labels)
        return (loss_a2t + loss_t2a) / 2.0
