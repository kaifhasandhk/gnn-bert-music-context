import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """Task 4: Symmetric InfoNCE Loss for Audio-Text Embeddings."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, g_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        g_norm = F.normalize(g_emb, dim=-1)
        t_norm = F.normalize(t_emb, dim=-1)
        
        sim_matrix = torch.matmul(g_norm, t_norm.T) / self.temperature
        labels = torch.arange(g_emb.size(0), device=g_emb.device)
        
        loss_a2t = F.cross_entropy(sim_matrix, labels)
        loss_t2a = F.cross_entropy(sim_matrix.T, labels)
        return (loss_a2t + loss_t2a) / 2.0
