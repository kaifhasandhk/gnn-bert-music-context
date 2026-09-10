import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import yaml

from gnn_model import AudioGraphSAGE
from fusion_model import CrossAttentionFusion

def train():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Initialize Models
    gnn_model = AudioGraphSAGE(
        in_dim=config["features"]["total_node_dim"],
        hidden_dim=config["model"]["gnn_hidden_dim"],
        out_dim=config["model"]["gnn_hidden_dim"]
    ).to(device)
    
    fusion_model = CrossAttentionFusion(
        d_gnn=config["model"]["gnn_hidden_dim"],
        d_bert=config["model"]["bert_dim"],
        d_k=config["model"]["cross_attn_dim"],
        num_tags=config["dataset"]["num_tags"]
    ).to(device)
    
    optimizer = torch.optim.Adam(
        list(gnn_model.parameters()) + list(fusion_model.parameters()),
        lr=config["training"]["learning_rate"]
    )
    criterion = nn.BCELoss()

    print("Setup complete. Starting training loop...")
    # Training loop implementation loads preprocessed graph tensors from data/processed/graphs/
    
if __name__ == "__main__":
    train()
