import torch
import librosa
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from audio_features import segment_audio, extract_segment_features

def build_audio_graph(audio_path: str, sr: int = 22050, num_segments: int = 9, threshold: float = 0.75) -> Data:
    """Implements Algorithm 1: Raw Audio to Segment Graph Data Structure."""
    y, _ = librosa.load(audio_path, sr=sr, duration=29.0)
    segments = segment_audio(y, sr, num_segments)
    
    # Extract 152-dim node feature vectors for each segment node
    node_features = [extract_segment_features(seg, sr) for seg in segments]
    X = torch.tensor(np.array(node_features), dtype=torch.float)
    
    edges = []
    # 1. Temporal sequential edges
    for i in range(num_segments - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
        
    # 2. Cosine similarity threshold edges
    cos_sim = cosine_similarity(X.numpy())
    for i in range(num_segments):
        for j in range(num_segments):
            if i != j and cos_sim[i, j] > threshold:
                edges.append((i, j))
                
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=X, edge_index=edge_index)
